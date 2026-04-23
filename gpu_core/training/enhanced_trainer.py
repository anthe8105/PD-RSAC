"""
Enhanced SAC Trainer with full paper implementation.

Integrates:
- Semi-MDP duration discounting (paper Eq. 12-13)
- WDRO robust targets (paper Section 4)
- Temperature annealing (paper Eq. 16)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import time

from .trainer import SACTrainer, FleetSACTrainer, TrainingMetrics
from .semi_mdp import SemiMDPHandler
from .wdro import WDROAdversary, WDROConfig, ValueNetwork
from ..config import TrainingConfig, CheckpointConfig, LoggingConfig
from ..networks.sac import SACAgent, FleetSACAgent
from ..features.replay_buffer import GPUReplayBuffer


DEFAULT_WDRO_VALUE_SOURCE_SWITCH_EPISODE = 80
DEFAULT_WDRO_VALUE_TRAIN_STOP_EPISODE = 80


@dataclass
class EnhancedTrainingConfig:
    """Extended config for paper-compliant training."""
    # Base SAC config
    batch_size: int = 256
    gamma: float = 0.99
    gradient_steps: int = 1
    
    # Semi-MDP (paper Eq. 12-13)
    use_semi_mdp: bool = True
    step_duration_minutes: float = 5.0
    avg_speed_kmh: float = 30.0
    
    # WDRO (paper Section 4)
    use_wdro: bool = True
    wdro_rho: float = 0.3           # Ambiguity radius (0.3 = moderate robustness)
    wdro_metric: str = "mag"        # Ground metric: "mag" or "euclidean"
    wdro_rho_target: float = 0.2    # Target risk budget
    wdro_lambda_lr: float = 0.01    # Lambda update rate
    wdro_inner_steps: int = 3       # Inner optimization steps
    wdro_value_source_switch_episode: int = DEFAULT_WDRO_VALUE_SOURCE_SWITCH_EPISODE  # First episode where WDRO uses V_phi instead of exact Eq. 27 value
    wdro_value_train_stop_episode: int = DEFAULT_WDRO_VALUE_TRAIN_STOP_EPISODE    # First episode where V_phi stops training; can equal the switch or be later
    
    # Temperature annealing (paper Eq. 16)
    use_temperature_annealing: bool = True
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    
    # Mixed precision training (performance optimization)
    use_amp: bool = True  # Automatic Mixed Precision
    temperature_decay_episodes: int = 500
    
    use_milp: bool = False


class EnhancedReplayBuffer(GPUReplayBuffer):
    """Replay buffer that stores action durations for Semi-MDP."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional storage for durations
        self._durations = None
    
    def push_with_duration(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
        duration: float = 1.0
    ):
        """Push transition with action duration."""
        # Call parent push
        self.push(state, action, reward, next_state, done)
        
        # Store duration
        if self._durations is None:
            self._durations = torch.ones(self.capacity, device=self.device)
        
        idx = (self._position - 1) % self.capacity
        self._durations[idx] = duration
    
    def sample_with_durations(self, batch_size: int):
        """Sample batch with durations."""
        batch = self.sample(batch_size)
        
        if self._durations is not None and hasattr(batch, 'indices'):
            durations = self._durations[batch.indices]
        else:
            durations = torch.ones(batch_size, device=self.device)
        
        return batch, durations


class EnhancedSACTrainer(SACTrainer):
    """
    SAC Trainer with full paper implementation.
    
    Features:
    - Semi-MDP discounting with variable durations
    - WDRO robust backup targets
    - Temperature annealing
    """
    
    def __init__(
        self,
        agent: SACAgent,
        replay_buffer: GPUReplayBuffer,
        training_config: TrainingConfig,
        enhanced_config: Optional[EnhancedTrainingConfig] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
        device: str = 'cuda',
        adjacency_matrix: Optional[torch.Tensor] = None,
        od_matrix: Optional[torch.Tensor] = None
    ):
        # Mixed precision training
        self.use_amp = enhanced_config.use_amp if enhanced_config else False
        self.scaler = GradScaler(enabled=self.use_amp)

        super().__init__(
            agent, replay_buffer, training_config,
            checkpoint_config, logging_config, device,
            use_amp=self.use_amp
        )
        
        self.enhanced_config = enhanced_config or EnhancedTrainingConfig()
        
        # Initialize Semi-MDP handler
        if self.enhanced_config.use_semi_mdp:
            self.semi_mdp = SemiMDPHandler(
                gamma=training_config.gamma,
                step_duration_minutes=self.enhanced_config.step_duration_minutes,
                avg_speed_kmh=self.enhanced_config.avg_speed_kmh,
                device=device
            )
        else:
            self.semi_mdp = None
        
        # Initialize WDRO adversary
        if self.enhanced_config.use_wdro:
            wdro_config = WDROConfig(
                rho=self.enhanced_config.wdro_rho,
                metric=self.enhanced_config.wdro_metric,
                rho_target=self.enhanced_config.wdro_rho_target,
                lambda_lr=self.enhanced_config.wdro_lambda_lr,
                inner_steps=self.enhanced_config.wdro_inner_steps,
                support_radius=self.enhanced_config.wdro_rho,
            )

            self.wdro = WDROAdversary(
                scenario_dim=agent.num_hexes,
                config=wdro_config,
                device=device,
                adjacency_matrix=adjacency_matrix,
                critic=agent.critic,
                actor=agent.actor,
                build_hex_vehicle_summary_fn=agent._build_hex_vehicle_summary_from_features,
                build_vehicle_counts_fn=agent._build_vehicle_counts,
                khop_neighbor_indices=agent._khop_neighbor_indices,
                khop_neighbor_mask=agent._khop_neighbor_mask,
            )

            # Build OD graph for MAG metric (paper Eq. 14)
            if od_matrix is not None and self.enhanced_config.wdro_metric == "mag":
                self.wdro.mag_metric.build_od_graph(od_matrix)

            # V_φ: lightweight MLP head on shared GCN embeddings (paper Section VI-C)
            critic_module = agent.critic.module if hasattr(agent.critic, 'module') else agent.critic
            gcn_output_dim = critic_module.critic1.gcn_output_dim
            self.value_network = ValueNetwork(
                gcn_output_dim=gcn_output_dim,
                vehicle_feature_dim=agent.vehicle_feature_dim,
                context_dim=agent.context_dim,
            ).to(device)
            self.value_optimizer = torch.optim.Adam(
                self.value_network.parameters(), lr=1e-4
            )
            # Attach to WDRO adversary (used in Phase 2 inner loop)
            self.wdro.value_network = self.value_network
        else:
            self.wdro = None
            self.value_network = None
            self.value_optimizer = None
        
        # Temperature annealing state
        self.current_temperature = self.enhanced_config.initial_temperature

    def _call_critic(
        self,
        critic,
        states_flat: torch.Tensor,
        actions: Optional[torch.Tensor],
        hex_embeddings_q1: torch.Tensor = None,
        hex_embeddings_q2: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call critic via agent's unified interface."""
        return self.agent.call_critic(critic, states_flat, actions,
                                      hex_embeddings_q1=hex_embeddings_q1,
                                      hex_embeddings_q2=hex_embeddings_q2)

    def _compute_critic_hex_embeddings(
        self,
        states_flat: torch.Tensor,
        critic,
    ):
        """Pre-compute GCN hex embeddings for both sub-critics.

        Returns (emb_q1, emb_q2) so the caller can pass them into _call_critic
        and skip the GCN forward on subsequent calls with the same states.
        Returns (None, None) for non-GCN critics.
        """
        if not self.agent.use_gcn_critic:
            return None, None
        hex_features, _, _, _ = self.agent._parse_flat_state(states_flat)
        adj = self.agent._adjacency_matrix
        critic_module = critic.module if hasattr(critic, 'module') else critic
        emb_q1 = critic_module.critic1.gcn(hex_features, adj)
        emb_q2 = critic_module.critic2.gcn(hex_features, adj)
        return emb_q1, emb_q2

    def get_temperature(self) -> float:
        """Get current temperature based on episode count."""
        if not self.enhanced_config.use_temperature_annealing:
            return 1.0
        
        progress = min(1.0, self.episode / self.enhanced_config.temperature_decay_episodes)
        temp = self.enhanced_config.initial_temperature * (1 - progress) + \
               self.enhanced_config.final_temperature * progress
        
        self.current_temperature = temp
        return temp
    
    def _current_wdro_phase(self) -> int:
        if self.episode < self.enhanced_config.wdro_value_source_switch_episode:
            return 1
        if self.episode < self.enhanced_config.wdro_value_train_stop_episode:
            return 2
        return 3

    def _ensure_finite(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite fleet training tensor detected in {name}")
        return tensor

    def _apply_scenario_to_state(
        self,
        states_flat: torch.Tensor,
        xi_scenario: torch.Tensor,
        num_hexes: int,
        hex_feature_dim: int,
        vehicle_dim: int
    ) -> torch.Tensor:
        """
        Apply scenario ξ to state by replacing hex demand features.
        
        This is used for WDRO adversarial optimization where ξ represents
        the full demand scenario (not a perturbation).
        
        Args:
            states_flat: Flattened states [batch, state_dim]
            xi_scenario: Full demand scenario [batch, num_hexes] - HAS GRADIENT
            num_hexes: Number of hexes
            hex_feature_dim: Hex feature dimension
            vehicle_dim: Vehicle feature dimension
        
        Returns:
            States with scenario applied [batch, state_dim]
        """
        batch_size = states_flat.shape[0]
        
        # DEBUG (optional): enable if you need to trace grad flow
        # if False:
        #     print(f"[DEBUG _apply_scenario_to_state]")
        #     print(f"  states_flat: requires_grad={states_flat.requires_grad}, grad_fn={states_flat.grad_fn}")
        #     print(f"  xi_scenario: requires_grad={xi_scenario.requires_grad}, is_leaf={xi_scenario.is_leaf}, grad_fn={xi_scenario.grad_fn}")
        
        # Extract dimensions
        hex_start = vehicle_dim
        hex_end = hex_start + num_hexes * hex_feature_dim
        context_dim = self.agent.context_dim
        
        # CRITICAL FIX: Clone sliced tensors to create new tensors (not views)
        # PyTorch cat() doesn't preserve requires_grad from xi_scenario when 
        # concatenating with VIEWS of non-grad tensors (slicing creates views).
        # Solution: .clone() creates new tensors that can participate in autograd.
        
        # Extract components and CLONE to break view relationship
        vehicle_features = states_flat[:, :vehicle_dim].clone()  # New tensor, can have grad
        hex_flat = states_flat[:, hex_start:hex_end].clone()     # New tensor, can have grad
        context_features = states_flat[:, hex_end:].clone()      # New tensor, can have grad
        
        # Reshape hex features
        hex_features = hex_flat.reshape(batch_size, num_hexes, hex_feature_dim)
        
        # Replace demand feature (column 2) with scenario ξ
        # Hex feature layout: [0]=vehicle_counts, [1]=available, [2]=demand, [3]=station, [4]=avail
        # xi_scenario has gradient (leaf variable from WDRO loop)
        # Other hex features from replay buffer (no grad)
        cols_before = hex_features[:, :, :2].clone()   # columns 0,1 [batch, H, 2]
        cols_after  = hex_features[:, :, 3:].clone()   # columns 3,4 [batch, H, 2]

        # Reconstruct hex features with scenario demand at column 2
        # GRADIENT FLOW: xi_scenario (leaf, requires_grad=True) → unsqueeze → cat → hex_features_new
        hex_features_new = torch.cat([
            cols_before,                    # [batch, H, 2] - No grad (from replay)
            xi_scenario.unsqueeze(-1),      # [batch, H, 1] - LEAF with grad!
            cols_after                      # [batch, H, 2] - No grad (from replay)
        ], dim=-1)  # [batch, H, 5] - requires_grad=True from xi_scenario!
        
        # Concatenate back
        # GRADIENT FLOW: xi_scenario → hex_features_new → reshape → cat → perturbed
        perturbed = torch.cat([
            vehicle_features,                           # No grad
            hex_features_new.reshape(batch_size, -1),  # HAS GRAD from xi_scenario!
            context_features                            # No grad
        ], dim=1)  # perturbed.requires_grad = True ✓
        
        # DEBUG (optional): enable if you need to trace grad flow
        # if False:
        #     print(f"  perturbed: requires_grad={perturbed.requires_grad}, grad_fn={perturbed.grad_fn}")
        
        return perturbed

    def _apply_scenario_structured(
        self,
        states_flat: torch.Tensor,
        xi_scenario: torch.Tensor,
        num_hexes: int,
        hex_feature_dim: int,
        vehicle_dim: int
    ) -> Dict[str, torch.Tensor]:
        """Apply scenario ξ and return structured state components for GCN-based V.

        Returns dict with hex_features [batch, H, 5], vehicle_features [batch, V, vf],
        context_features [batch, ctx].  Gradient flows through xi_scenario → hex_features[:,:,2].
        """
        batch_size = states_flat.shape[0]
        hex_start = vehicle_dim
        hex_end = hex_start + num_hexes * hex_feature_dim

        vehicle_flat = states_flat[:, :vehicle_dim].clone()
        hex_flat = states_flat[:, hex_start:hex_end].clone()
        context_features = states_flat[:, hex_end:].clone()

        hex_features = hex_flat.reshape(batch_size, num_hexes, hex_feature_dim)

        # Replace demand column (2) with xi_scenario — gradient flows through here
        cols_before = hex_features[:, :, :2].clone()
        cols_after = hex_features[:, :, 3:].clone()
        hex_features_new = torch.cat([
            cols_before,
            xi_scenario.unsqueeze(-1),
            cols_after
        ], dim=-1)  # [batch, H, 5]

        vehicle_features = vehicle_flat.reshape(
            batch_size, self.agent.num_vehicles, self.agent.vehicle_feature_dim
        )

        # vehicle_hex_ids: normalized position (col 0) → hex index
        vehicle_hex_ids = (vehicle_features[:, :, 0] * num_hexes).long().clamp(0, num_hexes - 1)

        return {
            'hex_features': hex_features_new,       # [batch, H, 5] — xi at col 2
            'vehicle_features': vehicle_features,   # [batch, V, 16]
            'vehicle_hex_ids': vehicle_hex_ids,     # [batch, V]
            'context_features': context_features,   # [batch, ctx]
        }

    def _apply_scenario_to_fleet_state(
        self,
        next_states: Union[torch.Tensor, Dict[str, torch.Tensor]],
        xi_scenario: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        parsed_next_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Replace hex demand feature (col 2) with adversarial scenario ξ."""
        if parsed_next_state is None:
            hex_features, vehicle_features, _, context_features = self.agent._parse_flat_state(next_states)
        else:
            hex_features, vehicle_features, _, context_features = parsed_next_state

        cols_before = hex_features[:, :, :2].clone()
        cols_after = hex_features[:, :, 3:].clone()
        hex_features_new = torch.cat([
            cols_before,
            xi_scenario.unsqueeze(-1),
            cols_after,
        ], dim=-1)

        return {
            'hex_features': hex_features_new,
            'vehicle_features': vehicle_features,
            'vehicle_hex_ids': vehicle_hex_ids,
            'context_features': context_features,
        }

    def _perturb_state_with_scenario(
        self,
        states_flat: torch.Tensor,
        scenario_perturbation: torch.Tensor,
        num_hexes: int,
        hex_feature_dim: int
    ) -> torch.Tensor:
        """
        DEPRECATED: Use _apply_scenario_to_state instead.
        Perturb state with scenario perturbation for WDRO.
        
        Scenario ξ affects demand features in state.
        For simplicity, add perturbation to hex demand features.
        
        Args:
            states_flat: Flattened states [batch, state_dim]
            scenario_perturbation: Perturbation Δξ [batch, num_hexes]
            num_hexes: Number of hexes
            hex_feature_dim: Hex feature dimension
        
        Returns:
            Perturbed states [batch, state_dim]
        """
        batch_size = states_flat.shape[0]
        
        # Extract dimensions
        vehicle_dim = states_flat.shape[1] - num_hexes * hex_feature_dim - 9  # 9 is context_dim
        hex_start = vehicle_dim
        hex_end = hex_start + num_hexes * hex_feature_dim
        
        # Extract components (no clone to preserve gradient)
        vehicle_features = states_flat[:, :vehicle_dim]
        hex_flat = states_flat[:, hex_start:hex_end]
        context_features = states_flat[:, hex_end:]
        
        # Reshape hex features
        hex_features = hex_flat.reshape(batch_size, num_hexes, hex_feature_dim)
        
        # Perturb first feature (demand) of each hex
        # CRITICAL: Only the perturbation needs gradient, base state is detached
        demand_base = hex_features[:, :, 0].detach()
        demand_perturbed = demand_base + scenario_perturbation  # Gradient flows from scenario_perturbation!
        other_hex_features = hex_features[:, :, 1:].detach()  # Detach other features
        
        # Reconstruct hex features with perturbed demand
        hex_features_perturbed = torch.cat([
            demand_perturbed.unsqueeze(-1),  # [batch, num_hexes, 1] - HAS GRADIENT
            other_hex_features               # [batch, num_hexes, hex_feature_dim-1] - NO GRADIENT
        ], dim=-1)  # [batch, num_hexes, hex_feature_dim]
        
        # Concatenate back (gradient only flows through demand_perturbed -> scenario_perturbation)
        perturbed = torch.cat([
            vehicle_features.detach(),  # No grad
            hex_features_perturbed.reshape(batch_size, -1),  # Gradient from demand_perturbed
            context_features.detach()   # No grad
        ], dim=1)
        
        return perturbed
    
    def train_step_enhanced(
        self,
        adjacency: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Fleet training step with WDRO robust targets active from episode 1."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.config.batch_size, adjacency=adjacency)

        states = batch.states
        next_states = batch.next_states
        rewards = batch.rewards
        dones = batch.dones
        durations = batch.durations

        hex_allocations = batch.hex_allocations
        hex_charge_power = batch.hex_charge_power
        vehicle_hex_ids = batch.vehicle_hex_ids

        hex_features_cur, vehicle_features_cur, parsed_hex_ids_cur, context_cur = self.agent._parse_flat_state(states)
        next_hex, next_veh, next_hex_ids, next_ctx = self.agent._parse_flat_state(next_states)
        if vehicle_hex_ids is None:
            vehicle_hex_ids = parsed_hex_ids_cur

        hex_veh_summary_cur = self.agent._build_hex_vehicle_summary_from_features(
            vehicle_features_cur, vehicle_hex_ids,
        )
        veh_counts_cur = self.agent._build_vehicle_counts(vehicle_hex_ids)
        active_hex_mask_cur = self.agent._build_active_hex_mask(hex_veh_summary_cur)

        rewards_normalized = self.agent._normalize_rewards(rewards)

        wdro_phase = self._current_wdro_phase()
        if self.wdro is not None:
            self.wdro.use_learned_value = (wdro_phase >= 2)

        with autocast(enabled=self.use_amp):
            if self.wdro is not None:
                critic_loss = self._compute_fleet_critic_loss_wdro(
                    states, hex_allocations, hex_charge_power,
                    rewards_normalized, next_states, dones,
                    vehicle_hex_ids, durations,
                    current_state_tensors=(hex_features_cur, vehicle_features_cur, parsed_hex_ids_cur, context_cur),
                    next_state_tensors=(next_hex, next_veh, next_hex_ids, next_ctx),
                    current_hex_vehicle_summary=hex_veh_summary_cur,
                    current_vehicle_counts=veh_counts_cur,
                )
            else:
                critic_loss = self.agent.compute_critic_loss(
                    states=states,
                    hex_allocations=hex_allocations,
                    charge_power=hex_charge_power,
                    rewards=rewards_normalized,
                    next_states=next_states,
                    dones=dones,
                    vehicle_hex_ids=vehicle_hex_ids,
                    durations=durations,
                )

        self.agent.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.agent.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=1.0)
        self.scaler.step(self.agent.critic_optimizer)
        self.scaler.update()

        self._last_value_debug = {'v_label_mean': 0.0, 'v_pred_mean': 0.0}
        value_loss = torch.tensor(0.0, device=self.device)
        value_training_active = (
            self.value_network is not None and
            self.episode < self.enhanced_config.wdro_value_train_stop_episode
        )

        with torch.no_grad():
            critic_module = self.agent.critic.module if hasattr(self.agent.critic, 'module') else self.agent.critic
            cached_critic_emb1 = critic_module.critic1.gcn(hex_features_cur, self.agent._adjacency_matrix)
            cached_critic_emb2 = critic_module.critic2.gcn(hex_features_cur, self.agent._adjacency_matrix)
            cached_target_critic_emb1 = None
            cached_target_critic_emb2 = None
            cached_actor_emb = None
            if value_training_active:
                target_critic_module = self.agent.critic_target.module if hasattr(self.agent.critic_target, 'module') else self.agent.critic_target
                cached_target_critic_emb1 = target_critic_module.critic1.gcn(hex_features_cur, self.agent._adjacency_matrix)
                cached_target_critic_emb2 = target_critic_module.critic2.gcn(hex_features_cur, self.agent._adjacency_matrix)
                actor_module = self.agent.actor.module if hasattr(self.agent.actor, 'module') else self.agent.actor
                cached_actor_emb = actor_module.gcn(hex_features_cur, self.agent._adjacency_matrix)

        if value_training_active:
            value_loss = self._train_value_network_fleet(
                states,
                vehicle_hex_ids,
                actor_hex_embeddings=cached_actor_emb,
                critic_hex_embeddings_q1=cached_target_critic_emb1,
                critic_hex_embeddings_q2=cached_target_critic_emb2,
                hex_features=hex_features_cur,
                vehicle_features=vehicle_features_cur,
                context_features=context_cur,
                hex_vehicle_summary=hex_veh_summary_cur,
                vehicle_counts=veh_counts_cur,
                active_hex_mask=active_hex_mask_cur,
            )

        with autocast(enabled=self.use_amp):
            actor_loss, entropy, aux = self.agent.compute_actor_loss(
                states=states,
                vehicle_hex_ids=vehicle_hex_ids,
                critic_hex_embeddings_q1=cached_critic_emb1,
                critic_hex_embeddings_q2=cached_critic_emb2,
                hex_features=hex_features_cur,
                vehicle_features=vehicle_features_cur,
                context_features=context_cur,
                hex_vehicle_summary=hex_veh_summary_cur,
                vehicle_counts=veh_counts_cur,
                active_hex_mask=active_hex_mask_cur,
            )

        self.agent.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.agent.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=1.0)
        self.scaler.step(self.agent.actor_optimizer)
        self.scaler.update()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.agent.auto_alpha:
            with autocast(enabled=self.use_amp):
                alpha_loss = self.agent.compute_alpha_loss(entropy)
            self.agent.alpha_optimizer.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.agent.alpha_optimizer)
            self.scaler.update()

        if self.wdro is not None:
            self.wdro.update_lambda()

        self.agent.soft_update_target()
        self.global_step += 1

        if hasattr(self, 'actor_scheduler'):
            self.actor_scheduler.step()
        if hasattr(self, 'critic_scheduler'):
            self.critic_scheduler.step()

        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            'alpha': self.agent.alpha.item(),
            'entropy': entropy.item() if isinstance(entropy, torch.Tensor) else entropy,
            'q_mean': aux.get('q_mean', 0.0),
            'repos_aux_loss': aux.get('repos_aux_loss', 0.0),
            'actor_serve_frac': aux.get('serve_frac', 0.0),
            'actor_charge_frac': aux.get('charge_frac', 0.0),
            'actor_repos_frac': aux.get('repos_frac', 0.0),
            'v_label_mean': self._last_value_debug.get('v_label_mean', 0.0),
            'v_pred_mean': self._last_value_debug.get('v_pred_mean', 0.0),
            'temperature': self.current_temperature,
            'lr_actor': self.agent.actor_optimizer.param_groups[0]['lr'],
            'lr_critic': self.agent.critic_optimizer.param_groups[0]['lr'],
        }
        if self.value_network is not None:
            metrics['value_loss'] = value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss
        if self.wdro is not None:
            metrics['wdro_lambda'] = self.wdro.lambda_.item()
            metrics['wdro_rho_hat'] = self.wdro.running_rho_hat
            metrics['wdro_phase'] = wdro_phase
            metrics['wdro_value_source_is_learned'] = 1 if self.wdro.use_learned_value else 0
            metrics['wdro_value_training_active'] = 1 if value_training_active else 0
            metrics.update(self.wdro.last_debug)

        return metrics

    # ---- WDRO robust critic loss (Phase 2+) ----
    def _compute_fleet_critic_loss_wdro(
        self,
        states: Union[torch.Tensor, Dict[str, torch.Tensor]],
        hex_allocations: torch.Tensor,
        hex_charge_power: torch.Tensor,
        rewards: torch.Tensor,
        next_states: Union[torch.Tensor, Dict[str, torch.Tensor]],
        dones: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        durations: Optional[torch.Tensor],
        current_state_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        next_state_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        current_hex_vehicle_summary: Optional[torch.Tensor] = None,
        current_vehicle_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Critic loss with WDRO robust backup target (paper Eq. 21)."""
        if current_state_tensors is None:
            hex_features, vehicle_features, parsed_hex_ids, context = self.agent._parse_flat_state(states)
        else:
            hex_features, vehicle_features, parsed_hex_ids, context = current_state_tensors
        if next_state_tensors is None:
            next_hex, next_veh, next_hex_ids, next_ctx = self.agent._parse_flat_state(next_states)
        else:
            next_hex, next_veh, next_hex_ids, next_ctx = next_state_tensors

        if vehicle_hex_ids is None:
            vehicle_hex_ids = parsed_hex_ids

        batch_size = hex_features.shape[0]
        xi_hat = next_hex[:, :, 2]

        self.wdro.mag_metric.update_statistics(xi_hat.detach())

        def next_state_fn(xi_scenario):
            return self._apply_scenario_to_fleet_state(
                next_states, xi_scenario, next_hex_ids, parsed_next_state=(next_hex, next_veh, next_hex_ids, next_ctx)
            )

        def duration_fn(xi):
            return durations if durations is not None else torch.ones(batch_size, device=rewards.device)

        target_q = self.wdro.compute_robust_target(
            rewards=rewards,
            xi_hat=xi_hat,
            next_state_fn=next_state_fn,
            gamma=self.agent.gamma,
            duration_fn=duration_fn,
            dones=dones,
            action_probs=None,
            action_log_probs=None,
            alpha=self.agent.alpha.item(),
        ).detach()
        self._ensure_finite('target_q', target_q)

        if current_hex_vehicle_summary is None:
            hex_veh_summary = self.agent._build_hex_vehicle_summary_from_features(vehicle_features, vehicle_hex_ids)
        else:
            hex_veh_summary = current_hex_vehicle_summary
        if current_vehicle_counts is None:
            veh_counts = self.agent._build_vehicle_counts(vehicle_hex_ids)
        else:
            veh_counts = current_vehicle_counts

        q1, q2 = self.agent.critic(
            hex_features=hex_features,
            hex_vehicle_summary=hex_veh_summary,
            context_features=context,
            hex_allocations=hex_allocations,
            charge_power=hex_charge_power,
            adjacency=self.agent._adjacency_matrix,
            vehicle_counts=veh_counts,
        )

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self._ensure_finite('critic_loss_wdro', critic_loss)
        return critic_loss

    # ---- V_phi training while enabled ----
    def _train_value_network_fleet(
        self,
        states: Union[torch.Tensor, Dict[str, torch.Tensor]],
        vehicle_hex_ids: torch.Tensor,
        actor_hex_embeddings: Optional[torch.Tensor] = None,
        critic_hex_embeddings_q1: Optional[torch.Tensor] = None,
        critic_hex_embeddings_q2: Optional[torch.Tensor] = None,
        hex_features: Optional[torch.Tensor] = None,
        vehicle_features: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
        hex_vehicle_summary: Optional[torch.Tensor] = None,
        vehicle_counts: Optional[torch.Tensor] = None,
        active_hex_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Train V_φ on a stabilized soft-value label."""
        if hex_features is None or vehicle_features is None or context_features is None:
            hex_features, vehicle_features, parsed_hex_ids, context = self.agent._parse_flat_state(states)
            if vehicle_hex_ids is None:
                vehicle_hex_ids = parsed_hex_ids
        else:
            context = context_features
        if hex_vehicle_summary is None:
            hex_veh_summary = self.agent._build_hex_vehicle_summary_from_features(vehicle_features, vehicle_hex_ids)
        else:
            hex_veh_summary = hex_vehicle_summary
        if vehicle_counts is None:
            veh_counts = self.agent._build_vehicle_counts(vehicle_hex_ids)
        else:
            veh_counts = vehicle_counts
        if active_hex_mask is None:
            active_hex_mask = self.agent._build_active_hex_mask(hex_veh_summary)

        with torch.no_grad():
            actor_out = self.agent.actor(
                hex_features=hex_features,
                hex_vehicle_summary=hex_veh_summary,
                context_features=context,
                adj=self.agent._adjacency_matrix,
                active_hex_mask=active_hex_mask,
                khop_neighbor_indices=self.agent._khop_neighbor_indices,
                khop_neighbor_mask=self.agent._khop_neighbor_mask,
                temperature=1.0,
                deterministic=True,
                hex_embeddings=actor_hex_embeddings,
            )

            alloc_probs = actor_out['allocation_probs']
            charge_pow = actor_out['charge_power']
            alloc_entropy = actor_out['allocation_entropy']
            charge_log_prob = actor_out['charge_power_log_prob']

            target_critic = self.agent.critic_target.module if hasattr(self.agent.critic_target, 'module') else self.agent.critic_target
            q_fleet = target_critic.min_q(
                hex_features=hex_features,
                hex_vehicle_summary=hex_veh_summary,
                context_features=context,
                hex_allocations=alloc_probs,
                charge_power=charge_pow,
                adjacency=self.agent._adjacency_matrix,
                vehicle_counts=veh_counts,
                hex_embeddings_q1=critic_hex_embeddings_q1,
                hex_embeddings_q2=critic_hex_embeddings_q2,
            )

            total_entropy, _, _ = self.agent._compute_total_entropy(
                allocation_entropy=alloc_entropy,
                charge_power_log_prob=charge_log_prob,
                active_hex_mask=actor_out['active_hex_mask'],
            )

            v_label = q_fleet + self.agent.alpha.detach() * total_entropy
            self._ensure_finite('v_label', v_label)

            if critic_hex_embeddings_q1 is not None:
                hex_emb = critic_hex_embeddings_q1
            else:
                target_critic = self.agent.critic_target.module if hasattr(self.agent.critic_target, 'module') else self.agent.critic_target
                hex_emb = target_critic.critic1.gcn(hex_features, self.agent._adjacency_matrix)

        with autocast(enabled=self.use_amp):
            v_pred = self.value_network(hex_emb, vehicle_features, context)
            self._ensure_finite('v_pred', v_pred)
            value_loss = F.mse_loss(v_pred, v_label)
            self._ensure_finite('value_loss', value_loss)

        self._last_value_debug = {
            'v_label_mean': float(v_label.mean().item()),
            'v_pred_mean': float(v_pred.mean().item()),
        }

        self.value_optimizer.zero_grad()
        self.scaler.scale(value_loss).backward()
        self.scaler.unscale_(self.value_optimizer)
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.scaler.step(self.value_optimizer)
        self.scaler.update()

        return value_loss

    # ---- Checkpoint ----
    def save_checkpoint(self, path: str):
        import os
        # Respect absolute paths and paths with directory components; otherwise
        # prepend the configured checkpoint directory.
        p_obj = __import__('pathlib').Path(path)
        if p_obj.is_absolute() or p_obj.parent != __import__('pathlib').Path('.'):
            ckpt_path = str(p_obj)
        else:
            ckpt_path = os.path.join(self.checkpoint_config.checkpoint_dir, path)
        os.makedirs(os.path.dirname(ckpt_path) or self.checkpoint_config.checkpoint_dir,
                     exist_ok=True)

        checkpoint = {
            'agent': self.agent.state_dict(),
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict() if hasattr(self, 'actor_scheduler') else None,
            'critic_scheduler': self.critic_scheduler.state_dict() if hasattr(self, 'critic_scheduler') else None,
            'scaler': self.scaler.state_dict(),
            'global_step': self.global_step,
            'episode': self.episode,
            'best_reward': self.best_reward,
            'best_train_reward': getattr(self, 'best_train_reward', float('-inf')),
            'total_steps': getattr(self, 'total_steps', 0),
            'recent_rewards_window': list(getattr(self, 'recent_rewards_window', [])),
            'current_temperature': self.current_temperature,
            # Reward normalisation running stats (plain Python attrs, not in state_dict)
            'reward_norm': {
                'mean':  getattr(self.agent, 'reward_mean',  -200.0),
                'std':   getattr(self.agent, 'reward_std',    150.0),
                'count': getattr(self.agent, 'reward_count',   1000),
            },
            'checkpoint_metadata': {
                'vehicle_feature_dim': int(getattr(self.agent, 'vehicle_feature_dim', -1)),
                'num_vehicles': int(getattr(self.agent, 'num_vehicles', -1)),
                'num_hexes': int(getattr(self.agent, 'num_hexes', -1)),
                'context_dim': int(getattr(self.agent, 'context_dim', -1)),
                'uses_wdro': bool(self.wdro is not None),
                'has_value_network': bool(self.value_network is not None),
            },
        }
        if self.agent.auto_alpha:
            checkpoint['alpha_optimizer'] = self.agent.alpha_optimizer.state_dict()
        if self.wdro is not None:
            checkpoint['wdro_state_dict'] = self.wdro.state_dict()
            checkpoint['wdro_runtime_state'] = self.wdro.get_runtime_state()
        if self.value_network is not None:
            checkpoint['value_network_state_dict'] = self.value_network.state_dict()
            checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
        torch.save(checkpoint, ckpt_path)

    def load_checkpoint(self, path: str):
        import os
        from pathlib import Path as _Path
        # Accept absolute paths, paths with directory components, or bare filenames.
        # Bare filenames are resolved relative to checkpoint_dir (same logic as save_checkpoint).
        p_obj = _Path(path)
        if p_obj.is_absolute() or p_obj.parent != _Path('.'):
            load_path = str(p_obj)
        else:
            load_path = os.path.join(self.checkpoint_config.checkpoint_dir, path)
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        metadata = checkpoint.get('checkpoint_metadata')
        if metadata is not None:
            expected = {
                'vehicle_feature_dim': int(getattr(self.agent, 'vehicle_feature_dim', -1)),
                'num_vehicles': int(getattr(self.agent, 'num_vehicles', -1)),
                'num_hexes': int(getattr(self.agent, 'num_hexes', -1)),
                'context_dim': int(getattr(self.agent, 'context_dim', -1)),
            }
            mismatches = [
                f"{key}: checkpoint={metadata.get(key)} current={value}"
                for key, value in expected.items()
                if metadata.get(key) not in (None, value)
            ]
            if mismatches:
                mismatch_str = "; ".join(mismatches)
                raise RuntimeError(
                    "Checkpoint is incompatible with the current fleet schema. "
                    f"Mismatched fields: {mismatch_str}"
                )

        self.agent.load_state_dict(checkpoint['agent'])
        self.global_step = checkpoint.get('global_step', 0)
        self.episode = checkpoint.get('episode', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.best_train_reward = checkpoint.get('best_train_reward', float('-inf'))
        self.total_steps = checkpoint.get('total_steps', 0)
        self.recent_rewards_window = list(checkpoint.get('recent_rewards_window', []))
        self.current_temperature = checkpoint.get('current_temperature', 1.0)

        if 'actor_optimizer' in checkpoint:
            self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'critic_optimizer' in checkpoint:
            self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'alpha_optimizer' in checkpoint and self.agent.auto_alpha:
            self.agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if checkpoint.get('actor_scheduler') and hasattr(self, 'actor_scheduler'):
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
        if checkpoint.get('critic_scheduler') and hasattr(self, 'critic_scheduler'):
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
        if checkpoint.get('scaler') and hasattr(self, 'scaler'):
            self.scaler.load_state_dict(checkpoint['scaler'])
        if self.wdro is not None and 'wdro_state_dict' in checkpoint:
            self.wdro.load_state_dict(checkpoint['wdro_state_dict'])
            self.wdro.load_runtime_state(checkpoint.get('wdro_runtime_state'))
        if self.value_network is not None and 'value_network_state_dict' in checkpoint:
            self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

        # Restore reward normalisation running statistics (not part of module state_dict)
        rn = checkpoint.get('reward_norm')
        if rn is not None:
            self.agent.reward_mean  = float(rn.get('mean',  -200.0))
            self.agent.reward_std   = float(rn.get('std',    150.0))
            self.agent.reward_count = int  (rn.get('count',   1000))



FleetEnhancedSACTrainer = EnhancedSACTrainer

def create_enhanced_trainer(
    agent: FleetSACAgent,
    replay_buffer: GPUReplayBuffer,
    training_config: TrainingConfig,
    use_semi_mdp: bool = True,
    use_wdro: bool = True,
    use_temperature_annealing: bool = True,
    device: str = 'cuda'
) -> 'FleetEnhancedSACTrainer':
    """Factory function to create the fleet enhanced trainer with paper features."""

    enhanced_config = EnhancedTrainingConfig(
        batch_size=training_config.batch_size,
        gamma=training_config.gamma,
        use_semi_mdp=use_semi_mdp,
        use_wdro=use_wdro,
        use_temperature_annealing=use_temperature_annealing,
    )

    return FleetEnhancedSACTrainer(
        agent=agent,
        replay_buffer=replay_buffer,
        training_config=training_config,
        enhanced_config=enhanced_config,
        device=device,
    )
