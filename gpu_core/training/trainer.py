import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import time
import os
from pathlib import Path

from ..config import TrainingConfig, CheckpointConfig, LoggingConfig
from ..networks.sac import SACAgent, FleetSACAgent
from ..features.replay_buffer import GPUReplayBuffer, Transition


@dataclass
class TrainingMetrics:
    episode: int = 0
    step: int = 0
    total_reward: float = 0.0
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    alpha: float = 0.0
    entropy: float = 0.0
    q_mean: float = 0.0
    episode_length: int = 0
    fps: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


class SACTrainer:
    def __init__(
        self,
        agent: SACAgent,
        replay_buffer: GPUReplayBuffer,
        training_config: TrainingConfig,
        checkpoint_config: Optional[CheckpointConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
        device: str = 'cuda',
        use_amp: bool = False,
    ):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = training_config
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.logging_config = logging_config or LoggingConfig()
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.global_step = 0
        self.episode = 0
        self.best_reward = float('-inf')
        self.best_train_reward = float('-inf')
        self.total_steps = 0
        self.recent_rewards_window = []

        self._compile_if_enabled()
        self._setup_lr_schedulers()
    
    def _setup_lr_schedulers(self):
        """Setup learning rate schedulers for better convergence."""
        # Cosine annealing with warm restarts
        total_steps = getattr(self.config, 'total_steps', 1000000)
        
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.agent.actor_optimizer,
            T_0=total_steps // 10,  # Restart every 10% of training
            T_mult=2,
            eta_min=1e-6
        )
        
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.agent.critic_optimizer,
            T_0=total_steps // 10,
            T_mult=2,
            eta_min=1e-6
        )
    
    def _compile_if_enabled(self):
        use_compile = getattr(self.config, 'compile_model', False)
        if use_compile and hasattr(torch, 'compile'):
            self.agent.actor = torch.compile(self.agent.actor, mode='reduce-overhead', options={"triton.cudagraphs": False})
            self.agent.critic = torch.compile(self.agent.critic, mode='reduce-overhead', options={"triton.cudagraphs": False})
    
    def train_step(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones
        
        # Check if prioritized replay is enabled
        use_prioritized = getattr(self.config, 'use_prioritized_replay', False) or \
                          getattr(self.replay_buffer, 'prioritized', False)
        should_update_priorities = use_prioritized and batch.indices is not None

        states_flat = self.agent._flatten_states(states)
        next_states_flat = self.agent._flatten_states(next_states)
        rewards_normalized = self.agent._normalize_rewards(rewards)

        # Critic update (+ optional td_error extraction for PER)
        with autocast(enabled=self.use_amp):
            critic_result = self.agent.compute_critic_loss(
                states=states_flat,
                actions=actions,
                rewards=rewards_normalized,
                next_states=next_states_flat,
                dones=dones,
                serve_vehicle_idx=batch.serve_vehicle_idx,
                serve_trip_idx=batch.serve_trip_idx,
                num_served=batch.num_served,
                charge_vehicle_idx=batch.charge_vehicle_idx,
                charge_station_idx=batch.charge_station_idx,
                num_charged=batch.num_charged,
                return_td_error=should_update_priorities,
            )

        if should_update_priorities:
            critic_loss, td_error = critic_result
        else:
            critic_loss = critic_result
            td_error = None

        self.agent.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.agent.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=1.0)
        self.scaler.step(self.agent.critic_optimizer)
        self.scaler.update()

        # Actor update
        with autocast(enabled=self.use_amp):
            actor_loss, log_prob, aux_losses = self.agent.compute_actor_loss(
                states=states_flat,
                serve_vehicle_idx=batch.serve_vehicle_idx,
                serve_trip_idx=batch.serve_trip_idx,
                num_served=batch.num_served,
                charge_vehicle_idx=batch.charge_vehicle_idx,
                charge_station_idx=batch.charge_station_idx,
                num_charged=batch.num_charged,
            )

        self.agent.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.agent.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=1.0)
        self.scaler.step(self.agent.actor_optimizer)
        self.scaler.update()

        # Alpha update
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.agent.auto_alpha:
            with autocast(enabled=self.use_amp):
                alpha_loss = self.agent.compute_alpha_loss(log_prob)
            self.agent.alpha_optimizer.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.agent.alpha_optimizer)
            self.scaler.update()

        self.agent.soft_update_target()
        self.global_step += 1
        
        if should_update_priorities and td_error is not None:
            self.replay_buffer.update_priorities(
                batch.indices,
                td_error  # Keep as tensor, not numpy
            )

        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            'alpha': self.agent.alpha.item(),
            'log_prob': log_prob.mean().item(),
            'q_mean': aux_losses.get('q_mean', 0.0),
            'q_std': aux_losses.get('q_std', 0.0),
            'repos_aux_loss': aux_losses.get('repos_aux_loss', 0.0),
        }
        
        # Step learning rate schedulers
        if hasattr(self, 'actor_scheduler'):
            self.actor_scheduler.step()
        if hasattr(self, 'critic_scheduler'):
            self.critic_scheduler.step()
        
        # Add LR to metrics
        metrics['lr_actor'] = self.agent.actor_optimizer.param_groups[0]['lr']
        metrics['lr_critic'] = self.agent.critic_optimizer.param_groups[0]['lr']
        
        return metrics
    
    def train_episode(
        self,
        env,
        max_steps: int = 1000,
        warmup_steps: int = 0
    ) -> TrainingMetrics:
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        start_time = time.time()
        
        metrics_accum = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'alpha': 0.0,
            'q_mean': 0.0,
            'updates': 0
        }
        
        for step in range(max_steps):
            if self.global_step < warmup_steps:
                action_type = torch.randint(0, self.agent.action_dim, (1,), device=self.device)
                reposition_target = torch.randint(0, self.agent.num_hexes, (1,), device=self.device)
            else:
                output = self.agent.select_action(
                    state.unsqueeze(0) if state.dim() == 1 else state
                )
                action_type = output.action_type
                reposition_target = output.reposition_target
            
            next_state, reward, done, info = env.step(action_type.item(), reposition_target.item())
            
            transition = Transition(
                state=state,
                action=action_type,
                reward=torch.tensor([reward], device=self.device),
                next_state=next_state,
                done=torch.tensor([done], device=self.device)
            )
            self.replay_buffer.add(transition)
            
            episode_reward += reward
            episode_length += 1
            
            if self.global_step >= warmup_steps and len(self.replay_buffer) >= self.config.batch_size:
                for _ in range(self.config.gradient_steps):
                    step_metrics = self.train_step()
                    if step_metrics:
                        metrics_accum['actor_loss'] += step_metrics.get('actor_loss', 0)
                        metrics_accum['critic_loss'] += step_metrics.get('critic_loss', 0)
                        metrics_accum['alpha'] += step_metrics.get('alpha', 0)
                        metrics_accum['updates'] += 1
            
            state = next_state
            
            if done:
                break
        
        elapsed = time.time() - start_time
        fps = episode_length / elapsed if elapsed > 0 else 0
        
        num_updates = max(metrics_accum['updates'], 1)
        
        self.episode += 1
        
        return TrainingMetrics(
            episode=self.episode,
            step=self.global_step,
            total_reward=episode_reward,
            actor_loss=metrics_accum['actor_loss'] / num_updates,
            critic_loss=metrics_accum['critic_loss'] / num_updates,
            alpha=metrics_accum['alpha'] / num_updates,
            episode_length=episode_length,
            fps=fps
        )
    
    def train(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int = 1000,
        warmup_steps: int = 1000,
        eval_interval: int = 10,
        eval_episodes: int = 5,
        callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> list:
        all_metrics = []
        
        for ep in range(num_episodes):
            metrics = self.train_episode(
                env=env,
                max_steps=max_steps_per_episode,
                warmup_steps=warmup_steps
            )
            all_metrics.append(metrics)
            
            if callback:
                callback(metrics)
            
            if self.logging_config.log_interval > 0 and ep % self.logging_config.log_interval == 0:
                self._log_metrics(metrics)
            
            if eval_interval > 0 and ep % eval_interval == 0:
                eval_reward = self.evaluate(env, eval_episodes, max_steps_per_episode)
                metrics.extra['eval_reward'] = eval_reward
                
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    if self.checkpoint_config.save_best:
                        self.save_checkpoint('best.pt')
            
            if self.checkpoint_config.save_interval > 0 and ep % self.checkpoint_config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_{ep}.pt')
        
        return all_metrics
    
    def evaluate(
        self,
        env,
        num_episodes: int,
        max_steps: int = 1000
    ) -> float:
        self.agent.eval()
        total_reward = 0.0
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            
            for _ in range(max_steps):
                with torch.no_grad():
                    output = self.agent.select_action(
                        state.unsqueeze(0) if state.dim() == 1 else state,
                        deterministic=True
                    )
                    action_type = output.action_type
                    reposition_target = output.reposition_target
                
                next_state, reward, done, _ = env.step(action_type.item(), reposition_target.item())
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            total_reward += episode_reward
        
        self.agent.train()
        return total_reward / num_episodes
    
    def _log_metrics(self, metrics: TrainingMetrics):
        print(f"Episode {metrics.episode} | "
              f"Step {metrics.step} | "
              f"Reward: {metrics.total_reward:.2f} | "
              f"Actor Loss: {metrics.actor_loss:.4f} | "
              f"Critic Loss: {metrics.critic_loss:.4f} | "
              f"Alpha: {metrics.alpha:.4f} | "
              f"FPS: {metrics.fps:.1f}")
    
    def save_checkpoint(self, filename: str):
        path = Path(self.checkpoint_config.checkpoint_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'agent': self.agent.state_dict(),
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict() if hasattr(self, 'actor_scheduler') else None,
            'critic_scheduler': self.critic_scheduler.state_dict() if hasattr(self, 'critic_scheduler') else None,
            'scaler': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'global_step': self.global_step,
            'episode': self.episode,
            'best_reward': self.best_reward,
            'best_train_reward': getattr(self, 'best_train_reward', float('-inf')),
            'total_steps': getattr(self, 'total_steps', 0),
            'recent_rewards_window': list(getattr(self, 'recent_rewards_window', [])),
            # Reward normalisation running stats (plain Python attrs, not in state_dict)
            'reward_norm': {
                'mean':  getattr(self.agent, 'reward_mean',  -200.0),
                'std':   getattr(self.agent, 'reward_std',    150.0),
                'count': getattr(self.agent, 'reward_count',   1000),
            },
            'config': {
                'training': self.config.__dict__,
                'checkpoint': self.checkpoint_config.__dict__
            }
        }
        if self.agent.auto_alpha:
            checkpoint['alpha_optimizer'] = self.agent.alpha_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        # If the caller passes an absolute path or a path that already contains
        # directory separators (e.g. "checkpoints/best.pt"), use it directly.
        # Only prepend checkpoint_dir when given a bare filename.
        p = Path(filename)
        if p.is_absolute() or p.parent != Path('.'):
            path = p
        else:
            path = Path(self.checkpoint_config.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.agent.load_state_dict(checkpoint['agent'])
        self.global_step = checkpoint.get('global_step', 0)
        self.episode = checkpoint.get('episode', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.best_train_reward = checkpoint.get('best_train_reward', float('-inf'))
        self.total_steps = checkpoint.get('total_steps', 0)
        self.recent_rewards_window = list(checkpoint.get('recent_rewards_window', []))

        # Optimizer states — guarded for backward compat with old checkpoints
        if 'actor_optimizer' in checkpoint:
            self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'critic_optimizer' in checkpoint:
            self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'alpha_optimizer' in checkpoint and self.agent.auto_alpha:
            self.agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if checkpoint.get('actor_scheduler') is not None and hasattr(self, 'actor_scheduler'):
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
        if checkpoint.get('critic_scheduler') is not None and hasattr(self, 'critic_scheduler'):
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
        if checkpoint.get('scaler') is not None and hasattr(self, 'scaler'):
            self.scaler.load_state_dict(checkpoint['scaler'])

        # Restore reward normalisation running statistics (not part of module state_dict)
        rn = checkpoint.get('reward_norm')
        if rn is not None:
            self.agent.reward_mean  = float(rn.get('mean',  -200.0))
            self.agent.reward_std   = float(rn.get('std',    150.0))
            self.agent.reward_count = int  (rn.get('count',   1000))
        # Restore reward normalisation running statistics (not part of module state_dict)
        rn = checkpoint.get('reward_norm')
        if rn is not None:
            self.agent.reward_mean  = float(rn.get('mean',  -200.0))
            self.agent.reward_std   = float(rn.get('std',    150.0))
            self.agent.reward_count = int  (rn.get('count',   1000))



class FleetSACTrainer:
    """Trainer for FleetSACAgent using hex-level actions.

    Replaces per-vehicle critic/actor loss with fleet-level losses:
    - Critic takes soft allocation probs [B, H, 3] + charge_power [B, H]
    - Actor loss uses allocation entropy + charge entropy for SAC
    - No action enumeration needed
    """

    def __init__(
        self,
        agent: FleetSACAgent,
        replay_buffer: GPUReplayBuffer,
        training_config: TrainingConfig,
        checkpoint_config: Optional[CheckpointConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
        device: str = 'cuda',
        use_amp: bool = False,
    ):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = training_config
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.logging_config = logging_config or LoggingConfig()
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=self.use_amp)

        self.global_step = 0
        self.episode = 0
        self.best_reward = float('-inf')
        self.best_train_reward = float('-inf')
        self.total_steps = 0
        self.recent_rewards_window = []

        self._setup_lr_schedulers()

    def _setup_lr_schedulers(self):
        total_steps = getattr(self.config, 'total_steps', 1000000)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.agent.actor_optimizer,
            T_0=total_steps // 10, T_mult=2, eta_min=1e-6,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.agent.critic_optimizer,
            T_0=total_steps // 10, T_mult=2, eta_min=1e-6,
        )

    def train_step(self, adjacency: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Single gradient step using fleet-level actions from replay buffer."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.config.batch_size, adjacency=adjacency)

        states = batch.states
        next_states = batch.next_states
        rewards = batch.rewards
        dones = batch.dones
        durations = batch.durations

        hex_allocations = batch.hex_allocations     # [B, H, 3]
        hex_charge_power = batch.hex_charge_power   # [B, H]
        vehicle_hex_ids = batch.vehicle_hex_ids     # [B, V]

        # Pre-parse states and build summaries (shared by critic and actor)
        hex_features, vehicle_features, parsed_hex_ids, context = self.agent._parse_flat_state(states)
        if vehicle_hex_ids is None:
            vehicle_hex_ids = parsed_hex_ids
        hex_veh_summary = self.agent._build_hex_vehicle_summary_from_features(vehicle_features, vehicle_hex_ids)
        veh_counts = self.agent._build_vehicle_counts(vehicle_hex_ids)
        active_hex_mask = self.agent._build_active_hex_mask(hex_veh_summary)

        # Normalise rewards
        rewards_normalized = self.agent._normalize_rewards(rewards)

        # ---------- Critic update ----------
        with autocast(enabled=self.use_amp):
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

        # Cache critic GCN embeddings under no_grad so the actor backward
        # pass doesn't build autograd graphs for the critic's GCN (~0.7 GB saved)
        with torch.no_grad():
            critic_module = self.agent.critic.module if hasattr(self.agent.critic, 'module') else self.agent.critic
            cached_critic_emb1 = critic_module.critic1.gcn(hex_features, self.agent._adjacency_matrix)
            cached_critic_emb2 = critic_module.critic2.gcn(hex_features, self.agent._adjacency_matrix)

        # ---------- Actor update ----------
        with autocast(enabled=self.use_amp):
            actor_loss, entropy, aux = self.agent.compute_actor_loss(
                states=states,
                vehicle_hex_ids=vehicle_hex_ids,
                critic_hex_embeddings_q1=cached_critic_emb1,
                critic_hex_embeddings_q2=cached_critic_emb2,
                hex_features=hex_features,
                vehicle_features=vehicle_features,
                context_features=context,
                hex_vehicle_summary=hex_veh_summary,
                vehicle_counts=veh_counts,
                active_hex_mask=active_hex_mask,
            )

        self.agent.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.agent.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=1.0)
        self.scaler.step(self.agent.actor_optimizer)
        self.scaler.update()

        # ---------- Alpha (entropy temperature) update ----------
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.agent.auto_alpha:
            with autocast(enabled=self.use_amp):
                alpha_loss = self.agent.compute_alpha_loss(entropy)
            self.agent.alpha_optimizer.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.agent.alpha_optimizer)
            self.scaler.update()

        self.agent.soft_update_target()
        self.global_step += 1

        # LR schedulers
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
            'lr_actor': self.agent.actor_optimizer.param_groups[0]['lr'],
            'lr_critic': self.agent.critic_optimizer.param_groups[0]['lr'],
        }
        return metrics

    def save_checkpoint(self, filename: str):
        path = Path(self.checkpoint_config.checkpoint_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'agent': self.agent.state_dict(),
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict() if hasattr(self, 'actor_scheduler') else None,
            'critic_scheduler': self.critic_scheduler.state_dict() if hasattr(self, 'critic_scheduler') else None,
            'scaler': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'global_step': self.global_step,
            'episode': self.episode,
            'best_reward': self.best_reward,
            'best_train_reward': getattr(self, 'best_train_reward', float('-inf')),
            'total_steps': getattr(self, 'total_steps', 0),
            'recent_rewards_window': list(getattr(self, 'recent_rewards_window', [])),
            # Reward normalisation running stats (plain Python attrs, not in state_dict)
            'reward_norm': {
                'mean':  getattr(self.agent, 'reward_mean', -200.0),
                'std':   getattr(self.agent, 'reward_std',  150.0),
                'count': getattr(self.agent, 'reward_count', 1000),
            },
        }
        if self.agent.auto_alpha:
            checkpoint['alpha_optimizer'] = self.agent.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        # If the caller passes an absolute path or a path that already contains
        # directory separators (e.g. "checkpoints/best.pt"), use it directly.
        # Only prepend checkpoint_dir when given a bare filename.
        p = Path(filename)
        if p.is_absolute() or p.parent != Path('.'):
            path = p
        else:
            path = Path(self.checkpoint_config.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.agent.load_state_dict(checkpoint['agent'])
        self.global_step = checkpoint.get('global_step', 0)
        self.episode = checkpoint.get('episode', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.best_train_reward = checkpoint.get('best_train_reward', float('-inf'))
        self.total_steps = checkpoint.get('total_steps', 0)
        self.recent_rewards_window = list(checkpoint.get('recent_rewards_window', []))

        if 'actor_optimizer' in checkpoint:
            self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'critic_optimizer' in checkpoint:
            self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'alpha_optimizer' in checkpoint and self.agent.auto_alpha:
            self.agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if checkpoint.get('actor_scheduler') is not None and hasattr(self, 'actor_scheduler'):
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
        if checkpoint.get('critic_scheduler') is not None and hasattr(self, 'critic_scheduler'):
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
        if checkpoint.get('scaler') is not None and hasattr(self, 'scaler'):
            self.scaler.load_state_dict(checkpoint['scaler'])

        # Restore reward normalisation running statistics (not part of module state_dict)
        rn = checkpoint.get('reward_norm')
        if rn is not None:
            self.agent.reward_mean  = float(rn.get('mean',  -200.0))
            self.agent.reward_std   = float(rn.get('std',    150.0))
            self.agent.reward_count = int  (rn.get('count',   1000))