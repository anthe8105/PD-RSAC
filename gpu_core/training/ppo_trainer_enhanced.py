"""
ppo_trainer_enhanced.py — Standard MAPPO Trainer.

Implements the standard MAPPO update loop:
- Standard GAE: fixed γ and λ (no semi-MDP duration discounting)
- Global advantage normalization across all agents × timesteps
- Fixed ent_coef (no floor/clamp)
- Single optimizer step (actor + critic updated together)
- Standard PPO clip only — no extra hard ratio clamp
- Entropy scale logged every update for diagnostics
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from gpu_core.networks.ppo_agent import PPOAgent
from gpu_core.features.ppo_buffer import PPORolloutBuffer
from gpu_core.config import TrainingConfig, CheckpointConfig, LoggingConfig


class EnhancedPPOTrainer:
    def __init__(
        self,
        agent: PPOAgent,
        replay_buffer: PPORolloutBuffer,
        training_config: TrainingConfig,
        checkpoint_config: CheckpointConfig,
        logging_config: LoggingConfig,
        device: str = 'cuda',
    ):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = training_config
        self.ckpt_config = checkpoint_config
        self.log_config = logging_config
        self.device = device

        self.global_step = 0
        self.episode = 0
        self.best_reward = float('-inf')
        self._khop_neighbor_indices: Optional[torch.Tensor] = None
        self._khop_neighbor_mask: Optional[torch.Tensor] = None
        self._use_khop_candidates: bool = bool(getattr(training_config, 'mappo_use_khop_candidates', False))

    def compute_gae(
        self,
        data: Dict[str, Any],
        next_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation — standard formulation."""
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']

        T = rewards.shape[0]

        if values.dim() > 1:
            values = values.squeeze(-1)

        if rewards.dim() == 1:
            num_vehicles = 1
            rewards = rewards.unsqueeze(1)
        else:
            num_vehicles = rewards.shape[1]

        if dones.dim() > 1:
            dones = dones.squeeze(-1)

        alpha = float(getattr(self.config, 'mappo_reward_mix_alpha', 0.0))
        alpha = max(0.0, min(1.0, alpha))
        reward_global = rewards.mean(dim=1, keepdim=True).expand_as(rewards)
        rewards_mixed = (1.0 - alpha) * reward_global + alpha * rewards

        advantages = torch.zeros((T, num_vehicles), device=self.device)
        last_adv = torch.zeros(num_vehicles, device=self.device)

        gamma = self.config.gamma
        lam = self.config.gae_lambda

        for t in reversed(range(T)):
            if t == T - 1:
                next_v = next_value if isinstance(next_value, torch.Tensor) else torch.tensor(float(next_value), device=self.device)
            else:
                next_v = values[t + 1]

            delta = rewards_mixed[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
            last_adv = delta + gamma * lam * (1.0 - dones[t]) * last_adv
            advantages[t] = last_adv

        returns = advantages + values.unsqueeze(1).expand_as(advantages)
        return advantages.detach(), returns.detach()

    def _ensure_khop_neighbors(self, data: Dict[str, Any]) -> None:
        if self._khop_neighbor_indices is not None and self._khop_neighbor_mask is not None:
            return

        vehicle_hex_ids = data.get('vehicle_hex_ids')
        reposition_masks = data.get('reposition_masks')
        states = data.get('states')
        if vehicle_hex_ids is None or reposition_masks is None or states is None or len(states) == 0:
            return

        if 'hex' not in states[0]:
            return
        H = int(states[0]['hex'].shape[0])
        K = int(self.agent.max_k_neighbors)

        khop_indices = torch.full((H, K), -1, dtype=torch.long, device=self.device)
        khop_mask = torch.zeros((H, K), dtype=torch.bool, device=self.device)

        flat_hex = vehicle_hex_ids.reshape(-1).to(torch.long)
        flat_repos = reposition_masks.reshape(-1, reposition_masks.shape[-1])

        unique_hex = torch.unique(flat_hex)
        for hex_id_t in unique_hex:
            hex_id = int(hex_id_t.item())
            if hex_id < 0 or hex_id >= H:
                continue
            first_idx = (flat_hex == hex_id_t).nonzero(as_tuple=False)
            if first_idx.numel() == 0:
                continue
            row_mask = flat_repos[int(first_idx[0].item())]
            neighbors = row_mask.nonzero(as_tuple=False).squeeze(-1)
            if neighbors.numel() == 0:
                continue
            n = min(int(neighbors.numel()), K)
            khop_indices[hex_id, :n] = neighbors[:n]
            khop_mask[hex_id, :n] = True

        self._khop_neighbor_indices = khop_indices
        self._khop_neighbor_mask = khop_mask

    def train_step(self, next_value: float = 0.0) -> Dict[str, float]:
        """Execute a full standard MAPPO update over the rollout buffer."""
        if len(self.replay_buffer) == 0:
            return {}

        self.agent.train()
        data = self.replay_buffer.get_tensors(self.device)
        if self._use_khop_candidates:
            self._ensure_khop_neighbors(data)

        advantages, returns = self.compute_gae(data, next_value)
        critic_targets = returns.mean(dim=1) if returns.dim() > 1 else returns
        self.agent.update_value_norm_stats(critic_targets)

        norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        norm_adv = torch.nan_to_num(norm_adv, nan=0.0)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_ent_scale = 0.0
        total_consistency = 0.0
        total_serve_consistent = 0.0
        total_serve_attempts = 0.0
        total_charge_consistent = 0.0
        total_charge_attempts = 0.0
        total_reposition_consistent = 0.0
        total_reposition_attempts = 0.0
        total_failed_action_frac = 0.0
        total_soft_weight_mean = 0.0
        updates = 0

        T = len(data['rewards'])

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(T, device=self.device)

            for start in range(0, T, self.config.batch_size):
                mb_idx = indices[start:start + self.config.batch_size]

                states = self.replay_buffer._batch_states(mb_idx.cpu(), data['states'], self.device)
                actions = data['actions'][mb_idx]
                executed_actions = data['executed_actions'][mb_idx]
                old_logp = data['log_probs'][mb_idx]
                mb_old_values = data['values'][mb_idx]
                mb_adv = norm_adv[mb_idx]
                mb_ret = returns[mb_idx]

                action_mask = data['action_masks'][mb_idx] if data.get('action_masks') is not None else None
                reposition_mask = data['reposition_masks'][mb_idx] if data.get('reposition_masks') is not None else None
                vehicle_hex_ids = data['vehicle_hex_ids'][mb_idx] if data.get('vehicle_hex_ids') is not None else None
                trip_mask = None
                mb_trip_masks = data.get('trip_masks')
                if mb_trip_masks is not None:
                    trip_mask = mb_trip_masks[mb_idx]
                else:
                    mb_trip_counts = data.get('trip_active_counts')
                    if mb_trip_counts is not None:
                        mb_counts = mb_trip_counts[mb_idx]
                        if mb_counts.max() > 0:
                            max_trips = self.agent._max_trips
                            trip_mask = (
                                torch.arange(max_trips, device=self.device).unsqueeze(0)
                                < mb_counts.unsqueeze(1)
                            )

                new_logp, entropy = self.agent.evaluate_actions(
                    states,
                    action_type=actions[:, :, 0],
                    reposition_target=actions[:, :, 1],
                    selected_trip=actions[:, :, 2] if actions.shape[2] > 2 else None,
                    action_mask=action_mask,
                    reposition_mask=reposition_mask,
                    trip_mask=trip_mask,
                    vehicle_hex_ids=vehicle_hex_ids,
                    khop_neighbor_indices=self._khop_neighbor_indices if self._use_khop_candidates else None,
                    khop_neighbor_mask=self._khop_neighbor_mask if self._use_khop_candidates else None,
                )

                ratio = torch.exp(new_logp - old_logp)

                if action_mask is not None:
                    active_mask = action_mask.any(dim=-1).float()
                else:
                    active_mask = torch.ones_like(actions[:, :, 0], dtype=torch.float32)

                executed_valid = (executed_actions[:, :, 0] >= 0)
                intended_valid = (actions[:, :, 0] >= 0)
                consistency_mask = active_mask * (executed_valid & intended_valid).float()

                serve_consistency = (
                    (actions[:, :, 0] != 0)
                    | (
                        (executed_actions[:, :, 0] == 0)
                        & (executed_actions[:, :, 2] == actions[:, :, 2])
                    )
                )
                charge_consistency = (
                    (actions[:, :, 0] != 1)
                    | (executed_actions[:, :, 0] == 1)
                )
                reposition_consistency = (
                    (actions[:, :, 0] != 2)
                    | (
                        (executed_actions[:, :, 0] == 2)
                        & (executed_actions[:, :, 1] == actions[:, :, 1])
                    )
                )

                consistency_mask = consistency_mask * serve_consistency.float() * charge_consistency.float() * reposition_consistency.float()

                use_consistency_mask = bool(getattr(self.config, 'use_execution_consistency_mask', False))
                objective_mask = consistency_mask if use_consistency_mask else active_mask

                failed_mask = (executed_actions[:, :, 0] < 0) & (actions[:, :, 0] >= 0)
                soft_weight = torch.ones_like(active_mask)
                use_soft_weight = bool(getattr(self.config, 'mappo_use_execution_soft_weight', True))
                if use_soft_weight:
                    failed_action_weight = float(getattr(self.config, 'mappo_failed_action_weight', 0.35))
                    failed_serve_weight = float(getattr(self.config, 'mappo_failed_serve_weight', 0.25))
                    failed_action_weight = max(0.0, min(1.0, failed_action_weight))
                    failed_serve_weight = max(0.0, min(1.0, failed_serve_weight))
                    failed_serve_mask = failed_mask & (actions[:, :, 0] == 0)
                    soft_weight = torch.where(failed_mask, torch.full_like(soft_weight, failed_action_weight), soft_weight)
                    soft_weight = torch.where(failed_serve_mask, torch.full_like(soft_weight, failed_serve_weight), soft_weight)

                weighted_adv = mb_adv * objective_mask * soft_weight
                surr1 = ratio * weighted_adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * weighted_adv

                active_count = torch.clamp(objective_mask.sum(), min=1.0)
                actor_loss = -torch.min(surr1, surr2).sum() / active_count

                values_pred = self.agent.get_value(states)
                critic_target = mb_ret.mean(dim=1) if mb_ret.dim() > 1 else mb_ret

                vf_clip = getattr(self.config, 'vf_clip_eps', self.config.clip_eps)
                if mb_old_values.dim() > 1:
                    mb_old_values = mb_old_values.squeeze(-1)
                v_pred_clipped = mb_old_values + (values_pred - mb_old_values).clamp(-vf_clip, vf_clip)

                target_norm = self.agent.normalize_values(critic_target)
                pred_norm = self.agent.normalize_values(values_pred)
                pred_clip_norm = self.agent.normalize_values(v_pred_clipped)

                critic_loss_unclipped = F.mse_loss(pred_norm, target_norm)
                critic_loss_clipped = F.mse_loss(pred_clip_norm, target_norm)
                critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)

                ent_mean = (entropy * objective_mask).sum() / active_count
                ent_scale = self.config.ent_coef * ent_mean
                loss = actor_loss + self.config.vf_coef * critic_loss - ent_scale

                if torch.isnan(loss):
                    print(f'  [Warning] NaN detected: actor={actor_loss.item():.4f} critic={critic_loss.item():.4f} ent={ent_mean.item():.4f}')
                    continue

                self.agent.actor_optimizer.zero_grad()
                self.agent.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent.actor.parameters()) + list(self.agent.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.agent.actor_optimizer.step()
                self.agent.critic_optimizer.step()

                serve_attempt_mask = active_mask * (actions[:, :, 0] == 0).float()
                charge_attempt_mask = active_mask * (actions[:, :, 0] == 1).float()
                reposition_attempt_mask = active_mask * (actions[:, :, 0] == 2).float()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += ent_mean.item()
                total_ent_scale += ent_scale.item()
                total_consistency += (consistency_mask.sum() / torch.clamp(active_mask.sum(), min=1.0)).item()
                total_serve_consistent += (serve_consistency.float() * serve_attempt_mask).sum().item()
                total_serve_attempts += serve_attempt_mask.sum().item()
                total_charge_consistent += (charge_consistency.float() * charge_attempt_mask).sum().item()
                total_charge_attempts += charge_attempt_mask.sum().item()
                total_reposition_consistent += (reposition_consistency.float() * reposition_attempt_mask).sum().item()
                total_reposition_attempts += reposition_attempt_mask.sum().item()
                total_failed_action_frac += (failed_mask.float() * active_mask).sum().item() / torch.clamp(active_mask.sum(), min=1.0).item()
                total_soft_weight_mean += (soft_weight * objective_mask).sum().item() / torch.clamp(objective_mask.sum(), min=1.0).item()
                updates += 1

        self.replay_buffer.clear()

        n = max(1, updates)
        metrics = {
            'actor_loss': total_actor_loss / n,
            'critic_loss': total_critic_loss / n,
            'entropy': total_entropy / n,
            'ent_scale': total_ent_scale / n,
            'consistency_ratio': total_consistency / n,
            'serve_consistency': total_serve_consistent / max(1.0, total_serve_attempts),
            'charge_consistency': total_charge_consistent / max(1.0, total_charge_attempts),
            'reposition_consistency': total_reposition_consistent / max(1.0, total_reposition_attempts),
            'failed_action_fraction': total_failed_action_frac / n,
            'soft_weight_mean': total_soft_weight_mean / n,
            'value_norm_mean': self.agent.value_norm_mean.item(),
            'value_norm_std': torch.sqrt(self.agent.value_norm_var).item(),
        }

        if abs(metrics['actor_loss']) > 1e-6:
            ratio_ent = abs(metrics['ent_scale']) / (abs(metrics['actor_loss']) + 1e-8)
            if ratio_ent > 2.0:
                print(f'  [MAPPO] Entropy term ({metrics["ent_scale"]:.4f}) is {ratio_ent:.1f}× larger than actor loss ({metrics["actor_loss"]:.4f}). Consider reducing ent_coef ({self.config.ent_coef}).')

        return metrics

    def save_checkpoint(self, filename: str) -> None:
        from pathlib import Path
        save_dir = getattr(self.ckpt_config, 'checkpoint_dir', getattr(self.ckpt_config, 'save_dir', 'checkpoints'))
        path = Path(save_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'agent': self.agent.state_dict(),
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'optimizer': self.agent.actor_optimizer.state_dict(),
            'global_step': self.global_step,
            'episode': self.episode,
            'best_reward': self.best_reward,
            'value_norm_mean': self.agent.value_norm_mean,
            'value_norm_var': self.agent.value_norm_var,
            'value_norm_count': self.agent.value_norm_count,
        }, path)
        print(f'Saved checkpoint to {path}')

    def load_checkpoint(self, filename: str) -> None:
        import os
        from pathlib import Path

        path = Path(filename) if os.path.isabs(filename) else Path(getattr(self.ckpt_config, 'checkpoint_dir', getattr(self.ckpt_config, 'save_dir', 'checkpoints'))) / filename
        if not path.exists():
            print(f'Checkpoint not found: {path}')
            return

        ckpt = torch.load(path, map_location=self.device)
        self.global_step = ckpt.get('global_step', 0)
        self.episode = ckpt.get('episode', 0)
        self.best_reward = ckpt.get('best_reward', float('-inf'))

        if 'agent' in ckpt:
            model_state = self.agent.state_dict()
            incoming = ckpt['agent']
            filtered = {}
            skipped = []
            for k, v in incoming.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered[k] = v
                elif k in model_state:
                    skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            self.agent.load_state_dict(filtered, strict=False)
            if skipped:
                print('[MAPPO] Warning: skipped checkpoint tensors with shape mismatch:')
                for key, old_shape, new_shape in skipped:
                    print(f'  - {key}: ckpt={old_shape}, model={new_shape}')
        else:
            self.agent.actor.load_state_dict(ckpt.get('actor_state_dict', {}), strict=False)
            self.agent.critic.load_state_dict(ckpt.get('critic_state_dict', {}), strict=False)

        if 'actor_optimizer' in ckpt and 'critic_optimizer' in ckpt:
            self.agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
            self.agent.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        elif 'optimizer' in ckpt:
            self.agent.actor_optimizer.load_state_dict(ckpt['optimizer'])
            self.agent.critic_optimizer.load_state_dict(ckpt['optimizer'])

        if 'value_norm_mean' in ckpt and hasattr(self.agent, 'value_norm_mean'):
            self.agent.value_norm_mean.copy_(ckpt['value_norm_mean'].to(self.agent.value_norm_mean.device))
        if 'value_norm_var' in ckpt and hasattr(self.agent, 'value_norm_var'):
            self.agent.value_norm_var.copy_(ckpt['value_norm_var'].to(self.agent.value_norm_var.device))
        if 'value_norm_count' in ckpt and hasattr(self.agent, 'value_norm_count'):
            self.agent.value_norm_count.copy_(ckpt['value_norm_count'].to(self.agent.value_norm_count.device))

        print(f'Loaded checkpoint from {path}')

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        print(f'Episode {self.episode} | Step {self.global_step} | Actor Loss: {metrics.get("actor_loss", 0):.4f} | Critic Loss: {metrics.get("critic_loss", 0):.4f} | Entropy: {metrics.get("entropy", 0):.4f} | Ent Scale: {metrics.get("ent_scale", 0):.4f}')
