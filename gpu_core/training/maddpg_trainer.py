"""
maddpg_trainer.py — MADDPG Trainer (NIPS 2017, Lowe et al.).
"""

import torch
import torch.nn.functional as F
try:
    from torch.amp import autocast, GradScaler
    _AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_DEVICE = None
from typing import Optional, Dict

from gpu_core.networks.maddpg_agent import MADDPGAgent
from gpu_core.features.maddpg_buffer import MADDPGReplayBuffer, MADDPGBatch
from gpu_core.config import TrainingConfig, CheckpointConfig, LoggingConfig


class MADDPGTrainer:
    def __init__(
        self,
        agent: MADDPGAgent,
        replay_buffer: MADDPGReplayBuffer,
        training_config: TrainingConfig,
        checkpoint_config: Optional[CheckpointConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
        device: str = 'cuda',
        use_amp: bool = True,
    ):
        self.agent = agent
        self.buffer = replay_buffer
        self.config = training_config
        self.ckpt_config = checkpoint_config
        self.log_config = logging_config
        self.device = device
        self.use_amp = use_amp

        self.global_step = 0
        self.episode = 0
        self.best_reward = float('-inf')
        self._khop_neighbor_indices = None
        self._khop_neighbor_mask = None
        self.best_train_reward = float('-inf')
        self.total_steps = 0
        self.recent_rewards_window = []

        if _AMP_DEVICE is not None:
            self.scaler = GradScaler(_AMP_DEVICE, enabled=use_amp)
        else:
            self.scaler = GradScaler(enabled=use_amp)

    def _make_joint_onehot(self, actions_type: torch.Tensor) -> torch.Tensor:
        return MADDPGAgent.build_joint_actions_onehot(actions_type, self.agent.action_dim)

    @torch.no_grad()
    def _compute_target_actions_onehot(
        self,
        next_vehicle_features: torch.Tensor,
        next_context_features: torch.Tensor,
    ) -> torch.Tensor:
        B, N, vdim = next_vehicle_features.shape
        vf_flat = next_vehicle_features.reshape(B * N, vdim)
        cf_flat = next_context_features.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)

        action_type, _, _ = self.agent.actor_target.act(
            vehicle_feature=vf_flat,
            context_feature=cf_flat,
            deterministic=True,
        )
        action_type = action_type.reshape(B, N)
        return self._make_joint_onehot(action_type)

    def train_step(self) -> Dict[str, float]:
        if len(self.buffer) < self.config.batch_size:
            return {}

        batch: MADDPGBatch = self.buffer.sample(self.config.batch_size)
        device = torch.device(self.device)
        vf = batch.vehicle_states.to(device=device, non_blocking=True)
        cf = batch.context_states.to(device=device, non_blocking=True)
        next_vf = batch.next_vehicle_states.to(device=device, non_blocking=True)
        next_cf = batch.next_context_states.to(device=device, non_blocking=True)
        act_type = batch.actions_type.to(device=device, non_blocking=True)
        actions_repos = batch.actions_repos.to(device=device, non_blocking=True)
        rewards = batch.per_vehicle_rewards.to(device=device, non_blocking=True)
        dones = batch.dones.float().to(device=device, non_blocking=True)
        action_mask = batch.action_mask.to(device=device, non_blocking=True) if batch.action_mask is not None else None
        reposition_mask = batch.reposition_mask.to(device=device, non_blocking=True) if batch.reposition_mask is not None else None
        trip_mask = None
        vehicle_hex_ids = batch.vehicle_hex_ids.to(device=device, non_blocking=True) if batch.vehicle_hex_ids is not None else None
        reposition_candidate_hexes = None
        executed_actions_type = batch.executed_actions_type.to(device=device, non_blocking=True) if batch.executed_actions_type is not None else None
        executed_actions_repos = batch.executed_actions_repos.to(device=device, non_blocking=True) if batch.executed_actions_repos is not None else None

        with torch.no_grad():
            alpha = float(getattr(self.config, 'maddpg_reward_mix_alpha', 0.0))
            alpha = max(0.0, min(1.0, alpha))
            reward_global = rewards.mean(dim=1, keepdim=True).expand_as(rewards)
            rewards_mixed = (1.0 - alpha) * reward_global + alpha * rewards

            r_mean = rewards_mixed.mean()
            r_std = rewards_mixed.std().clamp(min=1e-6)
            rewards_norm = (rewards_mixed - r_mean) / r_std

            if executed_actions_type is not None:
                valid_exec_mask = (executed_actions_type >= 0).float()
            else:
                valid_exec_mask = torch.ones_like(rewards_norm)
            rewards_weighted = rewards_norm * valid_exec_mask

            next_actions_onehot = self._compute_target_actions_onehot(next_vf, next_cf)
            q_next = self.agent.critic_target(
                vehicle_features=next_vf,
                context_features=next_cf,
                actions_onehot=next_actions_onehot,
            )
            target_q = rewards_weighted + self.config.gamma * (1.0 - dones.unsqueeze(1)) * q_next.unsqueeze(1)
            target_q_mean = target_q.mean(dim=1)

        current_actions_onehot = self._make_joint_onehot(act_type)

        with autocast(_AMP_DEVICE or 'cpu', enabled=self.use_amp):
            q_current = self.agent.critic(
                vehicle_features=vf,
                context_features=cf,
                actions_onehot=current_actions_onehot,
            )
            critic_loss = F.mse_loss(q_current, target_q_mean.detach())

        self.agent.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.agent.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.config.max_grad_norm)
        self.scaler.step(self.agent.critic_optimizer)
        self.scaler.update()

        with autocast(_AMP_DEVICE or 'cpu', enabled=self.use_amp):
            soft_out = self.agent.gumbel_actions(
                vehicle_features=vf,
                context_features=cf,
                action_mask=action_mask,
                reposition_mask=None,
                trip_mask=None,
                khop_neighbor_indices=self._khop_neighbor_indices,
                khop_neighbor_mask=self._khop_neighbor_mask,
                vehicle_hex_ids=vehicle_hex_ids,
            )
            soft_actions = soft_out['action_type_soft']

            q_for_actor = self.agent.critic(
                vehicle_features=vf.detach(),
                context_features=cf.detach(),
                actions_onehot=soft_actions,
            )

            if executed_actions_type is not None:
                consistency_mask = (executed_actions_type >= 0).float()
                consistency_weight = consistency_mask.mean(dim=1)
                actor_loss = -(q_for_actor * consistency_weight).mean()
            else:
                actor_loss = -q_for_actor.mean()

            repos_aux_loss = torch.zeros((), device=q_for_actor.device, dtype=q_for_actor.dtype)
            if (
                self._khop_neighbor_indices is not None
                and vehicle_hex_ids is not None
                and executed_actions_repos is not None
            ):
                repos_mask = (act_type == 2) & (executed_actions_repos >= 0)
                if repos_mask.any():
                    cand_abs = self._khop_neighbor_indices[vehicle_hex_ids]
                    valid_slot = cand_abs >= 0
                    target_repos = executed_actions_repos.clamp(min=0)
                    slot_match = (cand_abs == target_repos.unsqueeze(-1)) & valid_slot
                    has_match = slot_match.any(dim=-1)
                    train_mask = repos_mask & has_match
                    if train_mask.any():
                        repos_soft = soft_out['reposition_soft']
                        target_slot = slot_match.float().argmax(dim=-1)
                        logp = torch.log(repos_soft.clamp(min=1e-8))
                        repos_ce = -logp.gather(-1, target_slot.unsqueeze(-1)).squeeze(-1)
                        repos_aux_loss = repos_ce[train_mask].mean()

            repos_aux_weight = float(getattr(self.config, 'maddpg_repos_aux_weight', 0.05))
            actor_loss = actor_loss + repos_aux_weight * repos_aux_loss

        self.agent.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.agent.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=self.config.max_grad_norm)
        self.scaler.step(self.agent.actor_optimizer)
        self.scaler.update()

        self.agent.soft_update_target()
        self.global_step += 1

        with torch.no_grad():
            avg_reward = rewards.mean().item()
            q_mean = q_current.mean().item()

            if executed_actions_type is not None:
                attempted_serve = (act_type == 0)
                attempted_charge = (act_type == 1)
                attempted_repos = (act_type == 2)

                serve_success = ((executed_actions_type == 0) & attempted_serve).sum().item()
                charge_success = ((executed_actions_type == 1) & attempted_charge).sum().item()
                repos_success = ((executed_actions_type == 2) & attempted_repos).sum().item()

                serve_attempted = attempted_serve.sum().item()
                charge_attempted = attempted_charge.sum().item()
                repos_attempted = attempted_repos.sum().item()

                serve_consistency = serve_success / max(1, serve_attempted)
                charge_consistency = charge_success / max(1, charge_attempted)
                reposition_consistency = repos_success / max(1, repos_attempted)
                failed_action_fraction = (executed_actions_type < 0).float().mean().item()
            else:
                serve_consistency = 1.0
                charge_consistency = 1.0
                reposition_consistency = 1.0
                failed_action_fraction = 0.0

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'repos_aux_loss': repos_aux_loss.item() if 'repos_aux_loss' in locals() else 0.0,
            'q_mean': q_mean,
            'avg_reward': avg_reward,
            'buffer_size': len(self.buffer),
            'serve_consistency': serve_consistency,
            'charge_consistency': charge_consistency,
            'reposition_consistency': reposition_consistency,
            'failed_action_fraction': failed_action_fraction,
        }

    def save_checkpoint(self, filename: str) -> None:
        from pathlib import Path
        save_dir = getattr(self.ckpt_config, 'checkpoint_dir', 'checkpoints')
        path = Path(save_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_target_state_dict': self.agent.actor_target.state_dict(),
            'critic_target_state_dict': self.agent.critic_target.state_dict(),
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.use_amp else None,
            'global_step': self.global_step,
            'episode': self.episode,
            'best_reward': self.best_reward,
            'best_train_reward': self.best_train_reward,
            'total_steps': self.total_steps,
            'recent_rewards_window': list(self.recent_rewards_window),
        }, path)
        print(f'[MADDPG] Checkpoint saved → {path}')

    def load_checkpoint(self, filename: str) -> None:
        from pathlib import Path
        p = Path(filename)
        save_dir = getattr(self.ckpt_config, 'checkpoint_dir', 'checkpoints')
        path = p if (p.is_absolute() or p.parent != Path('.')) else Path(save_dir) / filename
        if not path.exists():
            print(f'[MADDPG] Checkpoint not found: {path}')
            return
        ckpt = torch.load(path, map_location=self.device)
        def _shape_filtered_load(module, incoming, tag: str):
            model_state = module.state_dict()
            filtered = {}
            skipped = []
            for k, v in incoming.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered[k] = v
                elif k in model_state:
                    skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            module.load_state_dict(filtered, strict=False)
            if skipped:
                print(f'[MADDPG] Warning: skipped {tag} tensors with shape mismatch:')
                for key, old_shape, new_shape in skipped:
                    print(f'  - {key}: ckpt={old_shape}, model={new_shape}')

        _shape_filtered_load(self.agent.actor, ckpt['actor_state_dict'], 'actor')
        _shape_filtered_load(self.agent.critic, ckpt['critic_state_dict'], 'critic')
        _shape_filtered_load(self.agent.actor_target, ckpt['actor_target_state_dict'], 'actor_target')
        _shape_filtered_load(self.agent.critic_target, ckpt['critic_target_state_dict'], 'critic_target')
        self.agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.agent.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        if self.use_amp and ckpt.get('scaler') is not None:
            self.scaler.load_state_dict(ckpt['scaler'])
        self.global_step = ckpt.get('global_step', 0)
        self.episode = ckpt.get('episode', 0)
        self.best_reward = ckpt.get('best_reward', float('-inf'))
        self.best_train_reward = ckpt.get('best_train_reward', float('-inf'))
        self.total_steps = ckpt.get('total_steps', 0)
        self.recent_rewards_window = list(ckpt.get('recent_rewards_window', []))
        print(f'[MADDPG] Checkpoint loaded from {path} (ep={self.episode}, step={self.global_step})')
