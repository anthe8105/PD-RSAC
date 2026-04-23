"""
ppo_buffer.py — On-policy Rollout Buffer for Standard MAPPO.

Stores per-step transitions and supports:
- GAE computation (standard fixed γ, λ — no semi-MDP durations)
- Minibatch sampling for PPO update epochs
- Cleared after each rollout (on-policy)

Mask storage policy (RAM optimized):
  - action_mask     [V, A=3]         uint8 → stored directly
  - trip_mask       [V, max_trips]   uint8 → stored for acting/training parity
  - reposition_mask [V, H]           bool  → dropped for pure-MLP MAPPO
"""

import torch
from typing import Dict, List, Optional


class PPORolloutBuffer:
    """
    On-policy Rollout Buffer for Standard MAPPO.

    Store trajectories for:
    - Standard GAE computation
    - PPO minibatch update

    Cleared after each update (on-policy).
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """Clear all stored transitions. Called after each PPO update."""
        self.states:    List[Dict[str, torch.Tensor]] = []
        self.actions:   List[torch.Tensor] = []
        self.executed_actions: List[torch.Tensor] = []

        self.rewards:   List = []       # per-agent scalar reward [N_veh] or float
        self.dones:     List[bool] = []

        self.values:    List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # action_mask [V, A=3] stored as uint8 — ~6 MB per rollout, negligible.
        self.action_masks: List[Optional[torch.Tensor]] = []

        # trip_active_count: int — number of currently unassigned trips.
        self.trip_active_counts: List[Optional[int]] = []

        # trip_mask: [V, max_trips] bool — stored for acting/training parity.
        self.trip_masks: List[Optional[torch.Tensor]] = []

        # reposition_mask: [V, H] bool — stored for acting/training parity.
        self.reposition_masks: List[Optional[torch.Tensor]] = []

        self.vehicle_hex_ids: List[Optional[torch.Tensor]] = []

        # Computed post-rollout (used by legacy callers)
        self.advantages: Optional[torch.Tensor] = None
        self.returns:    Optional[torch.Tensor] = None

   

    def store(
        self,
        state:              Dict[str, torch.Tensor],
        action:             torch.Tensor,
        executed_action:    Optional[torch.Tensor] = None,
        reward                                     = 0.0,     # float or [N_veh] tensor
        duration:           float                  = 1.0,     # kept for compat — not used in training
        done:               bool                   = False,
        value:              torch.Tensor           = None,
        log_prob:           torch.Tensor           = None,
        action_mask:        Optional[torch.Tensor] = None,    # [V, A] bool
        reposition_mask:    Optional[torch.Tensor] = None,    # [V, H] per-vehicle reposition feasibility
        trip_mask:          Optional[torch.Tensor] = None,    # [V, max_trips] per-vehicle feasibility
        trip_active_count:  Optional[int]          = None,    # compact trip count
        vehicle_hex_ids:    Optional[torch.Tensor] = None,    # [V] absolute hex id per vehicle
    ):
        self.states.append(state)
        self.actions.append(action)
        self.executed_actions.append(
            executed_action if executed_action is not None else action)

        # Store reward — tensor or scalar
        if isinstance(reward, torch.Tensor):
            self.rewards.append(reward.clone().detach())
        else:
            self.rewards.append(float(reward))

        self.dones.append(bool(done.item()) if isinstance(done, torch.Tensor) else bool(done))

        self.values.append(value.detach())
        self.log_probs.append(log_prob.detach())

        # action_mask: store as uint8 to save memory vs bool (same size, explicit)
        if action_mask is not None:
            self.action_masks.append(action_mask.to(torch.uint8).detach().cpu())
        else:
            self.action_masks.append(None)

        # trip_active_count: compact int storage
        self.trip_active_counts.append(trip_active_count)

        if vehicle_hex_ids is not None:
            self.vehicle_hex_ids.append(vehicle_hex_ids.to(torch.long).detach().cpu())
        else:
            self.vehicle_hex_ids.append(None)

        if trip_mask is not None:
            self.trip_masks.append(trip_mask.to(torch.uint8).detach().cpu())
        else:
            self.trip_masks.append(None)

        if reposition_mask is not None:
            self.reposition_masks.append(reposition_mask.to(torch.uint8).detach().cpu())
        else:
            self.reposition_masks.append(None)

    def __len__(self):
        return len(self.rewards)

    def size(self):
        return len(self.rewards)


    def get_tensors(self, device):
        """Convert stored data into tensors."""

        def _sq(t):
            """Squeeze leading dim-1 batch dimension [T, 1, ...] → [T, ...]."""
            if t.dim() >= 3 and t.shape[1] == 1:
                return t.squeeze(1)
            if t.dim() == 2 and t.shape[1] == 1:
                return t.squeeze(1)
            return t

        # Rewards — enforce shape-consistent [T, V] tensorization
        num_vehicles = self.actions[0].shape[0] if self.actions else 1
        reward_tensors = []
        for r in self.rewards:
            if isinstance(r, torch.Tensor):
                rr = r.clone().detach().to(dtype=torch.float32)
                if rr.dim() == 0:
                    rr = rr.expand(num_vehicles)
                elif rr.dim() > 1:
                    rr = rr.view(-1)
                if rr.numel() == 1:
                    rr = rr.expand(num_vehicles)
                if rr.numel() != num_vehicles:
                    raise ValueError(f'PPO reward shape mismatch: expected {num_vehicles}, got {rr.numel()}')
                reward_tensors.append(rr)
            else:
                reward_tensors.append(torch.full((num_vehicles,), float(r), dtype=torch.float32))
        rewards  = _sq(torch.stack(reward_tensors).to(device, dtype=torch.float32))
        dones    = _sq(torch.tensor(self.dones, dtype=torch.float32, device=device))

        actions          = _sq(torch.stack(self.actions).to(device))
        executed_actions = _sq(torch.stack(self.executed_actions).to(device))
        values           = _sq(torch.stack(self.values).to(device))
        log_probs        = _sq(torch.stack(self.log_probs).to(device))

        # action_masks: [T, V, A] uint8 — populated when provided
        def _stack_action_masks(mask_list):
            if all(m is None for m in mask_list):
                return None
            proto  = next(m for m in mask_list if m is not None)
            padded = [m if m is not None else torch.ones_like(proto) for m in mask_list]
            return _sq(torch.stack(padded).to(device, dtype=torch.bool))

        action_masks = _stack_action_masks(self.action_masks)

        # trip_active_counts: [T] int64 — None entries become -1 (no trips active)
        if any(c is not None for c in self.trip_active_counts):
            trip_active_counts = torch.tensor(
                [c if c is not None else 0 for c in self.trip_active_counts],
                dtype=torch.long, device=device
            )
        else:
            trip_active_counts = None

        def _stack_trip_masks(mask_list):
            if all(m is None for m in mask_list):
                return None
            proto = next(m for m in mask_list if m is not None)
            padded = [m if m is not None else torch.zeros_like(proto) for m in mask_list]
            return _sq(torch.stack(padded).to(device, dtype=torch.bool))

        trip_masks = _stack_trip_masks(self.trip_masks)

        def _stack_reposition_masks(mask_list):
            if all(m is None for m in mask_list):
                return None
            proto = next(m for m in mask_list if m is not None)
            padded = [m if m is not None else torch.zeros_like(proto) for m in mask_list]
            return _sq(torch.stack(padded).to(device, dtype=torch.bool))

        reposition_masks = _stack_reposition_masks(self.reposition_masks)

        def _stack_vehicle_hex_ids(hex_list):
            if all(h is None for h in hex_list):
                return None
            proto = next(h for h in hex_list if h is not None)
            padded = [h if h is not None else torch.zeros_like(proto) for h in hex_list]
            return _sq(torch.stack(padded).to(device, dtype=torch.long))

        vehicle_hex_ids = _stack_vehicle_hex_ids(self.vehicle_hex_ids)

        return {
            'rewards':            rewards,
            'dones':              dones,
            'actions':            actions,
            'executed_actions':   executed_actions,
            'values':             values,
            'log_probs':          log_probs,
            'states':             self.states,
            'action_masks':       action_masks,          # [T, V, A] bool or None
            'trip_active_counts': trip_active_counts,    # [T] int64 or None
            'reposition_masks':   reposition_masks,      # [T, V, H] bool or None
            'trip_masks':         trip_masks,            # [T, V, max_trips] bool or None
            'vehicle_hex_ids':    vehicle_hex_ids,       # [T, V] long or None
            'advantages':  self.advantages.to(device) if self.advantages is not None else None,
            'returns':     self.returns.to(device)    if self.returns    is not None else None,
        }


    def sample_minibatches(self, batch_size: int, device):
        """Yield randomized minibatches for PPO update epochs."""
        data    = self.get_tensors(device)
        N       = len(data['rewards'])
        indices = torch.randperm(N)

        for start in range(0, N, batch_size):
            mb_idx = indices[start:start + batch_size]
            batch  = {'idx': mb_idx}

            batch['states'] = self._batch_states(mb_idx, data['states'], device)

            for k in ['actions', 'executed_actions', 'values', 'log_probs',
                      'rewards', 'dones', 'advantages', 'returns']:
                if data[k] is not None:
                    batch[k] = data[k][mb_idx]
                else:
                    batch[k] = None

            # Compact mask fields
            batch['action_masks']       = data['action_masks'][mb_idx]       if data['action_masks']       is not None else None
            batch['trip_active_counts'] = data['trip_active_counts'][mb_idx] if data['trip_active_counts'] is not None else None

            yield batch


    def _batch_states(self, indices, states, device):
        batch = {}
        keys  = states[0].keys()
        # Convert to plain Python ints — PyTorch 0-dim tensors cause TypeError
        # when used as Python list indices (e.g. states[tensor(3)] fails).
        int_indices = [int(i) for i in indices]
        for k in keys:
            tensors = [states[i][k] for i in int_indices]
            stacked = torch.stack(tensors).to(device)
            if stacked.dim() >= 3 and stacked.shape[1] == 1:
                stacked = stacked.squeeze(1)
            batch[k] = stacked
        return batch
