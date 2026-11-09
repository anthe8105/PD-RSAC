"""
Trip arrivals for the EV fleet environment.

With real data the manager replays a ``TripDataset`` on a wall clock (workstream B3/B4):

- an episode starts at a calendar time ``t0`` (drawn uniformly from the split's step-aligned
  starts with the environment's generator, or fixed by the caller for evaluation);
- the decision epoch ``k`` is ``t_k = t0 + k * step`` and sees the trips whose pickup time
  lies in ``(t_k - step, t_k]`` (arrival convention, audit D29);
- trip endpoints are domain indices from the manifest: no modulo remapping (D12) and no
  rewriting of intra-cell destinations (D11, F29);
- an empty window is genuine zero demand; nothing is synthesised in its place (D26).

Without data (tests, smoke runs) trips are generated synthetically.
"""

import datetime as _dt
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..config import Config
    from ..state import TensorTripState
    from ..spatial import HexGrid
    from ..data.dataset import TripDataset

# Synthetic demand mirrors the real experiment per vehicle: new orders per vehicle per 5-min
# epoch by hour of day, measured on a 10 % sample of the January 2009 training days with 1000
# vehicles (manifest pickup_by_hour) and scaled linearly to ``data.sample_ratio``. Fares follow
# the least-squares fit fare = 3.50 + 1.37 * km.
SYNTHETIC_RATE_SAMPLE = 0.1
SYNTHETIC_RATE_BY_HOUR = (0.129, 0.094, 0.071, 0.053, 0.037, 0.030, 0.067, 0.123, 0.163, 0.163, 0.148, 0.153,
                          0.167, 0.169, 0.176, 0.177, 0.168, 0.192, 0.229, 0.235, 0.204, 0.192, 0.186, 0.162)
SYNTHETIC_FARE = (3.50, 1.37)

# Synthetic runs start at a fixed Monday 08:00 so clock features are well defined.
SYNTHETIC_START_S = int(_dt.datetime(2009, 1, 5, 8, 0).replace(tzinfo=_dt.timezone.utc).timestamp())


class TripManager:
    """Loads trip arrivals into the trip state, from data or synthetically."""

    def __init__(
        self,
        config: 'Config',
        hex_grid: 'HexGrid',
        trip_state: 'TensorTripState',
        trip_loader: Optional['TripDataset'],
        device: torch.device,
        num_hexes: int,
        num_vehicles: int,
        episode_steps: int,
        generator: Optional[torch.Generator] = None,
    ):
        self.config = config
        # CPU generator owned by the environment (A6); None falls back to a fresh generator.
        self._rng = generator if generator is not None else torch.Generator()
        self.hex_grid = hex_grid
        self.trip_state = trip_state
        self.dataset = trip_loader
        self.device = device
        self.num_hexes = num_hexes
        self.num_vehicles = num_vehicles
        self.episode_steps = episode_steps
        self.step_s = int(round(config.episode.step_duration_minutes * 60))

        self._next_trip_id = 0
        self.episode_start_s = SYNTHETIC_START_S
        self.trips_loaded = 0       # admitted (in-domain) arrivals this episode
        self.out_of_domain = 0      # arrivals dropped because an endpoint is outside the domain

    @property
    def uses_data(self) -> bool:
        return self.dataset is not None

    def epoch_s(self, step: int) -> int:
        """Wall-clock time (epoch seconds, naive local time) of decision epoch ``step``."""
        return self.episode_start_s + step * self.step_s

    def reset(self, episode_idx: int = 0, episode_start_s: Optional[int] = None):
        """Start a new episode. ``episode_start_s`` fixes the start (evaluation); otherwise it is drawn."""
        del episode_idx
        self._next_trip_id = 0
        self.trips_loaded = 0
        self.out_of_domain = 0
        if episode_start_s is not None:
            self.episode_start_s = int(episode_start_s)
        elif self.uses_data:
            starts = self.dataset.episode_starts(self.episode_steps, self.config.episode.step_duration_minutes)
            pick = torch.randint(0, len(starts), (1,), generator=self._rng).item()
            self.episode_start_s = int(starts[pick])
        else:
            self.episode_start_s = SYNTHETIC_START_S

    def calculate_max_trips(self) -> int:
        """Initial buffer size; the trip state grows on demand, so this is only a hint."""
        if self.uses_data:
            per_step = self.dataset.mean_arrivals_per_step(self.config.episode.step_duration_minutes)
            return max(1024, int(per_step * (self.config.reward.max_wait_steps + 2) * 4))
        return max(1024, max(10, self.num_vehicles // 10) * 20)

    # ---------------------------------------------------------------- loading
    def load_initial_trips(self):
        if self.uses_data:
            self._load_window(0, initial=True)
        else:
            self._generate_synthetic(self._synthetic_count(0), initial=True)

    def load_step_trips(self, current_step: int):
        """Arrivals of decision epoch ``current_step`` >= 1 (epoch 0 is loaded by ``load_initial_trips``)."""
        if self.uses_data:
            self._load_window(current_step, initial=False)
        else:
            self._generate_synthetic(self._synthetic_count(current_step), initial=False)

    def _load_window(self, step: int, initial: bool):
        t_hi = self.epoch_s(step)
        t_lo = t_hi - self.step_s
        self.out_of_domain += self.dataset.out_of_domain_between(t_lo, t_hi)
        trips = self.dataset.trips_between(t_lo, t_hi, self.device)
        self._admit(trips["pickup"], trips["dropoff"], trips["fare"], trips["distance_km"], initial)

    def _synthetic_count(self, step: int) -> int:
        hour = (self.epoch_s(step) % 86400) // 3600
        lam = SYNTHETIC_RATE_BY_HOUR[hour] * self.num_vehicles * self.config.data.sample_ratio / SYNTHETIC_RATE_SAMPLE
        return int(torch.poisson(torch.tensor(float(lam)), generator=self._rng).item())

    def _generate_synthetic(self, n: int, initial: bool):
        pickup = torch.randint(0, self.num_hexes, (n,), generator=self._rng).to(self.device)
        dropoff = torch.randint(0, self.num_hexes, (n,), generator=self._rng).to(self.device)
        distances = self.hex_grid.distance_matrix.get_distances_batch(pickup, dropoff)
        fares = SYNTHETIC_FARE[0] + SYNTHETIC_FARE[1] * distances
        self._admit(pickup, dropoff, fares, distances, initial)

    def _admit(self, pickup, dropoff, fares, distances, initial: bool):
        n = len(pickup)
        if initial:
            ids = torch.arange(self._next_trip_id, self._next_trip_id + n, device=self.device)
            self.trip_state.load_trips(trip_ids=ids, pickup_hexes=pickup, dropoff_hexes=dropoff,
                                       fares=fares, distances=distances)
        elif n > 0:
            ids = torch.arange(self._next_trip_id, self._next_trip_id + n, device=self.device)
            self.trip_state.add_trips(trip_ids=ids, pickup_hexes=pickup, dropoff_hexes=dropoff,
                                      fares=fares, distances=distances)
        self._next_trip_id += n
        self.trips_loaded += n
