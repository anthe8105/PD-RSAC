"""
GPU environment for the EV fleet (workstream C).

Decision rhythm (C1, audit F52, N3): ``reset`` and ``step`` return the **pre-decision state** of
epoch k: commitments ending at k are completed and the arrivals of (t_k - dt, t_k] are loaded
before the observation is built. Policy, projection and execution therefore see the same
vehicles, ports and orders. Inside ``step`` (epoch k):

1. the executor applies the actions of the vehicles available at k (``executor.py``);
2. vehicles committed to charging during k charge one step (power measured on the feeder);
3. orders still waiting pay the wait penalty, age by one epoch, and expire after
   ``max_wait_steps`` epochs (drop penalty);
4. the clock moves to k + 1, completions and arrivals of k + 1 are applied, and the next
   pre-decision observation is returned.

Rewards follow Eq. (8) per vehicle (``reward.py``); the fleet reward is their sum.
The feeder is measured, never enforced (Q9); ports are a physical constraint.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..config import Config
from ..data.dataset import TripDataset
from ..features.builder import CONTEXT_DIM, FeatureBuilder
from ..spatial import HexGrid
from ..spatial.manifest import SpatialManifest
from ..state import TensorFleetState, TensorStationState, TensorTripState, VehicleStatus
from .executor import EXEC_CHARGE, EXEC_SERVE, EXEC_STAY, ActionExecutor, Outcome
from .candidates import CandidateGenerator
from .physics import Physics
from .reward import RewardComputer
from .trip_manager import TripManager

FEEDER_TOL_KW = 1e-3   # a feeder filled exactly by the projection may exceed P_feed by float rounding


@dataclass
class EnvInfo:
    """Episode totals (cumulative) plus the last step's details."""
    trips_served: int = 0          # orders accepted (fare credited at acceptance)
    trips_completed: int = 0       # orders delivered
    trips_dropped: int = 0
    trips_loaded: int = 0
    trips_out_of_domain: int = 0
    revenue: float = 0.0           # fares of accepted orders
    driving_cost: float = 0.0      # every driven km (serve, reposition, drive to a station)
    energy_cost: float = 0.0       # electricity bought
    wait_penalty: float = 0.0
    drop_penalty: float = 0.0
    energy_charged_kwh: float = 0.0  # stored in batteries
    energy_bought_kwh: float = 0.0   # drawn from the grid
    feeder_load_kw: float = 0.0      # grid power drawn during the last step
    vehicles_charging_now: int = 0   # vehicles that charged during the last step
    feeder_peak_kw: float = 0.0
    feeder_violation_steps: int = 0
    infeasible_executed: int = 0     # actions that could not be executed (vehicle stayed)
    serve_unmatched: int = 0         # raw path: SERVE vehicles left without an order
    charge_no_port: int = 0          # raw path: CHARGE with no free port among the candidates
    power_clipped: int = 0           # exact path: power outside [0, min(P_max, headroom)]
    vehicles_charging: int = 0
    vehicles_serving: int = 0
    vehicles_idle: int = 0
    avg_soc: float = 0.0             # mean battery level, % of capacity
    step: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # last step
    serve_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    charge_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    per_vehicle_reward: Optional[torch.Tensor] = None   # scaled, sums to the step reward
    executed: Optional[Outcome] = None
    deciders: Optional[torch.Tensor] = None
    num_serve_attempted: int = 0
    num_serve_success: int = 0
    num_charge_attempted: int = 0
    num_charge_success: int = 0

    @property
    def profit(self) -> float:
        return self.revenue - self.driving_cost - self.energy_cost


class GPUEnvironmentV2:

    def __init__(
        self,
        config: Config,
        hex_grid: HexGrid,
        trip_loader: Optional[TripDataset] = None,
        device: str = "cuda",
        manifest: Optional[SpatialManifest] = None,
    ):
        """``trip_loader`` is the split's TripDataset (None = synthetic trips); ``manifest`` fixes
        the stations and the initial vehicle distribution. Real data requires a manifest."""
        if trip_loader is not None and manifest is None:
            raise ValueError("real trip data needs the spatial manifest (stations, domain)")
        self.config = config
        self.hex_grid = hex_grid
        self.trip_loader = trip_loader
        self.manifest = manifest
        self.device = torch.device(device)

        self.num_vehicles = config.environment.num_vehicles
        self.num_hexes = config.environment.num_hexes
        self.num_stations = config.environment.num_stations
        self.episode_steps = config.episode.steps_per_episode
        if self.num_stations <= 0:
            raise ValueError("environment.num_stations must be positive")

        self.physics = Physics.from_config(config)
        self.feature_builder = FeatureBuilder(hex_grid=hex_grid, num_vehicles=self.num_vehicles,
                                              max_soc=self.physics.battery_kwh, device=device)

        self.fleet_state: Optional[TensorFleetState] = None
        self.trip_state: Optional[TensorTripState] = None
        self.station_state: Optional[TensorStationState] = None
        self.current_step = 0
        self.episode_reward = 0.0
        self._rng = torch.Generator()       # all environment randomness (A6)
        self._rng.manual_seed(0)
        self.episode_info = EnvInfo()
        self.debug_invariants = False

        # Feature widths used by the learning code
        self._action_dim = 3                 # SERVE=0, CHARGE=1, REPOSITION=2
        self._vehicle_feature_dim = 16
        self._fleet_vehicle_feature_dim = 13
        self._hex_feature_dim = 5
        self._context_dim = CONTEXT_DIM
        self._state_dim = (self.num_vehicles * self._vehicle_feature_dim
                           + self.num_hexes * self._hex_feature_dim + self._context_dim)

        self._distances = hex_grid.distance_matrix._distances.to(self.device)
        adj = hex_grid.get_adjacency_matrix().to(self.device)
        if tuple(adj.shape) != (self.num_hexes, self.num_hexes):
            raise ValueError(f"adjacency shape {tuple(adj.shape)} does not match {self.num_hexes} cells")
        self._ring1 = (adj > 0) | torch.eye(self.num_hexes, dtype=torch.bool, device=self.device)
        # Largest pickup distance inside a 1-ring (features and the pre-D MILP adapter use km).
        self.max_pickup_distance = float(self._distances[self._ring1].max())
        self._compute_adjacency_matrix(adj)

        self._station_cells = self._select_station_hexes()
        self.candidates = CandidateGenerator(self.physics, self._distances, self._ring1, self._station_cells)
        self.candidates.build_slots(list(self.hex_grid.hex_ids))
        self._executor = ActionExecutor(self.physics, self._distances, self._ring1, self._station_cells,
                                        config.station.num_ports, self.device, candidates=self.candidates)
        self._reward = RewardComputer(config, self._ring1, self.device)
        self._start_weights = self._initial_position_weights()
        self._trip_manager: Optional[TripManager] = None

    # ------------------------------------------------------------------ setup
    def _select_station_hexes(self) -> torch.Tensor:
        """Station cells: from the manifest with real data; placed on the fly for synthetic grids."""
        if self.manifest is not None:
            if len(self.manifest.station_cells) != self.num_stations:
                raise ValueError(f"config has {self.num_stations} stations, manifest has {len(self.manifest.station_cells)}")
            if self.manifest.ports_per_station != self.config.station.num_ports:
                raise ValueError(f"config has {self.config.station.num_ports} ports per station, "
                                 f"manifest has {self.manifest.ports_per_station}")
            if list(self.hex_grid.hex_ids) != list(self.manifest.cells):
                raise ValueError("hex grid cells differ from the manifest cells")
            return torch.tensor(self.manifest.station_indices, dtype=torch.long, device=self.device)
        if self.num_stations > self.num_hexes:
            raise ValueError(f"{self.num_stations} stations for {self.num_hexes} cells")
        ids = list(self.hex_grid.hex_ids)
        import h3
        if all(isinstance(h, str) and h3.is_valid_cell(h) for h in ids):
            from ..spatial.manifest import place_stations
            cells, _ = place_stations(ids, {h: 1.0 for h in ids}, self.num_stations,
                                      self.config.station.cover_rings, 1.0)
            return torch.tensor([ids.index(c) for c in cells], dtype=torch.long, device=self.device)
        step = self.num_hexes / self.num_stations  # non-H3 synthetic grid: evenly spaced indices
        return torch.tensor([int(k * step) for k in range(self.num_stations)], dtype=torch.long, device=self.device)

    def _initial_position_weights(self) -> Optional[torch.Tensor]:
        """[24, H] pickup counts of the training split by hour of day (C6), or None (uniform)."""
        if self.manifest is None:
            return None
        w = torch.tensor(self.manifest.pickup_by_hour, dtype=torch.float64)
        if tuple(w.shape) != (24, self.num_hexes) or (w.sum(dim=1) <= 0).any():
            raise ValueError("manifest pickup_by_hour must be 24 x cells with positive rows; rebuild the manifest")
        return w

    def _compute_adjacency_matrix(self, adj: torch.Tensor):
        """GCN graph = 1-ring adjacency with self-loops, D^-1/2 (A + I) D^-1/2 (F36)."""
        a = torch.clamp(adj + torch.eye(self.num_hexes, device=self.device), 0, 1)
        d = a.sum(dim=1).pow(-0.5)
        self._adjacency_matrix = d.unsqueeze(1) * a * d.unsqueeze(0)

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        return self._adjacency_matrix

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def clock_s(self) -> int:
        """Wall-clock time of the current decision epoch (naive local time, epoch seconds)."""
        return self._trip_manager.epoch_s(self.current_step)

    def epoch_clock_s(self, step: int) -> int:
        """Wall-clock time of decision epoch ``step`` of the current episode."""
        return self._trip_manager.epoch_s(step)

    # ------------------------------------------------------------------ episode
    def reset(
        self,
        seed: Optional[int] = None,
        episode_idx: int = 0,
        fleet_state_only: bool = False,
        episode_start_s: Optional[int] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if seed is not None:
            self._rng.manual_seed(int(seed))

        self.trip_state = TensorTripState(max_trips=self._calculate_max_trips(), device=str(self.device))
        self._trip_manager = TripManager(
            config=self.config, hex_grid=self.hex_grid, trip_state=self.trip_state,
            trip_loader=self.trip_loader, device=self.device, num_hexes=self.num_hexes,
            num_vehicles=self.num_vehicles, episode_steps=self.episode_steps, generator=self._rng,
        )
        self._trip_manager.reset(episode_idx, episode_start_s=episode_start_s)

        # Initial fleet (C6): energy U[0.5, 0.9] x E_max; positions ~ training pickups of the start hour.
        energy = self.physics.initial_energy(self.num_vehicles, self._rng)
        self.fleet_state = TensorFleetState(self.num_vehicles, device=str(self.device),
                                            max_soc=self.physics.battery_kwh, initial_energy=energy)
        if self._start_weights is not None:
            hour = (self._trip_manager.epoch_s(0) % 86400) // 3600
            pos = torch.multinomial(self._start_weights[hour], self.num_vehicles, replacement=True, generator=self._rng)
        else:
            pos = torch.randint(0, self.num_hexes, (self.num_vehicles,), generator=self._rng)
        self.fleet_state.positions.copy_(pos.to(self.device))

        self.station_state = TensorStationState(self.num_stations, device=str(self.device),
                                                num_ports=self.config.station.num_ports,
                                                max_power=self.physics.p_max_kw)
        self.station_state.set_locations(self._station_cells)

        self._trip_manager.load_initial_trips()
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_info = EnvInfo()
        self.episode_info.trips_loaded = self._trip_manager.trips_loaded
        self.episode_info.trips_out_of_domain = self._trip_manager.out_of_domain
        self._refresh_info()
        return self.get_fleet_replay_state() if fleet_state_only else self._get_state()

    def _calculate_max_trips(self) -> int:
        """Initial trip-buffer size; the buffer grows on demand, so nothing is dropped (D31, D32)."""
        if self.trip_loader is not None:
            per_step = self.trip_loader.mean_arrivals_per_step(self.config.episode.step_duration_minutes)
            return max(1024, int(per_step * (self.config.reward.max_wait_steps + 2) * 4))
        return max(1024, max(10, self.num_vehicles // 10) * 20)

    def step(
        self,
        action_type: torch.Tensor,
        reposition_target: Optional[torch.Tensor] = None,
        vehicle_charge_power: Optional[torch.Tensor] = None,
        fleet_state_only: bool = False,
        milp_serve_trip_ids: Optional[torch.Tensor] = None,
        charge_station: Optional[torch.Tensor] = None,
        session_power_kw: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, EnvInfo]:
        """One decision epoch. ``action_type`` [V]: -1 no-op, 0 SERVE, 1 CHARGE, 2 REPOSITION;
        ``vehicle_charge_power`` is, for a CHARGE booking, the share of the battery's free capacity
        (at arrival) to add during the session; ``session_power_kw`` [V] sets this epoch's power of
        vehicles in a charging session (projection; default: P_max^s); ``milp_serve_trip_ids`` (trip slots)
        selects the exact path used by the projection; ``charge_station`` [V] (-1 = nearest)."""
        k = self.current_step
        fleet, trips, info, ph = self.fleet_state, self.trip_state, self.episode_info, self.physics
        deciders = fleet.get_available_mask()
        vehicle_cells = fleet.positions.clone()

        # 1) actions
        out = self._executor.execute(
            fleet, trips, k, action_type.view(-1), repos_target=reposition_target, station=charge_station,
            power_frac=vehicle_charge_power, serve_trip=milp_serve_trip_ids, exact=milp_serve_trip_ids is not None)
        stayed = out.mode == EXEC_STAY
        fleet.idle_steps[stayed] += 1
        fleet.idle_steps[(out.mode >= 0) & ~stayed] = 0

        # 2) charging during epoch k (vehicles committed to charge until k + 1)
        charging_now = (fleet.status == VehicleStatus.CHARGING) & (fleet.busy_until == k + 1)
        bought = torch.zeros(self.num_vehicles, device=self.device)
        if charging_now.any():
            if session_power_kw is not None:                 # projection: this epoch's power (feeder)
                req = session_power_kw.to(self.device).float().view(-1)[charging_now]
                hi = ph.max_power(fleet.socs[charging_now])
                clipped = int(((req > hi + 1e-4) | (req < -1e-6)).sum())
                out.counts["power_clipped"] += clipped
                fleet.charge_power[charging_now] = torch.minimum(req.clamp(min=0.0), hi)
            e_new, stored, grid = ph.charge(fleet.socs[charging_now], fleet.charge_power[charging_now])
            fleet.socs[charging_now] = e_new
            bought[charging_now] = grid
            # session end: the target energy is reached (or the battery is full)
            idx = charging_now.nonzero(as_tuple=True)[0]
            done = (e_new >= fleet.charge_target[idx] - 1e-4) | (ph.max_power(e_new) <= 1e-3)
            if done.any():
                fleet.end_charging(idx[done])
            if (~done).any():
                fleet.busy_until[idx[~done]] = k + 2          # still charging during the next epoch
            info.energy_charged_kwh += float(stored.sum())
        feeder_kw = float(bought.sum()) / ph.step_hours

        # 3) waiting orders: wait penalty, ageing, expiry
        waiting = trips.get_unassigned_mask()
        w_idx = waiting.nonzero(as_tuple=True)[0]
        pen = torch.full((len(w_idx),), self._reward.wait_penalty, device=self.device)
        trips.increment_wait()
        expired = trips.get_expired_mask(self._reward.max_wait_steps)
        n_drop = trips.drop_expired(self._reward.max_wait_steps)
        pen = pen + expired[w_idx].float() * self._reward.drop_penalty

        # 4) rewards (Eq. 8), per vehicle, then the fleet sum
        share = self._reward.split_order_penalties(trips.pickup_hex[w_idx], pen, vehicle_cells, deciders)
        per_vehicle = out.fare - self._reward.c_drv * out.drive_km - self._reward.c_elec * bought - share
        reward_unscaled = per_vehicle.sum()
        per_vehicle = per_vehicle / self._reward.scale
        reward = (reward_unscaled / self._reward.scale).view(1)

        # bookkeeping
        n_srv = int((out.mode == EXEC_SERVE).sum())
        info.trips_served += n_srv
        info.trips_dropped += n_drop
        info.revenue += float(out.fare.sum())
        info.driving_cost += float(out.drive_km.sum()) * self._reward.c_drv
        info.energy_cost += float(bought.sum()) * self._reward.c_elec
        info.energy_bought_kwh += float(bought.sum())
        info.wait_penalty += len(w_idx) * self._reward.wait_penalty
        info.drop_penalty += n_drop * self._reward.drop_penalty
        info.feeder_load_kw = feeder_kw
        info.vehicles_charging_now = int(charging_now.sum())
        info.feeder_peak_kw = max(info.feeder_peak_kw, feeder_kw)
        info.feeder_violation_steps += int(feeder_kw > ph.feeder_kw + FEEDER_TOL_KW)   # float rounding of a full feeder is not a violation
        for key in ("infeasible", "serve_unmatched", "charge_no_port", "power_clipped"):
            name = "infeasible_executed" if key == "infeasible" else key
            setattr(info, name, getattr(info, name) + out.counts[key])
        srv_v = (out.mode == EXEC_SERVE).nonzero(as_tuple=True)[0]
        chg_v = (out.mode == EXEC_CHARGE).nonzero(as_tuple=True)[0]
        info.serve_assignments = (srv_v, out.trip[srv_v])
        info.charge_assignments = (chg_v, out.station[chg_v])
        info.num_serve_attempted = int((deciders & (action_type.view(-1).to(self.device) == 0)).sum())
        info.num_serve_success = n_srv
        info.num_charge_attempted = int((deciders & (action_type.view(-1).to(self.device) == 1)).sum())
        info.num_charge_success = len(chg_v)
        info.per_vehicle_reward = per_vehicle
        info.executed = out
        info.deciders = deciders
        self.episode_reward += float(reward)

        # 5) advance to the next pre-decision state
        self.current_step = k + 1
        done = self.current_step >= self.episode_steps
        self._pre_decision(load_arrivals=not done)
        if self.debug_invariants:
            self.check_invariants()
        next_state = self.get_fleet_replay_state() if fleet_state_only else self._get_state()
        return next_state, reward, torch.tensor([done], device=self.device), info

    def _pre_decision(self, load_arrivals: bool) -> None:
        """Complete commitments ending now and load this epoch's arrivals."""
        fleet, trips = self.fleet_state, self.trip_state
        slots = fleet.current_trip.clone()
        serve_done, _, _, _ = fleet.complete(self.current_step)
        if serve_done.any():
            done_slots = slots[serve_done]
            trips.valid_mask[done_slots] = False
            trips.assigned[done_slots] = False
            trips.assigned_vehicle[done_slots] = -1
            self.episode_info.trips_completed += int(serve_done.sum())
        if load_arrivals:
            self._trip_manager.load_step_trips(self.current_step)
        self.station_state.sync(fleet)
        self.episode_info.trips_loaded = self._trip_manager.trips_loaded
        self.episode_info.trips_out_of_domain = self._trip_manager.out_of_domain
        self._refresh_info()

    def _refresh_info(self) -> None:
        info, fleet = self.episode_info, self.fleet_state
        info.step = self.current_step
        info.avg_soc = float(fleet.socs.mean()) / self.physics.battery_kwh * 100.0
        st = fleet.status
        info.vehicles_idle = int((st == VehicleStatus.IDLE).sum())
        info.vehicles_serving = int((st == VehicleStatus.SERVING).sum())
        info.vehicles_charging = int(((st == VehicleStatus.CHARGING) | (st == VehicleStatus.CHARGE_TRAVEL)).sum())

    def check_invariants(self) -> None:
        """Debug checks: ports, energy bounds, commitments, trip conservation."""
        fleet, trips, ph = self.fleet_state, self.trip_state, self.physics
        holders = fleet.holds_port_mask()
        occ = torch.bincount(fleet.charging_station[holders], minlength=self.num_stations)
        assert torch.equal(occ.to(torch.int32), self.station_state.occupied), "port bookkeeping drifted"
        assert (self.station_state.occupied <= self.station_state.ports).all(), "port capacity exceeded"
        assert (fleet.socs >= -1e-4).all() and (fleet.socs <= ph.battery_kwh + 1e-4).all(), "energy out of bounds"
        busy = fleet.busy_until >= 0
        assert (fleet.busy_until[busy] > self.current_step).all(), "commitment already over"
        committed = (fleet.status == VehicleStatus.SERVING) | (fleet.status == VehicleStatus.REPOSITIONING) \
            | (fleet.status == VehicleStatus.CHARGE_TRAVEL)
        assert busy[committed].all(), "moving vehicle without an end epoch"
        assert not busy[fleet.status == VehicleStatus.IDLE].any(), "idle vehicle with a commitment"
        unassigned = int(trips.get_unassigned_mask().sum())
        assert trips.n_admitted_total == trips.n_assigned_total + trips.n_expired_total + unassigned, "trips lost"

    # ------------------------------------------------------------------ observations
    def get_available_actions(self) -> torch.Tensor:
        """[V, 3] modes a vehicle may choose now (all three for vehicles that decide)."""
        avail = self.fleet_state.get_available_mask()
        return avail.unsqueeze(1).expand(-1, self._action_dim).clone()

    def get_fleet_policy_inputs(self) -> Dict[str, torch.Tensor]:
        """Fleet-level tensors needed for action selection."""
        vehicle_hex_ids = self.fleet_state.positions.long()
        return {
            'hex_features': self.feature_builder.build_hex_features(
                self.fleet_state, self.trip_state, self.station_state, self.current_step),
            'context_features': self.feature_builder.build_context_features(
                self.clock_s, self.fleet_state, self.trip_state),
            'hex_vehicle_summary': self.feature_builder.build_hex_vehicle_summary(
                fleet=self.fleet_state, vehicle_hex_ids=vehicle_hex_ids, current_step=self.current_step),
            'vehicle_hex_ids': vehicle_hex_ids,
            'vehicle_socs': self.fleet_state.socs,
            'vehicle_status': self.fleet_state.status,
            'idle_steps': self.fleet_state.idle_steps,
            'available_mask': self.fleet_state.get_available_mask(),
        }

    def get_fleet_replay_state(self) -> Dict[str, torch.Tensor]:
        return self.feature_builder.build_fleet_replay_state(
            self.fleet_state, self.trip_state, self.station_state, self.current_step, self.clock_s)

    def _get_state(self) -> torch.Tensor:
        vehicle_features = self.feature_builder.build_vehicle_features(
            self.fleet_state, self.current_step, self.clock_s, trips=self.trip_state,
            stations=self.station_state, max_pickup_distance=self.max_pickup_distance)
        hex_features = self.feature_builder.build_hex_features(
            self.fleet_state, self.trip_state, self.station_state, self.current_step)
        context_features = self.feature_builder.build_context_features(self.clock_s, self.fleet_state, self.trip_state)
        return torch.cat([vehicle_features.view(-1), hex_features.view(-1), context_features.view(-1)])

    def _build_state_dict(self, flat_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split a flat ``_get_state`` tensor into vehicle / hex / context parts (baseline replay)."""
        v = self.num_vehicles * self._vehicle_feature_dim
        h = self.num_hexes * self._hex_feature_dim
        return {
            'vehicle': flat_state[:v].view(self.num_vehicles, self._vehicle_feature_dim),
            'hex': flat_state[v:v + h].view(self.num_hexes, self._hex_feature_dim),
            'context': flat_state[v + h:],
        }

    def get_metrics(self) -> Dict[str, float]:
        i = self.episode_info
        return {
            "episode_reward": self.episode_reward,
            "trips_served": i.trips_served, "trips_completed": i.trips_completed,
            "trips_dropped": i.trips_dropped, "trips_loaded": i.trips_loaded,
            "avg_soc": i.avg_soc, "revenue": i.revenue, "driving_cost": i.driving_cost,
            "energy_cost": i.energy_cost, "profit": i.profit,
            "energy_charged_kwh": i.energy_charged_kwh, "feeder_peak_kw": i.feeder_peak_kw,
            "feeder_violation_steps": i.feeder_violation_steps, "infeasible_executed": i.infeasible_executed,
        }


GPUEnvironment = GPUEnvironmentV2
