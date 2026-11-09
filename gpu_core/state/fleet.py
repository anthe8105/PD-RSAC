"""Tensor fleet state and the vehicle state machine (workstream C2).

| status          | holds a port | decides         | leaves when                                         |
|-----------------|--------------|-----------------|-----------------------------------------------------|
| IDLE            | no           | every epoch     | any action                                          |
| SERVING         | no           | no              | T(pickup) + T(trip) elapsed -> IDLE at the drop-off |
| REPOSITIONING   | no           | no              | T elapsed -> IDLE at the target                     |
| CHARGE_TRAVEL   | yes (booked) | no              | arrives -> CHARGING (charges from the arrival epoch) |
| CHARGING        | yes          | no              | session: charges until ``charge_target`` -> IDLE     |

``busy_until`` is the decision epoch at which the current commitment ends; -1 means not
committed. Every commitment lasts at least one epoch. Energy (``socs``) is in kWh.
"""

import torch
from enum import IntEnum
from typing import Optional


class VehicleStatus(IntEnum):
    IDLE = 0
    SERVING = 1
    REPOSITIONING = 2
    CHARGING = 3
    CHARGE_TRAVEL = 4


NOT_BUSY = -1


class TensorFleetState:
    """Per-vehicle tensors. ``socs`` is the battery energy in kWh."""

    def __init__(
        self,
        num_vehicles: int,
        device: str = "cuda",
        max_soc: float = 50.0,
        initial_energy: Optional[torch.Tensor] = None,
    ):
        self.num_vehicles = num_vehicles
        self.device = torch.device(device)
        self.max_soc = max_soc  # battery capacity E_max (kWh)

        self.positions = torch.zeros(num_vehicles, dtype=torch.long, device=self.device)
        if initial_energy is None:
            initial_energy = torch.full((num_vehicles,), max_soc)
        self.socs = initial_energy.to(self.device, torch.float32).clone()
        self.status = torch.zeros(num_vehicles, dtype=torch.int8, device=self.device)
        self.busy_until = torch.full((num_vehicles,), NOT_BUSY, dtype=torch.int64, device=self.device)
        self.target_hex = torch.full((num_vehicles,), -1, dtype=torch.long, device=self.device)
        self.current_trip = torch.full((num_vehicles,), -1, dtype=torch.long, device=self.device)
        self.charging_station = torch.full((num_vehicles,), -1, dtype=torch.long, device=self.device)
        # Power (kW) for the vehicle's current or next charging step.
        self.charge_power = torch.zeros(num_vehicles, dtype=torch.float32, device=self.device)
        self.charge_target = torch.zeros(num_vehicles, dtype=torch.float32, device=self.device)   # kWh at session end
        # Consecutive epochs spent IDLE (diagnostics only).
        self.idle_steps = torch.zeros(num_vehicles, dtype=torch.int64, device=self.device)

    # ------------------------------------------------------------------ queries
    def get_available_mask(self, current_step: int = 0) -> torch.Tensor:
        """Vehicles that take a decision now: IDLE, or CHARGING after its committed step."""
        del current_step  # completions are applied before the decision epoch
        free = self.busy_until < 0
        return free & ((self.status == VehicleStatus.IDLE) | (self.status == VehicleStatus.CHARGING))

    def get_busy_mask(self, current_step: int = 0) -> torch.Tensor:
        return self.busy_until >= 0

    def holds_port_mask(self) -> torch.Tensor:
        return (self.status == VehicleStatus.CHARGING) | (self.status == VehicleStatus.CHARGE_TRAVEL)

    def get_status_counts(self) -> dict:
        return {s.name: int((self.status == s.value).sum().item()) for s in VehicleStatus}

    def get_charging_mask(self) -> torch.Tensor:
        return self.status == VehicleStatus.CHARGING

    def get_serving_mask(self) -> torch.Tensor:
        return self.status == VehicleStatus.SERVING

    def get_low_soc_mask(self, threshold_kwh: float) -> torch.Tensor:
        return self.socs < threshold_kwh

    # ------------------------------------------------------------------ transitions
    def start_serving(self, idx, trip_slots, dropoff_cells, busy_until) -> None:
        self.status[idx] = VehicleStatus.SERVING
        self.current_trip[idx] = trip_slots
        self.target_hex[idx] = dropoff_cells
        self.busy_until[idx] = busy_until
        self._clear_charging(idx)

    def start_repositioning(self, idx, target_cells, busy_until) -> None:
        self.status[idx] = VehicleStatus.REPOSITIONING
        self.target_hex[idx] = target_cells
        self.busy_until[idx] = busy_until
        self.current_trip[idx] = -1
        self._clear_charging(idx)

    def start_charge_travel(self, idx, stations, station_cells, busy_until, power_kw, target_kwh=None) -> None:
        self.status[idx] = VehicleStatus.CHARGE_TRAVEL
        self.charging_station[idx] = stations
        self.target_hex[idx] = station_cells
        self.busy_until[idx] = busy_until
        self.charge_power[idx] = power_kw
        self.charge_target[idx] = self.max_soc if target_kwh is None else target_kwh
        self.current_trip[idx] = -1

    def end_charging(self, idx) -> None:
        """Session done: IDLE at the station, port released, decides at the next epoch."""
        self.status[idx] = VehicleStatus.IDLE
        self.busy_until[idx] = NOT_BUSY
        self.target_hex[idx] = -1
        self._clear_charging(idx)

    def continue_charging(self, idx, busy_until, power_kw) -> None:
        self.busy_until[idx] = busy_until
        self.charge_power[idx] = power_kw

    def stay(self, idx) -> None:
        """Stand still for one epoch: IDLE where the vehicle is (releases a held port)."""
        self.status[idx] = VehicleStatus.IDLE
        self.busy_until[idx] = NOT_BUSY
        self.target_hex[idx] = -1
        self.current_trip[idx] = -1
        self._clear_charging(idx)

    def _clear_charging(self, idx) -> None:
        self.charging_station[idx] = -1
        self.charge_power[idx] = 0.0
        self.charge_target[idx] = 0.0

    def complete(self, current_step: int):
        """Apply commitments ending at ``current_step``. Returns masks
        (serve_done, repos_done, arrived_at_station, charge_step_done)."""
        done = (self.busy_until >= 0) & (self.busy_until <= current_step)
        serve_done = done & (self.status == VehicleStatus.SERVING)
        repos_done = done & (self.status == VehicleStatus.REPOSITIONING)
        arrived = done & (self.status == VehicleStatus.CHARGE_TRAVEL)
        charge_done = done & (self.status == VehicleStatus.CHARGING)

        moved = serve_done | repos_done | arrived
        if moved.any():
            self.positions[moved] = self.target_hex[moved]
        ended = serve_done | repos_done
        if ended.any():
            self.status[ended] = VehicleStatus.IDLE
            self.busy_until[ended] = NOT_BUSY
            self.target_hex[ended] = -1
            self.current_trip[ended] = -1
        if arrived.any():
            # The first charging step is part of the CHARGE decision (Delta = T + 1, Q4).
            self.status[arrived] = VehicleStatus.CHARGING
            self.target_hex[arrived] = -1
            self.busy_until[arrived] = current_step + 1
        if charge_done.any():
            self.busy_until[charge_done] = NOT_BUSY  # still CHARGING, holds its port, decides now
        return serve_done, repos_done, arrived, charge_done

    # ------------------------------------------------------------------ misc
    def clone(self) -> "TensorFleetState":
        new = TensorFleetState.__new__(TensorFleetState)
        new.num_vehicles, new.device, new.max_soc = self.num_vehicles, self.device, self.max_soc
        for name in ("positions", "socs", "status", "busy_until", "target_hex", "current_trip",
                     "charging_station", "charge_power", "charge_target", "idle_steps"):
            setattr(new, name, getattr(self, name).clone())
        return new

    @property
    def mean_soc(self) -> float:
        return self.socs.mean().item()

    @property
    def num_available(self) -> int:
        return int(self.get_available_mask().sum().item())
