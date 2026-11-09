"""Vehicle and charging physics, the single source for simulator, projection and masks (C4).

Paper values (VIII-A) with the decisions of PD-RSAC_FIX_DECISIONS.md:

- battery E_max = 50 kWh, reserve E_min = 7.5 kWh (15 %), driving eta_drv = 0.2 kWh/km;
- charging E <- min(E_max, E + eta_c * p * dt) with eta_c = 0.95, p in [0, P_max^s = 50 kW]
  (the projection additionally asks p >= p_min = 20 kW when it charges);
- every action keeps E >= E_min afterwards, except the drive to a charging station, which only
  needs E >= 0 on arrival (Q4; a disclosed deviation from Eq. 16);
- travel time T(d) = max(1, ceil(d / v / dt)) steps at a fixed v = 25 km/h (Q14b, Eq. 9);
- energy units are kWh throughout; fleet tensors named ``socs`` hold kWh, not percent.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# Tolerance for energy comparisons (float32 kWh).
EPS = 1e-5


@dataclass(frozen=True)
class Physics:
    battery_kwh: float
    reserve_kwh: float
    energy_per_km: float
    charge_efficiency: float
    speed_kmh: float
    step_hours: float
    p_max_kw: float
    p_min_kw: float
    feeder_kw: float
    initial_energy_range: Tuple[float, float] = (0.5, 0.9)

    @classmethod
    def from_config(cls, config) -> "Physics":
        ph, st = config.physics, config.station
        lo, hi = config.vehicle.initial_energy_range
        if not (0.0 <= ph.reserve_kwh < ph.battery_kwh):
            raise ValueError("physics.reserve_kwh must lie in [0, battery_kwh)")
        if not (0.0 < st.min_power <= st.max_power):
            raise ValueError("station.min_power must lie in (0, max_power]")
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("vehicle.initial_energy_range must be 0 <= lo <= hi <= 1")
        return cls(
            battery_kwh=float(ph.battery_kwh), reserve_kwh=float(ph.reserve_kwh),
            energy_per_km=float(ph.energy_per_km), charge_efficiency=float(ph.charge_efficiency),
            speed_kmh=float(ph.avg_speed_kmh), step_hours=float(config.episode.step_duration_minutes) / 60.0,
            p_max_kw=float(st.max_power), p_min_kw=float(st.min_power), feeder_kw=float(st.feeder_kw),
            initial_energy_range=(float(lo), float(hi)),
        )

    # ------------------------------------------------------------------ driving
    def energy(self, km: torch.Tensor) -> torch.Tensor:
        """Energy used to drive ``km`` (kWh)."""
        return km * self.energy_per_km

    def travel_steps(self, km: torch.Tensor) -> torch.Tensor:
        """T(d) = max(1, ceil(d / v / dt)) decision epochs."""
        return torch.ceil(km / self.speed_kmh / self.step_hours - 1e-9).clamp(min=1).long()

    def can_drive(self, energy_kwh: torch.Tensor, km: torch.Tensor) -> torch.Tensor:
        """Energy floor for serve and reposition: E - eta_drv * d >= E_min."""
        return energy_kwh - self.energy(km) >= self.reserve_kwh - EPS

    def can_reach_station(self, energy_kwh: torch.Tensor, km: torch.Tensor) -> torch.Tensor:
        """The drive to a charging station only needs E >= 0 on arrival (Q4)."""
        return energy_kwh - self.energy(km) >= -EPS

    # ------------------------------------------------------------------ charging
    def max_power(self, energy_kwh: torch.Tensor) -> torch.Tensor:
        """Largest power that does not overfill the battery in one step, capped at P_max^s."""
        headroom = (self.battery_kwh - energy_kwh).clamp(min=0.0)
        return (headroom / (self.charge_efficiency * self.step_hours)).clamp(max=self.p_max_kw)

    def charge(self, energy_kwh: torch.Tensor, power_kw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One charging step. Returns (new energy, energy stored, energy drawn from the grid)."""
        stored = torch.minimum(self.charge_efficiency * power_kw * self.step_hours,
                               (self.battery_kwh - energy_kwh).clamp(min=0.0))
        return energy_kwh + stored, stored, stored / self.charge_efficiency

    # ------------------------------------------------------------------ initial state
    def initial_energy(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        lo, hi = self.initial_energy_range
        u = torch.rand(n, generator=generator)
        return (lo + (hi - lo) * u) * self.battery_kwh
