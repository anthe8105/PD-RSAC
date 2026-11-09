"""Charging station state (workstream C2/C3).

Ports are a physical constraint: a port is held from the moment a CHARGE decision books it
(CHARGE_TRAVEL) until the vehicle leaves the charger. Occupancy is always derived from the
fleet state (``sync``), so it cannot drift from the vehicles that actually hold ports (F28).
There is no queue: a vehicle can only book a station that has a free port.
"""

import torch


class TensorStationState:

    def __init__(self, num_stations: int, device: str = "cuda", num_ports: int = 5, max_power: float = 50.0):
        self.num_stations = num_stations
        self.device = torch.device(device)
        self.num_ports = num_ports
        self.max_power = max_power
        self.hex_ids = torch.zeros(num_stations, dtype=torch.long, device=self.device)
        self.ports = torch.full((num_stations,), num_ports, dtype=torch.int32, device=self.device)
        self.occupied = torch.zeros(num_stations, dtype=torch.int32, device=self.device)

    def set_locations(self, hex_ids: torch.Tensor) -> None:
        if len(hex_ids) != self.num_stations:
            raise ValueError(f"{len(hex_ids)} station cells for {self.num_stations} stations")
        self.hex_ids = hex_ids.to(self.device, torch.long).clone()

    def sync(self, fleet) -> None:
        """Occupied ports = vehicles holding a port (CHARGE_TRAVEL or CHARGING) per station."""
        holders = fleet.holds_port_mask()
        st = fleet.charging_station[holders]
        if (st < 0).any():
            raise RuntimeError("vehicle holds a port without a station")
        self.occupied = torch.bincount(st, minlength=self.num_stations).to(torch.int32)
        if (self.occupied > self.ports).any():
            raise RuntimeError("more vehicles hold ports than the station has")

    def get_available_ports(self) -> torch.Tensor:
        return torch.clamp(self.ports - self.occupied, min=0)

    def get_available_mask(self) -> torch.Tensor:
        return self.get_available_ports() > 0

    def get_utilization(self) -> torch.Tensor:
        return self.occupied.float() / self.ports.float().clamp(min=1)

    def get_station_at_hex(self, hex_idx: int) -> int:
        matches = torch.where(self.hex_ids == hex_idx)[0]
        return matches[0].item() if len(matches) > 0 else -1

    def reset(self) -> None:
        self.occupied.fill_(0)

    def to_feature_tensor(self) -> torch.Tensor:
        """[S, 3]: utilisation, free-port ratio, max power / 100."""
        return torch.stack([
            self.get_utilization(),
            self.get_available_ports().float() / max(self.num_ports, 1),
            torch.full((self.num_stations,), self.max_power / 100.0, device=self.device),
        ], dim=1)
