"""Greedy heuristic baseline of Tian et al. (Neurocomputing 622, 2025, 129343, Sec. 5.2), after
Chen, Kockelman and Hanna (Transp. Res. A 94, 2016):

"we will assign the nearest EV with sufficient SoC to serve travel demands. If vehicles are in
charging stations, assign vehicles with low SoCs to charge or discharge. Then, the empty vehicles
will be relocated to the grid with the largest supply-demand gap."

Per decision epoch, over the deciding vehicles, in that order:

1. order matching: open orders by longest wait first (then arrival); each goes to the nearest free
   vehicle within the pickup radius (the shared 1-ring, Q3a) whose energy covers the trip with
   E - eta (d_pickup + d_trip) >= E_min (Eq. 4; E_min is far more than any drive to a station, so
   the vehicle can still reach a charger after the drop-off, as the reference requires);
2. charging: a vehicle left without an order and below ``low`` of its battery books the nearest of
   its 5 candidate stations with a free port that it can reach (lowest energy first) and charges
   to ``charge_target`` (the reference charges to the maximum SoC). Discharging (V2G) is not part
   of this model;
3. relocation: every remaining empty vehicle moves to the cell of its reposition candidates (the
   learners' 91 slots) with the largest supply-demand gap, open orders minus empty vehicles in
   the cell, if that gap is larger than in its own cell and the move keeps E >= E_min; vehicles
   are processed one at a time and each counts as supply where it goes;
4. a vehicle that decides while holding a charger keeps charging below ``charge_target``.

The chosen (vehicle, trip, station, power) go to the simulator's exact path, so the pairs the
heuristic matched are the pairs executed (no re-matching); every action is feasible by
construction (``infeasible_executed`` stays 0). Charging sessions run at P_max^s, limited by the
battery headroom.
"""

from typing import Dict

import numpy as np
import torch

from ..simulator.executor import CHARGE, NOOP, REPOSITION, SERVE
from ..state import VehicleStatus


class GreedyPolicy:

    def __init__(self, env, low: float = None, charge_target: float = 1.0):
        self.env = env
        self.low = float(env.config.vehicle.low_energy_fraction if low is None else low)
        self.charge_target = float(charge_target)
        self.dist = env._distances.cpu().numpy().astype(np.float64)
        ring1 = env._ring1.cpu().numpy()
        # cells of each pickup's 1-ring, nearest first (ties: cell index)
        self.ring_sorted = [sorted(np.nonzero(ring1[c])[0].tolist(), key=lambda n, c=c: (self.dist[n, c], n))
                            for c in range(ring1.shape[0])]
        self.near = np.asarray(env.candidates.near_stations)
        self.station_cells = env._station_cells.cpu().numpy()
        self.slots = np.asarray(env.candidates.slot_cells)
        self.ports = int(env.config.station.num_ports)

    def act(self, env=None) -> Dict[str, torch.Tensor]:
        env = env or self.env
        ph, fl, ts = env.physics, env.fleet_state, env.trip_state
        V, H = env.num_vehicles, len(self.ring_sorted)
        dec = fl.get_available_mask().cpu().numpy()
        E = fl.socs.cpu().numpy().astype(np.float64)
        pos = fl.positions.cpu().numpy()
        status = fl.status.cpu().numpy()
        own = fl.charging_station.cpu().numpy()
        cap = ph.battery_kwh
        mode = np.full(V, NOOP, np.int64)
        target = pos.copy()
        station = np.full(V, -1, np.int64)
        power = np.ones(V, np.float32)
        trip = np.full(V, -1, np.int64)
        charging = dec & (status == VehicleStatus.CHARGING)
        free_veh = dec & ~charging

        def p_frac(energy):
            return float(ph.max_power(torch.tensor(energy, dtype=torch.float32))) / ph.p_max_kw

        # 1) order matching: each order to the nearest vehicle with sufficient energy
        open_slots = ts.get_unassigned_mask().nonzero(as_tuple=True)[0].cpu().numpy()
        taken = np.zeros(len(open_slots), bool)
        if len(open_slots):
            wait = ts.wait_steps.cpu().numpy()[open_slots]
            tid = ts.trip_ids.cpu().numpy()[open_slots]
            pick = ts.pickup_hex.cpu().numpy()[open_slots]
            km_trip = ts.distance_km.cpu().numpy().astype(np.float64)[open_slots]
            by_cell: Dict[int, list] = {}
            for v in np.nonzero(free_veh)[0]:
                by_cell.setdefault(int(pos[v]), []).append(int(v))
            for o in np.lexsort((tid, -wait)):
                p = int(pick[o])
                for c in self.ring_sorted[p]:
                    lst = by_cell.get(c)
                    if not lst:
                        continue
                    need = ph.energy_per_km * (self.dist[c, p] + km_trip[o])
                    hit = next((n for n, v in enumerate(lst) if E[v] - need >= ph.reserve_kwh - 1e-6), None)
                    if hit is not None:
                        v = lst.pop(hit)
                        mode[v], trip[v], taken[o] = SERVE, int(open_slots[o]), True
                        free_veh[v] = False
                        break

        # 2) low-energy vehicles without an order charge (nearest station with a free port)
        holders = fl.holds_port_mask().cpu().numpy()
        free = self.ports - np.bincount(own[holders], minlength=len(self.station_cells))
        low = np.nonzero(free_veh & (E < self.low * cap))[0]
        for v in low[np.argsort(E[low], kind="stable")]:
            for s in self.near[pos[v]]:
                if s < 0 or free[s] <= 0:
                    continue
                e_arr = E[v] - ph.energy_per_km * self.dist[pos[v], self.station_cells[s]]
                if e_arr < -1e-6 or p_frac(max(e_arr, 0.0)) * ph.p_max_kw < ph.p_min_kw:
                    continue
                free[s] -= 1
                e0 = max(e_arr, 0.0)
                share = (self.charge_target * cap - e0) / max(cap - e0, 1e-9)
                mode[v], station[v], power[v] = CHARGE, s, float(np.clip(share, 0.0, 1.0))
                free_veh[v] = False
                break

        # 3) relocation of the empty vehicles toward the largest supply-demand gap
        pick_all = ts.pickup_hex.cpu().numpy()
        demand = np.bincount(pick_all[open_slots[~taken]], minlength=H).astype(np.float64)
        supply = np.bincount(pos[free_veh], minlength=H).astype(np.float64)
        for v in np.nonzero(free_veh)[0]:
            h = int(pos[v])
            cells = self.slots[h]
            cells = cells[(cells >= 0) & (cells != h)]
            cells = cells[E[v] - ph.energy_per_km * self.dist[h, cells] >= ph.reserve_kwh - 1e-6]
            if len(cells) == 0:
                continue
            cells = cells[np.argsort(self.dist[h, cells], kind="stable")]          # ties: the nearest cell
            gap = demand[cells] - supply[cells]
            best = int(np.argmax(gap))
            if gap[best] <= demand[h] - supply[h]:
                continue
            g = int(cells[best])
            mode[v], target[v] = REPOSITION, g
            supply[h] -= 1.0
            supply[g] += 1.0

        # 4) a vehicle deciding at its charger keeps charging below the target, else unplugs
        for v in np.nonzero(charging)[0]:
            if E[v] < self.charge_target * cap and p_frac(E[v]) > 0:
                mode[v], station[v], power[v] = CHARGE, own[v], p_frac(E[v])
            else:
                mode[v], target[v] = REPOSITION, pos[v]          # explicit stay: unplug
        return {"mode": torch.as_tensor(mode), "target": torch.as_tensor(target), "station": torch.as_tensor(station),
                "power": torch.as_tensor(power), "trip": torch.as_tensor(trip)}

    def step(self, env=None):
        """Act and execute on the exact path; returns (intent, step outputs)."""
        env = env or self.env
        a = self.act(env)
        res = env.step(a["mode"], a["target"], vehicle_charge_power=a["power"], milp_serve_trip_ids=a["trip"],
                       charge_station=a["station"])
        return a["mode"], res
