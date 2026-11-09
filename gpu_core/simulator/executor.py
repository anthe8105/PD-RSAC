"""Action executor shared by every algorithm (workstream C7).

Input per vehicle (only vehicles available at this decision epoch are read):

- ``mode``: -1 no-op, 0 SERVE, 1 CHARGE, 2 REPOSITION;
- ``repos_target``: target cell for REPOSITION (the vehicle's own cell = stay);
- ``station``: station index for CHARGE (-1 = nearest station);
- ``power_frac``: for CHARGE, the share of the free battery capacity at arrival to add in the
  session (default 1 = full); the session charges at P_max^s (the projection may lower the power
  of vehicles in a session each epoch, under the feeder) and the vehicle is busy until done;
- ``serve_trip``: trip slot for SERVE, only on the exact (projection) path.

A no-op keeps the current activity for one epoch: an IDLE vehicle stays, a CHARGING vehicle
keeps charging at its previous power. An action that cannot be executed is a no-op as well, so
a charger whose SERVE finds no order keeps its port; only an explicit stay unplugs it. CHARGE at the vehicle's current station (or with no
station) also continues charging; any other action releases the port first.

Two paths, identical physics:

- **exact** (MILP projection): execute exactly the given (mode, trip, target, station, power);
  anything infeasible is not executed (the vehicle stays) and counted in ``infeasible``; the
  ports of all chargers that leave are released before any booking, as the projection assumes,
  so a charger whose new action fails unplugs instead of continuing;
- **raw** (no projection; ``no_milp``, ``sac``, ``mappo``, ``maddpg``):
  1. orders are matched first, one by one, longest-waiting first (ties: earlier arrival), each to
     the nearest vehicle within the pickup's 1-ring that keeps E >= E_min (Eq. 4), whatever it
     intended: any vehicle that is at a location (IDLE, or CHARGING, also inside a session) can
     be matched; vehicles on a trip, repositioning or driving to a station cannot (their position
     is only updated on arrival). Within a cell the vehicles that intended SERVE go first, then
     the other deciding vehicles, then the chargers in a session (whose session and port end). A
     SERVE vehicle left without an order stays;
  2. CHARGE vehicles are handled in increasing energy order: the chosen station if it has a
     free port, else the nearest station with a free port among the vehicle's 5 nearest,
     else the vehicle stays; power is clipped to [0, min(P_max^s, battery headroom)];
  3. REPOSITION is executed if the energy floor allows it, else the vehicle stays.

No fallback changes an action into another one (the old env turned failed repositions into
SERVE and failed charges into repositions, N12).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from ..state import VehicleStatus
from .physics import EPS, Physics

NOOP, SERVE, CHARGE, REPOSITION = -1, 0, 1, 2
# Executed action codes (``Outcome.mode``)
EXEC_NONE, EXEC_SERVE, EXEC_CHARGE, EXEC_REPOS, EXEC_STAY, EXEC_CONTINUE = -1, 0, 1, 2, 3, 4
STATION_CANDIDATES = 5


@dataclass
class Outcome:
    """What was executed at one decision epoch. Per-vehicle tensors have length V."""
    mode: torch.Tensor            # EXEC_* code; EXEC_NONE for vehicles that did not decide
    duration: torch.Tensor        # epochs until the vehicle decides again (0 if it did not decide)
    target_cell: torch.Tensor     # drop-off / reposition target / station cell, -1 otherwise
    station: torch.Tensor         # booked or continued station, -1 otherwise
    trip: torch.Tensor            # trip slot accepted, -1 otherwise
    power_kw: torch.Tensor        # nominal session power (kW) of a booking
    fare: torch.Tensor            # fare credited at acceptance
    drive_km: torch.Tensor        # km whose driving cost is booked now
    charge_frac: torch.Tensor = None  # share of the free capacity a booking will add (0 otherwise)
    counts: Dict[str, int] = field(default_factory=dict)


class ActionExecutor:

    def __init__(self, physics: Physics, distances: torch.Tensor, ring1: torch.Tensor,
                 station_cells: torch.Tensor, ports_per_station: int, device, candidates=None):
        self.physics = physics
        self.ports = int(ports_per_station)
        self.device = torch.device(device)
        self.dist = distances.to(self.device)                       # [H, H] km
        self.ring1 = ring1.to(self.device, torch.bool)              # [H, H] grid distance <= 1
        self.station_cells = station_cells.to(self.device, torch.long)
        self.num_stations = len(self.station_cells)
        # CPU copies for the sequential matcher
        self._dist_np = self.dist.detach().cpu().numpy().astype(np.float64)
        ring_np = self.ring1.cpu().numpy()
        self._ring_sorted = [sorted(np.nonzero(ring_np[c])[0].tolist(), key=lambda n, c=c: (self._dist_np[n, c], n))
                             for c in range(ring_np.shape[0])]
        if candidates is None:
            from .candidates import CandidateGenerator
            candidates = CandidateGenerator(physics, distances, ring1, station_cells)
        # Station candidates come from the shared generator (same set as the projection, D6).
        self.nearest_stations = torch.from_numpy(candidates.near_stations).to(self.device)

    # ------------------------------------------------------------------ main
    def execute(self, fleet, trips, k: int, mode: torch.Tensor,
                repos_target: Optional[torch.Tensor] = None,
                station: Optional[torch.Tensor] = None,
                power_frac: Optional[torch.Tensor] = None,
                serve_trip: Optional[torch.Tensor] = None,
                exact: bool = False) -> Outcome:
        V, dev, ph = fleet.num_vehicles, self.device, self.physics
        mode = mode.to(dev).long().view(-1).clone()
        if mode.numel() != V:
            raise ValueError(f"mode has {mode.numel()} entries for {V} vehicles")
        avail = fleet.get_available_mask()
        bad_mode = avail & ((mode < NOOP) | (mode > REPOSITION))
        if bad_mode.any():
            raise ValueError(f"unknown action mode(s) {mode[bad_mode].unique().tolist()}")
        mode[~avail] = -2                                            # not deciding: ignored
        out = Outcome(
            mode=torch.full((V,), EXEC_NONE, dtype=torch.long, device=dev),
            duration=torch.zeros(V, dtype=torch.long, device=dev),
            target_cell=torch.full((V,), -1, dtype=torch.long, device=dev),
            station=torch.full((V,), -1, dtype=torch.long, device=dev),
            trip=torch.full((V,), -1, dtype=torch.long, device=dev),
            power_kw=torch.zeros(V, device=dev),
            fare=torch.zeros(V, device=dev),
            drive_km=torch.zeros(V, device=dev),
            charge_frac=torch.zeros(V, device=dev),
            counts={"decisions": int(avail.sum()), "infeasible": 0, "serve_unmatched": 0,
                    "charge_no_port": 0, "power_clipped": 0, "released": 0},
        )
        req_station = station.to(dev).long().view(-1) if station is not None else torch.full((V,), -1, dtype=torch.long, device=dev)
        frac = power_frac.to(dev).float().view(-1) if power_frac is not None else None

        # 0) raw path: orders go to any vehicle at a location with enough energy, before anything else
        if not exact:
            mv, mt = self._serve_raw_any(fleet, trips, avail, mode)
            if len(mv):
                self._start_trips(fleet, trips, k, mv, mt, out)
                mode[mv] = -2                                            # matched: no other action

        # 1) chargers: continue (no-op or CHARGE at the same / unspecified station) or release
        charging = avail & (fleet.status == VehicleStatus.CHARGING)
        own = fleet.charging_station
        cont = charging & ((mode == NOOP) | ((mode == CHARGE) & ((req_station < 0) | (req_station == own))))
        if cont.any():
            idx = cont.nonzero(as_tuple=True)[0]
            p = fleet.charge_power[idx].clone()
            if frac is not None:
                set_p = mode[idx] == CHARGE
                p[set_p] = frac[idx[set_p]].clamp(min=0.0) * ph.p_max_kw
            p = self._clip_power(p, fleet.socs[idx], exact, out)
            fleet.continue_charging(idx, k + 1, p)
            self._record(out, idx, EXEC_CONTINUE, 1, station=own[idx], target=self.station_cells[own[idx]], power=p)
        leaving = charging & ~cont

        stay = avail & (mode == NOOP) & ~cont        # explicit stay (a charger that stays unplugs)
        failed = torch.zeros_like(stay)               # action could not be executed
        # 2) reposition (own cell = stay)
        rep = avail & (mode == REPOSITION)
        if rep.any():
            if repos_target is None:
                raise ValueError("REPOSITION actions need repos_target")
            idx = rep.nonzero(as_tuple=True)[0]
            tgt = repos_target.to(dev).long().view(-1)[idx]
            pos = fleet.positions[idx]
            valid = (tgt >= 0) & (tgt < self.dist.shape[0])
            same = valid & (tgt == pos)
            tgt_c = tgt.clamp(0, self.dist.shape[0] - 1)
            km = self.dist[pos, tgt_c]
            ok = valid & ~same & ph.can_drive(fleet.socs[idx], km)
            stay[idx[same]] = True
            bad = idx[~ok & ~same]
            failed[bad] = True
            out.counts["infeasible"] += len(bad)
            if ok.any():
                i, t, d = idx[ok], tgt_c[ok], km[ok]
                T = ph.travel_steps(d)
                fleet.socs[i] -= ph.energy(d)
                fleet.start_repositioning(i, t, k + T)
                self._record(out, i, EXEC_REPOS, T, target=t, km=d)

        # 3) serve (raw path: the matching was done in step 0; a SERVE vehicle still here got no order)
        srv = avail & (mode == SERVE)
        if srv.any():
            if exact:
                matched_v, matched_t = self._serve_exact(fleet, trips, srv, serve_trip, out)
            else:
                matched_v, matched_t = srv.nonzero(as_tuple=True)[0][:0], srv.nonzero(as_tuple=True)[0][:0]
                out.counts["serve_unmatched"] += int(srv.sum())
            unmatched = srv.clone()
            if len(matched_v):
                unmatched[matched_v] = False
                self._start_trips(fleet, trips, k, matched_v, matched_t, out)
            failed |= unmatched

        # 4) new charging bookings. Raw path: ports of chargers that already left (moved, serve or
        # stay) are free; a charger trying another station keeps its own port until it gets one.
        # Exact path: the projection counts the port of every leaving charger as free (Eq. 14), so
        # all of them are released first, whatever the booking order (a charger switching
        # station must not block the vehicle booked at its old port).
        released = leaving if exact else leaving & ((out.mode >= 0) | stay)
        out.counts["released"] = int(released.sum())
        keep = fleet.holds_port_mask() & ~released
        free = self.ports - torch.bincount(fleet.charging_station[keep], minlength=self.num_stations)
        chg = avail & (mode == CHARGE) & ~cont
        if chg.any():
            if self.num_stations == 0:
                failed |= chg
                out.counts["infeasible"] += int(chg.sum())
            else:
                self._charge(fleet, k, chg, req_station, frac, free, exact, out, failed)

        # 5) a failed action is a no-op: an idle vehicle stays, a charger keeps charging at its
        # port and previous power; explicit stays stand still (and unplug)
        failed &= out.mode == EXEC_NONE
        keep_charging = failed & charging & ~released     # a released port may have been re-booked
        if keep_charging.any():
            idx = keep_charging.nonzero(as_tuple=True)[0]
            p = self._clip_power(fleet.charge_power[idx].clone(), fleet.socs[idx], False, out)
            fleet.continue_charging(idx, k + 1, p)
            self._record(out, idx, EXEC_CONTINUE, 1, station=own[idx], target=self.station_cells[own[idx]], power=p)
        stay = (stay | failed) & (out.mode == EXEC_NONE)
        if stay.any():
            idx = stay.nonzero(as_tuple=True)[0]
            fleet.stay(idx)
            self._record(out, idx, EXEC_STAY, 1)
        return out

    # ------------------------------------------------------------------ helpers
    def _record(self, out, idx, code, duration, target=None, station=None, km=None, power=None):
        out.mode[idx] = code
        out.duration[idx] = duration if torch.is_tensor(duration) else torch.full_like(idx, int(duration))
        if target is not None:
            out.target_cell[idx] = target
        if station is not None:
            out.station[idx] = station
        if km is not None:
            out.drive_km[idx] = km
        if power is not None:
            out.power_kw[idx] = power

    def _start_trips(self, fleet, trips, k, matched_v, matched_t, out):
        ph = self.physics
        pos = fleet.positions[matched_v]
        pick, drop = trips.pickup_hex[matched_t], trips.dropoff_hex[matched_t]
        d_pick, d_trip = self.dist[pos, pick], trips.distance_km[matched_t]
        T = ph.travel_steps(d_pick) + ph.travel_steps(d_trip)        # Eq. (9)
        fleet.socs[matched_v] -= ph.energy(d_pick + d_trip)
        trips.assigned_vehicle[matched_t] = matched_v
        trips.assigned[matched_t] = True
        trips.n_assigned_total += len(matched_t)
        fleet.start_serving(matched_v, matched_t, drop, k + T)
        self._record(out, matched_v, EXEC_SERVE, T, target=drop, km=d_pick + d_trip)
        out.trip[matched_v] = matched_t
        out.fare[matched_v] = trips.fare[matched_t]

    def _clip_power(self, p, energy, exact, out):
        hi = self.physics.max_power(energy)
        clipped = (p > hi + 1e-4) | (p < 0)
        if exact:
            out.counts["power_clipped"] += int(clipped.sum())
        return torch.minimum(p.clamp(min=0.0), hi)

    def _serve_exact(self, fleet, trips, srv, serve_trip, out):
        """Validate the projection's (vehicle, trip) pairs; infeasible pairs are not executed."""
        if serve_trip is None:
            raise ValueError("the exact path needs serve_trip for SERVE actions")
        idx = srv.nonzero(as_tuple=True)[0]
        slot = serve_trip.to(self.device).long().view(-1)[idx]
        ok = (slot >= 0) & (slot < trips.max_trips)
        slot_c = slot.clamp(0, trips.max_trips - 1)
        ok &= trips.valid_mask[slot_c] & (trips.assigned_vehicle[slot_c] < 0)
        pos = fleet.positions[idx]
        pick = trips.pickup_hex[slot_c]
        ok &= self.ring1[pos, pick]
        ok &= self.physics.can_drive(fleet.socs[idx], self.dist[pos, pick] + trips.distance_km[slot_c])
        # each order at most once: keep the first vehicle per slot
        if ok.any():
            first = torch.zeros_like(ok)
            seen = set()
            for j in ok.nonzero(as_tuple=True)[0].tolist():
                s = int(slot_c[j])
                if s not in seen:
                    seen.add(s)
                    first[j] = True
            ok = first
        out.counts["infeasible"] += int((~ok).sum())
        return idx[ok], slot_c[ok]

    def _serve_raw_any(self, fleet, trips, avail, mode):
        """Order-centric greedy matcher over every vehicle at a location: deciding vehicles (SERVE
        intentions first, then the others) and chargers in a session last, nearest cell first."""
        status = fleet.status
        pool = (avail & (status == VehicleStatus.IDLE)) | (status == VehicleStatus.CHARGING)
        rank = torch.full_like(mode, 2)
        rank[avail & (mode == SERVE)] = 0
        rank[avail & (mode != SERVE)] = 1
        veh = pool.nonzero(as_tuple=True)[0]
        if len(veh):
            r = rank[veh].cpu().numpy()
            veh = veh[torch.as_tensor(np.lexsort((veh.cpu().numpy(), r)), device=veh.device)]
        return self._serve_raw(fleet, trips, None, veh=veh)

    def _serve_raw(self, fleet, trips, srv, veh=None):
        """Order-centric greedy matcher (Q19); ``veh`` (in priority order) replaces ``srv``."""
        open_slots = trips.get_unassigned_mask().nonzero(as_tuple=True)[0]
        veh = srv.nonzero(as_tuple=True)[0] if veh is None else veh
        if len(open_slots) == 0 or len(veh) == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty
        wait = trips.wait_steps[open_slots].cpu().numpy()
        tid = trips.trip_ids[open_slots].cpu().numpy()
        order = np.lexsort((tid, -wait))                         # longest wait first, then arrival
        slots = open_slots.cpu().numpy()[order]
        pick = trips.pickup_hex[open_slots].cpu().numpy()[order]
        trip_km = trips.distance_km[open_slots].cpu().numpy().astype(np.float64)[order]

        v_np = veh.cpu().numpy()
        v_pos = fleet.positions[veh].cpu().numpy()
        v_e = fleet.socs[veh].cpu().numpy().astype(np.float64)
        by_cell: Dict[int, list] = {}
        for j, c in enumerate(v_pos.tolist()):
            by_cell.setdefault(c, []).append(j)
        ph = self.physics
        floor = ph.reserve_kwh - EPS
        mv, mt = [], []
        for o in range(len(slots)):
            if not by_cell:
                break
            p = int(pick[o])
            for c in self._ring_sorted[p]:
                lst = by_cell.get(c)
                if not lst:
                    continue
                need = ph.energy_per_km * (self._dist_np[c, p] + trip_km[o])
                hit = next((n for n, j in enumerate(lst) if v_e[j] - need >= floor), None)
                if hit is not None:
                    j = lst.pop(hit)
                    if not lst:
                        del by_cell[c]
                    mv.append(int(v_np[j]))
                    mt.append(int(slots[o]))
                    break
        return (torch.tensor(mv, dtype=torch.long, device=self.device),
                torch.tensor(mt, dtype=torch.long, device=self.device))

    def _charge(self, fleet, k, chg, req_station, frac, free, exact, out, failed):
        ph, dev = self.physics, self.device
        idx = chg.nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(fleet.socs[idx], stable=True)]       # lowest energy first
        pos = fleet.positions[idx]
        want = req_station[idx]
        nearest = self.nearest_stations[pos, 0]
        want = torch.where(want >= 0, want, nearest)
        free_l = free.tolist()
        e_l = fleet.socs[idx].tolist()
        pos_l, want_l = pos.tolist(), want.tolist()
        cand_l = self.nearest_stations[pos].tolist()
        dist = self._dist_np
        cells = self.station_cells.tolist()
        chosen = []
        for n in range(len(idx)):
            s = want_l[n]
            options = [s] if exact else [s] + [c for c in cand_l[n] if c != s]
            pick = None
            had_port = False
            for c in options:
                if not 0 <= c < self.num_stations or free_l[c] <= 0:
                    continue
                had_port = True
                if e_l[n] - ph.energy_per_km * dist[pos_l[n], cells[c]] >= -EPS:
                    pick = c
                    break
            if pick is None:
                chosen.append(-1)
                out.counts["infeasible" if (exact or had_port) else "charge_no_port"] += 1
            else:
                free_l[pick] -= 1
                chosen.append(pick)
        chosen = torch.tensor(chosen, dtype=torch.long, device=dev)
        ok = chosen >= 0
        failed[idx[~ok]] = True
        if not ok.any():
            return
        i, s = idx[ok], chosen[ok]
        cell = self.station_cells[s]
        km = self.dist[fleet.positions[i], cell]
        T = ph.travel_steps(km)
        fleet.socs[i] -= ph.energy(km)
        f = frac[i].clamp(0.0, 1.0) if frac is not None else torch.ones(len(i), device=dev)
        e_arr = fleet.socs[i]
        target = e_arr + f * (ph.battery_kwh - e_arr).clamp(min=0.0)
        p = torch.full((len(i),), ph.p_max_kw, device=dev)
        fleet.start_charge_travel(i, s, cell, k + T, p, target)
        # duration 0 = until the session ends (T + charging epochs, not known in advance under the feeder)
        self._record(out, i, EXEC_CHARGE, 0, target=cell, station=s, km=km, power=p)
        out.charge_frac[i] = f
