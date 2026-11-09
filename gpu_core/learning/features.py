"""Observations of the vehicle-level agent (workstream E2, E4; audit F31, F44, D35, D36).

Two tiers, so that collection and learning build their inputs with **the same code**:

- a per-epoch ``Snapshot`` of the pre-decision state: per-cell features, this epoch's arrivals
  (the scenario xi-hat, by origin and destination), free ports per station, global features
  and, once the epoch has been executed, how many deciders in each cell executed each action;
- a per-vehicle ``Context`` for every vehicle that decides: cell, energy, charging state, held
  station, and the order information that only the full order list can give (whether an
  energy-feasible order exists in the 1-ring, how many, and the nearest one).

``build_inputs`` turns (snapshots, contexts) into every tensor the networks need: vehicle
observation, 1-ring summary, candidate features and masks for the 91 reposition slots and the
6 charging options (5 nearest stations + "continue charging"), and the global features. Masks
follow Eq. (4): SERVE is only available when an energy-feasible order is in the 1-ring.

Counts are divided by the mean number of vehicles per cell, so a 100-vehicle run on a 1 %
sample sees the same feature scale as 1000 vehicles on the full sample (same ratio).
"""

from dataclasses import dataclass, fields
from typing import Dict, Optional

import numpy as np
import torch

from ..simulator.physics import EPS
from ..state import VehicleStatus

# per-cell features (F_HEX); channels ARR_O / ARR_D are linear in the arrivals (WDRO scenario)
F_HEX = 12
ARR_O, ARR_D = 5, 6
F_GLOB = 9
F_VEH = 9
N_STATION_OPTIONS = 6          # 5 nearest stations + continue
CONTINUE = 5
F_STATION = 4
N_CODES = 5                    # executed codes: serve, repos, stay, charge (new port), continue
SERVE_C, REPOS_C, STAY_C, CHARGE_C, CONT_C = range(N_CODES)


@dataclass
class Snapshot:
    hex: torch.Tensor          # [U, H, F_HEX]
    xi: torch.Tensor           # [U, 2H] arrivals of the epoch (raw counts), origin then destination
    free_ports: torch.Tensor   # [U, S]
    glob: torch.Tensor         # [U, F_GLOB]
    exec_counts: torch.Tensor  # [U, H, N_CODES] executed codes of the deciders (after the step)
    feeder_share: torch.Tensor  # [U] feeder power per deciding charger (kW)
    serve_intent: torch.Tensor  # [U, H] deciders that intended SERVE (after the step)


@dataclass
class Context:
    snap: torch.Tensor         # [B] row of the snapshot batch
    cell: torch.Tensor         # [B]
    energy: torch.Tensor       # [B] kWh
    charging: torch.Tensor     # [B] bool: decides while holding a charger
    own_station: torch.Tensor  # [B] station held (-1)
    serve_ok: torch.Tensor     # [B] bool: an energy-feasible order in the 1-ring (Eq. 4)
    n_orders: torch.Tensor     # [B] number of such orders
    order: torch.Tensor        # [B, 3] nearest such order: pickup km / 10, trip km / 10, fare / 30

    def to(self, device) -> "Context":
        return Context(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def index(self, idx) -> "Context":
        return Context(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})


class Geometry:
    """Static tensors from the environment: distances, 1-ring / 2-hop index lists, slots, stations."""

    def __init__(self, env, device):
        cand, ph = env.candidates, env.physics
        self.device = torch.device(device)
        self.H = env.num_hexes
        self.V = env.num_vehicles
        self.S = env.num_stations
        self.ports = float(env.config.station.num_ports)
        self.count_scale = max(self.V / self.H, 1e-6)
        self.dist = torch.as_tensor(cand.dist, dtype=torch.float32, device=self.device)
        ring = torch.as_tensor(cand.ring1, device=self.device)
        self.ring_idx, self.ring_mask = _padded(ring)                              # [H, 7]
        two = (ring.float() @ ring.float()) > 0
        self.n2_idx, self.n2_mask = _padded(two)                                    # [H, <=19]
        self.slot_cells = torch.as_tensor(cand.slot_cells, device=self.device)      # [H, K]
        self.K = self.slot_cells.shape[1]
        self.slot_of = torch.as_tensor(cand.slot_of, device=self.device)            # [H, H]
        near = torch.as_tensor(cand.near_stations, device=self.device)               # [H, <=5]
        pad = N_STATION_OPTIONS - 1 - near.shape[1]
        self.near_st = torch.cat([near, torch.full((self.H, pad), -1, dtype=torch.long, device=self.device)], 1)             if pad > 0 else near                                                     # [H, 5], -1 = none
        self.st_cells = torch.as_tensor(cand.station_cells, device=self.device)     # [S]
        self.has_station = torch.zeros(self.H, device=self.device)
        self.has_station[self.st_cells] = 1.0
        self.ph = ph
        A = env.adjacency_matrix.to(self.device).float()                            # normalised A-hat
        self.A_hat = A
        # A-hat restricted to each cell's 1-ring (layer 2) and to each 1-ring member's 1-ring
        # expressed as positions in the cell's 2-hop list (layer 1): exact local GCN evaluation.
        H, R, N2 = self.H, self.ring_idx.shape[1], self.n2_idx.shape[1]
        pos_in_n2 = torch.full((H, H), -1, dtype=torch.long, device=self.device)
        rows = torch.arange(H, device=self.device)[:, None].expand(H, N2)
        pos_in_n2[rows[self.n2_mask], self.n2_idx[self.n2_mask]] = torch.arange(N2, device=self.device).expand(H, N2)[self.n2_mask]
        self.l2_w = A[rows[:, :R], self.ring_idx] * self.ring_mask                  # [H, R]
        u = self.ring_idx                                                            # [H, R] 1-ring members
        v = self.ring_idx[u]                                                         # [H, R, R] their 1-rings
        vm = self.ring_mask[u] & self.ring_mask[:, :, None]
        self.l1_pos = torch.where(vm, pos_in_n2[rows[:, :R, None].expand(H, R, R), v], torch.zeros_like(v))
        self.l1_w = A[u[:, :, None].expand(H, R, R), v] * vm
        if (self.l1_pos[vm] < 0).any():
            raise RuntimeError("2-hop lists do not contain every 1-ring neighbour's ring")
        self.ring_pos_n2 = torch.where(self.ring_mask, pos_in_n2[rows[:, :R], self.ring_idx], torch.zeros_like(self.ring_idx))


def _padded(mask: torch.Tensor):
    n = int(mask.sum(1).max())
    idx = torch.zeros(mask.shape[0], n, dtype=torch.long, device=mask.device)
    ok = torch.zeros(mask.shape[0], n, dtype=torch.bool, device=mask.device)
    for h in range(mask.shape[0]):
        nz = mask[h].nonzero().flatten()
        # the cell itself first (1-ring) so position 0 is always the centre
        nz = torch.cat([nz[nz == h], nz[nz != h]])
        idx[h, :len(nz)] = nz
        ok[h, :len(nz)] = True
    return idx, ok


# ------------------------------------------------------------------ from the live environment
def snapshot_from_env(env, geo: Geometry) -> Snapshot:
    fleet, trips, ph = env.fleet_state, env.trip_state, env.physics
    dev = geo.device
    H, c = geo.H, geo.count_scale
    pos = fleet.positions.to(dev)
    E = fleet.socs.to(dev)
    avail = fleet.get_available_mask().to(dev)
    busy = ~avail
    status = fleet.status.to(dev)

    def count(mask, idx=pos, w=None):
        out = torch.zeros(H, device=dev)
        return out.index_add_(0, idx[mask], torch.ones(int(mask.sum()), device=dev) if w is None else w[mask])

    n_dec, n_busy = count(avail), count(busy)
    e_busy = count(busy, w=E) / n_busy.clamp(min=1) / ph.battery_kwh
    e_dec = count(avail, w=E) / n_dec.clamp(min=1) / ph.battery_kwh
    unassigned = trips.get_unassigned_mask().to(dev)
    new = unassigned & (trips.wait_steps.to(dev) == 0)
    old = unassigned & ~new
    pick, drop = trips.pickup_hex.to(dev), trips.dropoff_hex.to(dev)
    n_old = count(old, pick)
    arr_o, arr_d = count(new, pick), count(new, drop)
    n_in = count(unassigned, drop)
    occ = env.station_state.occupied.to(dev).float()
    free = (env.station_state.ports.to(dev).float() - occ)
    port_free = torch.zeros(H, device=dev)
    port_free[geo.st_cells] = free / geo.ports
    charging = (status == VehicleStatus.CHARGING) | (status == VehicleStatus.CHARGE_TRAVEL)
    chg_here = count(charging & (status == VehicleStatus.CHARGING)) / geo.ports
    tgt = fleet.target_hex.to(dev)
    moving = busy & (tgt >= 0)
    inbound = count(moving, tgt.clamp(min=0))
    hexf = torch.stack([
        torch.log1p(n_dec / c), torch.log1p(n_busy / c), e_busy, e_dec, torch.log1p(n_old / c),
        arr_o / c, arr_d / c, torch.log1p(n_in / c), geo.has_station, port_free, chg_here * geo.has_station,
        torch.log1p(inbound / c)], dim=1)
    from ..features.builder import clock_features
    _, ts, tc, ds, dc = clock_features(env.clock_s)
    n_chg = float(charging.sum())
    feeder_load = env.episode_info.feeder_load_kw
    glob = torch.tensor([ts, tc, ds, dc, feeder_load / ph.feeder_kw, n_chg / max(geo.S * geo.ports, 1.0),
                         float(E.mean()) / ph.battery_kwh, float(unassigned.sum()) / geo.V, float(avail.sum()) / geo.V],
                        device=dev)
    committed = (status == VehicleStatus.CHARGING) & ~avail
    fixed = float(torch.minimum(fleet.charge_power.to(dev)[committed], ph.max_power(E[committed])).sum()) if committed.any() else 0.0
    n_dec_chg = int((avail & (status == VehicleStatus.CHARGING)).sum())
    share = max(0.0, ph.feeder_kw - fixed) / max(1, n_dec_chg)
    return Snapshot(hex=hexf[None], xi=torch.cat([arr_o, arr_d])[None], free_ports=free[None], glob=glob[None],
                    exec_counts=torch.zeros(1, H, N_CODES, device=dev), feeder_share=torch.tensor([share], device=dev),
                    serve_intent=torch.zeros(1, H, device=dev))


def contexts_from_env(env, geo: Geometry) -> Context:
    fleet, trips = env.fleet_state, env.trip_state
    avail = fleet.get_available_mask()
    dec = avail.nonzero(as_tuple=True)[0]
    cells = fleet.positions[dec].cpu().numpy()
    energy = fleet.socs[dec].cpu().numpy().astype(np.float64)
    open_slots = trips.get_unassigned_mask().nonzero(as_tuple=True)[0].cpu().numpy()
    pick = trips.pickup_hex.cpu().numpy()[open_slots]
    tkm = trips.distance_km.cpu().numpy().astype(np.float64)[open_slots]
    fare = trips.fare.cpu().numpy().astype(np.float64)[open_slots]
    vi, oi, _ = env.candidates.serve_pairs(cells, energy, pick, tkm)
    n = len(dec)
    n_orders = np.bincount(vi, minlength=n)
    order = np.zeros((n, 3))
    if len(vi):
        dpick = env.candidates.dist[cells[vi], pick[oi]]
        # nearest order per vehicle (ties: higher fare)
        key = np.lexsort((-fare[oi], dpick, vi))
        first = key[np.r_[True, vi[key][1:] != vi[key][:-1]]]
        order[vi[first]] = np.stack([dpick[first] / 10.0, tkm[oi[first]] / 10.0, fare[oi[first]] / 30.0], 1)
    st = fleet.status[dec]
    dev = geo.device
    return Context(
        snap=torch.zeros(n, dtype=torch.long, device=dev),
        cell=fleet.positions[dec].to(dev).long(),
        energy=fleet.socs[dec].to(dev).float(),
        charging=(st == VehicleStatus.CHARGING).to(dev),
        own_station=torch.where(st == VehicleStatus.CHARGING, fleet.charging_station[dec], torch.full_like(dec, -1)).to(dev).long(),
        serve_ok=torch.as_tensor(n_orders > 0, device=dev),
        n_orders=torch.as_tensor(n_orders, dtype=torch.float32, device=dev),
        order=torch.as_tensor(order, dtype=torch.float32, device=dev),
    ), dec


# ------------------------------------------------------------------ network inputs
def build_inputs(snap: Snapshot, ctx: Context, geo: Geometry, projected: bool) -> Dict[str, torch.Tensor]:
    """Every tensor the actor / critics / value network read for the vehicles in ``ctx``.

    ``projected`` selects the power bounds of the executor in use: the MILP needs p >= p_min,
    the raw executor only p >= 0 (both cap at P_max and the battery headroom)."""
    ph, dev = geo.ph, geo.device
    b, cell, E = ctx.snap, ctx.cell, ctx.energy
    B = len(cell)
    hexb = snap.hex[b]                                                               # [B, H, F]
    ring = geo.ring_idx[cell]; rmask = geo.ring_mask[cell].float()
    ring_feat = torch.gather(hexb, 1, ring[..., None].expand(-1, -1, F_HEX))
    local = (ring_feat * rmask[..., None]).sum(1) / rmask.sum(1, keepdim=True)        # [B, F]
    exec_ring = torch.gather(snap.exec_counts[b], 1, ring[..., None].expand(-1, -1, N_CODES))
    ex = (exec_ring * rmask[..., None]).sum(1)
    meanfield = ex / ex.sum(1, keepdim=True).clamp(min=1.0)                           # [B, 5]
    # Contention for orders in the 1-ring at this epoch: share of SERVE intentions that got an
    # order (Laplace-smoothed; 1 when nobody tried). Used by Pi-hat for SERVE (Q21).
    want = (torch.gather(snap.serve_intent[b], 1, ring) * rmask).sum(1)
    p_serve = ((ex[:, 0] + 1.0) / (want + 1.0)).clamp(max=1.0)
    # reposition slots (slot 0 = stay is always valid)
    sc = geo.slot_cells[cell]                                                        # [B, K]
    in_dom = sc >= 0
    scc = sc.clamp(min=0)
    sd = geo.dist[cell[:, None], scc]
    e_after = E[:, None] - ph.energy_per_km * sd
    slot_ok = in_dom & (e_after >= ph.reserve_kwh - EPS)
    slot_ok[:, 0] = True
    slot_hex = torch.gather(hexb, 1, scc[..., None].expand(-1, -1, F_HEX)) * in_dom[..., None]
    slot_feat = torch.cat([slot_hex, (sd / 10.0)[..., None], (e_after / ph.battery_kwh)[..., None]], -1) * in_dom[..., None]
    # charging options: 5 nearest stations + continue at the held port
    lo = ph.p_min_kw if projected else 1e-3
    st = geo.near_st[cell]                                                           # [B, 5]
    st_real = st >= 0
    stc = st.clamp(min=0)
    sdist = geo.dist[cell[:, None], geo.st_cells[stc]] * st_real
    e_arr = E[:, None] - ph.energy_per_km * sdist
    free = torch.gather(snap.free_ports[b], 1, stc) * st_real
    st_ok = st_real & (free >= 1) & (e_arr >= -EPS) & (ph.max_power(e_arr.clamp(min=0)) >= lo) & (st != ctx.own_station[:, None])
    cont_ok = ctx.charging & (ph.max_power(E) >= lo)
    opt_ok = torch.cat([st_ok, cont_ok[:, None]], 1)
    opt_feat = torch.cat([
        torch.cat([sdist / 10.0, torch.zeros(B, 1, device=dev)], 1)[..., None],
        torch.cat([free / geo.ports, torch.zeros(B, 1, device=dev)], 1)[..., None],
        torch.cat([e_arr, E[:, None]], 1)[..., None] / ph.battery_kwh,
        torch.cat([torch.zeros(B, 5, device=dev), torch.ones(B, 1, device=dev)], 1)[..., None]], -1)
    opt_ub = torch.cat([ph.max_power(e_arr.clamp(min=0)), ph.max_power(E)[:, None]], 1)    # kW
    share = snap.feeder_share[b]
    if projected:                       # continuing chargers share the feeder room of this epoch
        opt_ub[:, CONTINUE] = torch.minimum(opt_ub[:, CONTINUE], share)
    mode_ok = torch.stack([ctx.serve_ok, torch.ones(B, dtype=torch.bool, device=dev), opt_ok.any(1)], 1)
    veh = torch.stack([
        E / ph.battery_kwh, (E - ph.reserve_kwh) / ph.battery_kwh, ctx.charging.float(), ctx.serve_ok.float(),
        torch.log1p(ctx.n_orders), ctx.order[:, 0], ctx.order[:, 1], ctx.order[:, 2], geo.has_station[cell]], 1)
    ports = free / geo.ports                                                         # [B, 5]
    return dict(veh=veh, local=local, ports=ports, meanfield=meanfield, glob=snap.glob[b],
                slot_feat=slot_feat, slot_ok=slot_ok, slot_cells=sc, opt_feat=opt_feat, opt_ok=opt_ok,
                opt_lo=torch.full((B, N_STATION_OPTIONS), lo, device=dev), opt_ub=opt_ub, mode_ok=mode_ok,
                n2_hex=torch.gather(hexb, 1, geo.n2_idx[cell][..., None].expand(-1, -1, F_HEX)), cell=cell,
                stations=st, p_serve=p_serve)
