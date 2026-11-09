"""Episode loop of the vehicle-level agent (workstream E5, E9-E12; Algorithm 2).

Every decision epoch: snapshot + contexts -> intentions -> projection (MILP) or the raw executor
-> env.step -> executed actions and per-vehicle rewards -> transitions close when each vehicle
decides again -> ``updates_per_step`` gradient updates (interleaved with the environment, E11).
Warm-up epochs draw uniform intentions over the valid candidates and go through the same
projection / executor path (E12).

Per-episode diagnostics guard against policy collapse: intention and executed mode shares,
Pi-hat / Pi agreement, entropies, energy levels and vehicles stuck below the serve threshold.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..simulator.executor import EXEC_CHARGE, EXEC_CONTINUE, EXEC_REPOS, EXEC_SERVE, EXEC_STAY
from ..simulator.projection import Intention, check_executed
from ..state import VehicleStatus
from .features import (CHARGE_C, CONT_C, CONTINUE, N_CODES, REPOS_C, SERVE_C, STAY_C, contexts_from_env,
                       snapshot_from_env)
from .policy import Sample, codes, joint_log_prob, sample
from .replay import ReplayBuffer, TransitionTracker, ctx_rows

EXEC_TO_CODE = {EXEC_SERVE: SERVE_C, EXEC_REPOS: REPOS_C, EXEC_STAY: STAY_C, EXEC_CHARGE: CHARGE_C, EXEC_CONTINUE: CONT_C}


@dataclass
class EpisodeReport:
    steps: int = 0
    reward: float = 0.0
    revenue: float = 0.0
    profit: float = 0.0
    served: int = 0
    dropped: int = 0
    loaded: int = 0
    intention_share: Dict[str, float] = field(default_factory=dict)   # among deciders
    executed_share: Dict[str, float] = field(default_factory=dict)
    pi_hat_match: float = 0.0
    serve_available_share: float = 0.0     # deciders with an energy-feasible order nearby
    serve_when_available: float = 0.0      # P(intention = serve | serve available)
    low_energy_idle_share: float = 0.0     # deciders below 20 % that neither charge nor move to charge
    mean_energy_end: float = 0.0
    stuck_vehicles_end: int = 0            # IDLE below E_min + one trip's worth (cannot serve)
    updates: int = 0
    update_stats: Dict[str, float] = field(default_factory=dict)
    projection: Dict = field(default_factory=dict)
    seconds: float = 0.0
    discarded_reward: float = 0.0          # rewards of transitions still open at the end (dropped)
    mean_delta: float = 0.0                # epochs per closed transition (sojourn length, training only)
    timing: Dict[str, float] = field(default_factory=dict)   # mean ms per epoch by stage, max epoch (F6)


def random_sample(inp, generator=None) -> Sample:
    """Uniform intention over the valid candidates (warm-up)."""
    out = {k: torch.zeros(m.shape, device=m.device).masked_fill(~m, -1e9)
           for k, m in (("mode", inp["mode_ok"]), ("slot", inp["slot_ok"]), ("station", inp["opt_ok"]))}
    B = inp["mode_ok"].shape[0]
    out["mu"] = torch.zeros(B, device=inp["mode_ok"].device)
    out["log_std"] = torch.full((B,), 0.5, device=inp["mode_ok"].device)
    return sample(out, 1.0, relaxed=False, generator=generator)


def run_episode(env, agent, *, replay: Optional[ReplayBuffer] = None, projection=None, seed=None,
                deterministic=False, warmup_steps: int = 0, train: bool = False, robust: bool = False,
                rng: Optional[np.random.Generator] = None, episode_start_s=None,
                on_step: Optional[Callable] = None, max_steps: Optional[int] = None) -> EpisodeReport:
    """``on_step(t, info, intent)`` is called after every env step; ``intent`` [V] holds the env
    action code each vehicle intended (-1 for vehicles that did not decide), before projection."""
    geo, dev = agent.geo, agent.device
    cfg = agent.cfg
    t0 = time.time()
    env.reset(seed=seed, episode_start_s=episode_start_s)
    tracker = TransitionTracker(env.num_vehicles, agent.gamma, agent.semi_mdp)
    if replay is not None:
        replay.new_episode()
    rng = rng or np.random.default_rng(seed)
    gen = torch.Generator(device=dev)          # samples are drawn on the networks' device
    gen.manual_seed(int(rng.integers(2**31)))
    rep = EpisodeReport()
    int_counts, exe_counts = np.zeros(3), np.zeros(N_CODES)
    match, n_dec, serve_av, serve_pick, low_idle, low_n = 0, 0, 0, 0, 0, 0
    upd: List[Dict[str, float]] = []
    proj_row0 = len(projection.log.rows) if projection is not None else 0
    t, done = 0, False
    delta_sum, delta_n = 0, 0
    clock = {"policy": 0.0, "projection": 0.0, "env": 0.0, "learn": 0.0}
    max_epoch, total_epoch = 0.0, 0.0
    while not done:
        t_start = time.perf_counter()
        snap = snapshot_from_env(env, geo)
        ctx, dec = contexts_from_env(env, geo)
        dec_np = dec.cpu().numpy()
        if replay is not None:
            replay.put_snapshot(t, snap)
            closed = tracker.close(dec_np, t, ctx_rows(ctx, "n_"), done=False)
            if closed:
                replay.add_transitions(closed)
                delta_sum, delta_n = delta_sum + int(closed["delta"].sum()), delta_n + len(closed["delta"])
        warm = warmup_steps > 0 and t < warmup_steps
        inp = agent.inputs(snap, ctx)
        out = None
        if warm:
            s = random_sample(inp, gen)
        else:
            s, inp, out = agent.act(snap, ctx, deterministic=deterministic, generator=gen)
        # intention -> environment action
        V = env.num_vehicles
        mode_i = s.mode.argmax(1)
        slot_i = s.slot.argmax(1)
        opt_i = s.opt.argmax(1)
        target = inp["slot_cells"].gather(1, slot_i[:, None]).squeeze(1)
        target = torch.where(target >= 0, target, ctx.cell)
        stations = torch.cat([inp["stations"], ctx.own_station[:, None]], 1).gather(1, opt_i[:, None]).squeeze(1)
        env_mode = torch.tensor([0, 2, 1], device=dev)[mode_i]
        full_mode = torch.full((V,), -1, dtype=torch.long)
        full_target = env.fleet_state.positions.clone().long()
        full_station = torch.full((V,), -1, dtype=torch.long)
        full_power = torch.zeros(V)
        d_cpu = dec.cpu()
        full_mode[d_cpu] = env_mode.cpu()
        full_target[d_cpu] = target.cpu()
        full_station[d_cpu] = torch.where(mode_i == 2, stations, torch.full_like(stations, -1)).cpu()
        full_power[d_cpu] = s.p.cpu()
        t_policy = time.perf_counter()
        clock["policy"] += t_policy - t_start
        if projection is not None:
            # Intention.mode_probs columns follow the env codes (serve, charge, reposition), not the
            # actor's mode order (serve, reposition, charge)
            probs = torch.zeros(V, 3)
            probs[d_cpu] = F.one_hot(env_mode, 3).float().cpu()
            st_int = torch.full((V,), -1, dtype=torch.long)
            st_int[d_cpu] = stations.cpu()
            a = projection.project(Intention(mode_probs=probs, repos_target=full_target, power_frac=full_power, station=st_int))
            t_proj = time.perf_counter()
            clock["projection"] += t_proj - t_policy
            _, _, d, info = env.step(a.mode, a.target, **a.step_kwargs())
            check_executed(info, projection)
        else:
            t_proj = t_policy
            _, _, d, info = env.step(full_mode, full_target, vehicle_charge_power=full_power, charge_station=full_station)
        t_env = time.perf_counter()
        clock["env"] += t_env - t_proj
        done = bool(d) or (max_steps is not None and t + 1 >= max_steps)
        if on_step is not None:
            on_step(t, info, full_mode)
        # executed actions of the deciders
        ex = info.executed
        em = ex.mode[dec].cpu().numpy()
        cells = ctx.cell.cpu().numpy()
        code = np.array([EXEC_TO_CODE[int(m)] for m in em], np.int64)
        tgt = ex.target_cell[dec].cpu().numpy()
        slot = np.full(len(dec_np), -1, np.int64)
        rep_m = code == REPOS_C
        slot[rep_m] = env.candidates.slot_of[cells[rep_m], tgt[rep_m]]
        slot[code == STAY_C] = 0
        if (slot[rep_m] < 0).any():
            raise RuntimeError("executed reposition target is not a slot of the vehicle's cell")
        opt = np.full(len(dec_np), -1, np.int64)
        chg = code == CHARGE_C
        if chg.any():
            near = env.candidates.near_stations[cells[chg]]
            st_ex = ex.station[dec].cpu().numpy()[chg]
            hit = near == st_ex[:, None]
            if not hit.any(1).all():
                raise RuntimeError("executed station is not a candidate of the vehicle's cell")
            opt[chg] = hit.argmax(1)
        opt[code == CONT_C] = CONTINUE
        power = ex.charge_frac[dec].cpu().numpy().astype(np.float32)     # share of the free capacity (charge)
        order = np.zeros((len(dec_np), 3), np.float32)
        srv = code == SERVE_C
        if srv.any():
            trips = ex.trip[dec].cpu().numpy()[srv]
            ts = env.trip_state
            pk = ts.pickup_hex.cpu().numpy()[trips]
            order[srv] = np.stack([env.candidates.dist[cells[srv], pk] / 10.0,
                                   ts.distance_km.cpu().numpy()[trips] / 10.0, ts.fare.cpu().numpy()[trips] / 30.0], 1)
        # diagnostics
        int_codes = codes(s.mode, s.slot, s.opt).argmax(1).cpu().numpy()
        match += int((int_codes == code).sum())
        n_dec += len(dec_np)
        np.add.at(int_counts, mode_i.cpu().numpy(), 1)
        np.add.at(exe_counts, code, 1)
        sv = ctx.serve_ok.cpu().numpy()
        serve_av += int(sv.sum())
        serve_pick += int((sv & (mode_i.cpu().numpy() == 0)).sum())
        e = ctx.energy.cpu().numpy()
        low = e < env.config.vehicle.low_energy_fraction * env.physics.battery_kwh
        low_n += int(low.sum())
        low_idle += int((low & (code == STAY_C)).sum())
        if replay is not None:
            counts = torch.zeros(geo.H, N_CODES)
            counts.index_put_((torch.as_tensor(cells), torch.as_tensor(code)), torch.ones(len(code)), accumulate=True)
            intents = torch.zeros(geo.H).index_add_(0, torch.as_tensor(cells), (mode_i.cpu() == 0).float())
            replay.put_exec_counts(t, counts, intents)
            rows = ctx_rows(ctx, "")
            # behaviour log-probability of the intention (on-policy learners)
            logp = (joint_log_prob(out, mode_i, slot_i, opt_i, s.u).cpu().numpy().astype(np.float32)
                    if out is not None and getattr(agent, "on_policy", False) else np.zeros(len(dec_np), np.float32))
            rows.update(t=np.full(len(dec_np), t, np.int64), veh=dec_np.astype(np.int64), mode=mode_i.cpu().numpy(),
                        i_slot=slot_i.cpu().numpy(), i_opt=opt_i.cpu().numpy(), i_u=s.u.cpu().numpy().astype(np.float32),
                        logp=logp, code=code, slot=slot, opt=opt, power=power, ex_order=order)
            tracker.start(dec_np, t, rows, ex.duration[dec].cpu().numpy())
            tracker.reward(info.per_vehicle_reward.cpu().numpy().astype(np.float64))
        rep.reward += float(info.per_vehicle_reward.sum())
        t += 1
        t_learn = time.perf_counter()
        if train and not warm and replay is not None and not getattr(agent, "on_policy", False):
            for _ in range(cfg.agent.updates_per_step):
                batch = replay.sample(cfg.training.batch_size, rng, dev)
                if batch is None:
                    break
                upd.append(agent.update(batch, robust=robust))
        t_end = time.perf_counter()
        clock["learn"] += t_end - t_learn
        max_epoch = max(max_epoch, t_end - t_start)
        total_epoch += t_end - t_start
    # The episode is a window of a continuing day, so its end truncates rather than terminates
    # (time-limit truncation): vehicles deciding at the final epoch close normally and are
    # bootstrapped; transitions of vehicles still busy are dropped instead of being marked terminal.
    if replay is not None:
        snap = snapshot_from_env(env, geo)
        ctx, dec = contexts_from_env(env, geo)
        replay.put_snapshot(t, snap)
        closed = tracker.close(dec.cpu().numpy(), t, ctx_rows(ctx, "n_"), done=False)
        if closed:
            replay.add_transitions(closed)
        rep.discarded_reward = float(tracker.R[tracker.open].sum())
        tracker.open[:] = False
    info = env.episode_info
    rep.steps, rep.revenue, rep.profit = t, info.revenue, info.profit
    rep.served, rep.dropped, rep.loaded = info.trips_served, info.trips_dropped, info.trips_loaded
    names = ("serve", "reposition", "charge")
    rep.intention_share = {n: float(c / max(n_dec, 1)) for n, c in zip(names, int_counts)}
    rep.executed_share = {n: float(c / max(n_dec, 1)) for n, c in zip(("serve", "reposition", "stay", "charge", "continue"), exe_counts)}
    rep.pi_hat_match = match / max(n_dec, 1)
    rep.mean_delta = delta_sum / max(delta_n, 1)
    rep.serve_available_share = serve_av / max(n_dec, 1)
    rep.serve_when_available = serve_pick / max(serve_av, 1)
    rep.low_energy_idle_share = low_idle / max(low_n, 1)
    E = env.fleet_state.socs
    rep.mean_energy_end = float(E.mean()) / env.physics.battery_kwh
    one_trip = env.physics.reserve_kwh + env.physics.energy_per_km * 3.0
    rep.stuck_vehicles_end = int(((env.fleet_state.status == VehicleStatus.IDLE) & (E < one_trip)).sum())
    rep.updates = len(upd)
    if upd:
        rep.update_stats = {k: float(np.mean([u[k] for u in upd if k in u])) for k in upd[-1]}
    if projection is not None:
        rep.projection = projection.log.summary(projection.log.rows[proj_row0:])
    rep.seconds = time.time() - t0
    n = max(t, 1)
    rep.timing = {f"{k}_ms": 1000.0 * v / n for k, v in clock.items()}
    rep.timing["other_ms"] = 1000.0 * (total_epoch - sum(clock.values())) / n   # features, diagnostics, replay rows
    rep.timing["epoch_ms"] = 1000.0 * total_epoch / n
    rep.timing["max_epoch_ms"] = 1000.0 * max_epoch
    return rep
