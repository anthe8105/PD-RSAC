"""Per-epoch metrics of one evaluation episode (workstream F5).

Attach ``StepRecorder.on_step`` to the episode loop (``run_episode(..., on_step=rec.on_step)``)
or call it after every ``env.step`` of a heuristic. It keeps one row per decision epoch with the
epoch's wall-clock time and the step increments of the economic counters, and accumulates:

- feeder: load actually drawn per epoch, peak, epochs above the limit (violations) and epochs at
  or above ``tight_fraction`` x the limit (the feeder is binding), for every variant (Q9);
- stations: ports held per station each epoch (bookings in travel + charging), mean utilisation
  and the share of epochs each station is full;
- bookings: new charging bookings intended by deciders and how many were not granted (no free
  port on the raw path, or changed by the projection); chargers that continue are not requests;
- projection (with the MILP): the outcome code of each epoch's call and whether the executed
  action came from Algorithm 1 (``fallback``: warm start only, no incumbent or solver error).

Rows are grouped by calendar day for the weekly table (F7).
"""

import datetime as _dt
from typing import Dict, List, Optional

import numpy as np
import torch

from ..simulator.executor import CHARGE, EXEC_CHARGE, EXEC_CONTINUE
from ..simulator.environment import FEEDER_TOL_KW
from ..simulator.projection import FALLBACK_CODES

_COUNTERS = ("revenue", "driving_cost", "energy_cost", "wait_penalty", "drop_penalty", "trips_served",
             "trips_completed", "trips_dropped", "trips_loaded", "energy_charged_kwh", "energy_bought_kwh",
             "infeasible_executed", "charge_no_port")


def _date(epoch_s: int) -> str:
    return (_dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=int(epoch_s))).strftime("%Y-%m-%d")


def _hhmm(epoch_s: int) -> str:
    return (_dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=int(epoch_s))).strftime("%H:%M")


class StepRecorder:

    def __init__(self, env, tight_fraction: float = 0.98, label: str = "", projection=None):
        self.env = env
        self.projection = projection
        self.label = label
        self.tight_fraction = float(tight_fraction)
        self.feeder_kw = float(env.physics.feeder_kw)
        self.ports = int(env.config.station.num_ports)
        self.num_stations = int(env.num_stations)
        self.reset()

    def reset(self) -> None:
        self.rows: List[Dict] = []
        self._prev = {k: 0.0 for k in _COUNTERS}
        self._held_sum = np.zeros(self.num_stations)
        self._full_epochs = np.zeros(self.num_stations)
        self._requests = 0
        self._denied = 0
        proj = getattr(self, "projection", None)
        self._proj_seen = len(proj.log.rows) if proj is not None else 0

    # ------------------------------------------------------------------ per epoch
    def on_step(self, t: int, info, intent: Optional[torch.Tensor] = None) -> None:
        env = self.env
        k = env.current_step - 1                       # the epoch that was just executed
        clock = env.epoch_clock_s(k)
        row = {"label": self.label, "step": int(t), "clock_s": int(clock), "date": _date(clock), "time": _hhmm(clock)}
        for key in _COUNTERS:
            cur = float(getattr(info, key))
            row[key] = cur - self._prev[key]
            self._prev[key] = cur
        row["profit"] = row["revenue"] - row["driving_cost"] - row["energy_cost"]
        load = float(info.feeder_load_kw)
        row["feeder_load_kw"] = load
        row["vehicles_charging"] = int(info.vehicles_charging_now)
        row["feeder_violation"] = int(load > self.feeder_kw + FEEDER_TOL_KW)
        row["feeder_tight"] = int(load >= self.tight_fraction * self.feeder_kw)

        fleet = env.fleet_state
        holds = fleet.holds_port_mask()
        held = torch.bincount(fleet.charging_station[holds].long().cpu(), minlength=self.num_stations).numpy()
        self._held_sum += held
        self._full_epochs += held >= self.ports
        row["ports_held"] = int(held.sum())
        row["stations_full"] = int((held >= self.ports).sum())

        req = den = 0
        if intent is not None and info.executed is not None and info.deciders is not None:
            dec = info.deciders.cpu()
            ex = info.executed.mode.cpu()
            want = dec & (intent.cpu().view(-1) == CHARGE) & (ex != EXEC_CONTINUE)
            req = int(want.sum())
            den = int((want & (ex != EXEC_CHARGE)).sum())
        self._requests += req
        self._denied += den
        row["charge_requests"], row["charge_denied"] = req, den
        row["proj_code"], row["fallback"] = "", 0
        if self.projection is not None:
            new = self.projection.log.rows[self._proj_seen:]
            self._proj_seen = len(self.projection.log.rows)
            if new:
                row["proj_code"] = new[-1]["code"]
                row["fallback"] = int(any(r["code"] in FALLBACK_CODES for r in new))
        self.rows.append(row)

    # ------------------------------------------------------------------ summaries
    @property
    def steps(self) -> int:
        return len(self.rows)

    def summary(self) -> Dict:
        n = max(self.steps, 1)
        loads = np.array([r["feeder_load_kw"] for r in self.rows]) if self.rows else np.zeros(1)
        util = self._held_sum / (n * self.ports)
        full = self._full_epochs / n
        tot = {k: float(sum(r[k] for r in self.rows)) for k in _COUNTERS}
        peak_i = int(loads.argmax())
        return {
            "steps": self.steps,
            "feeder_limit_kw": self.feeder_kw,
            "feeder_peak_kw": float(loads.max()),
            "feeder_peak_time": f"{self.rows[peak_i]['date']} {self.rows[peak_i]['time']}" if self.rows else None,
            "feeder_mean_kw": float(loads.mean()),
            "feeder_violation_steps": int(sum(r["feeder_violation"] for r in self.rows)),
            "feeder_violation_share": float(np.mean([r["feeder_violation"] for r in self.rows])) if self.rows else 0.0,
            "feeder_tight_share": float(np.mean([r["feeder_tight"] for r in self.rows])) if self.rows else 0.0,
            "feeder_tight_fraction": self.tight_fraction,
            "energy_charged_kwh": tot["energy_charged_kwh"],
            "energy_bought_kwh": tot["energy_bought_kwh"],
            "port_utilisation_mean": float(util.mean()) if len(util) else 0.0,
            "port_utilisation_max_station": float(util.max()) if len(util) else 0.0,
            "port_utilisation_by_station": [round(float(u), 4) for u in util],
            "station_full_share_by_station": [round(float(f), 4) for f in full],
            "stations_full_share_mean": float(full.mean()) if len(full) else 0.0,
            "charge_requests": int(self._requests),
            "charge_denied": int(self._denied),
            "charge_denied_rate": self._denied / max(self._requests, 1),
            "fallback_steps": int(sum(r.get("fallback", 0) for r in self.rows)),
        }

    def day_table(self) -> List[Dict]:
        """One row per calendar day of the episode (weekly test, F7)."""
        days: Dict[str, List[Dict]] = {}
        for r in self.rows:
            days.setdefault(r["date"], []).append(r)
        out = []
        for date, rows in days.items():
            s = {k: float(sum(r[k] for r in rows)) for k in _COUNTERS}
            loads = [r["feeder_load_kw"] for r in rows]
            out.append({
                "label": self.label, "date": date, "steps": len(rows),
                "profit": s["revenue"] - s["driving_cost"] - s["energy_cost"],
                "revenue": s["revenue"], "driving_cost": s["driving_cost"], "energy_cost": s["energy_cost"],
                "wait_penalty": s["wait_penalty"], "drop_penalty": s["drop_penalty"],
                "trips_loaded": int(s["trips_loaded"]), "trips_served": int(s["trips_served"]),
                "trips_dropped": int(s["trips_dropped"]),
                "service_rate": s["trips_served"] / max(s["trips_loaded"], 1.0),
                "energy_charged_kwh": s["energy_charged_kwh"],
                "feeder_peak_kw": max(loads), "feeder_violation_steps": int(sum(r["feeder_violation"] for r in rows)),
                "feeder_tight_share": float(np.mean([r["feeder_tight"] for r in rows])),
                "charge_requests": int(sum(r["charge_requests"] for r in rows)),
                "charge_denied": int(sum(r["charge_denied"] for r in rows)),
                "fallback_steps": int(sum(r.get("fallback", 0) for r in rows)),
            })
        return out
