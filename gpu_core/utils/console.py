"""Console formatting shared by training and evaluation: durations, ETA, progress lines."""

import time
from collections import deque
from typing import Optional


def fmt_duration(seconds: float) -> str:
    """3725 -> '1h02m', 95 -> '1m35s', 7.2 -> '7.2s'."""
    if seconds is None or seconds != seconds:
        return "?"
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def rule(title: str = "", width: int = 100) -> str:
    if not title:
        return "=" * width
    pad = max(2, width - len(title) - 2)
    return f"{'=' * (pad // 2)} {title} {'=' * (pad - pad // 2)}"


class RateMeter:
    """Moving average of the time per unit (episode or epoch) for ETA estimates."""

    def __init__(self, window: int = 20):
        self.samples = deque(maxlen=window)

    def add(self, seconds: float, units: float = 1.0) -> None:
        if units > 0:
            self.samples.append(seconds / units)

    @property
    def per_unit(self) -> Optional[float]:
        return sum(self.samples) / len(self.samples) if self.samples else None

    def eta(self, remaining_units: float) -> Optional[float]:
        r = self.per_unit
        return None if r is None else r * max(0.0, remaining_units)


class EpisodeProgress:
    """``on_step`` hook printing progress of a long evaluation episode every ``every`` epochs:
    epochs done, wall-clock time of the simulation, running profit and service, s/epoch and ETA."""

    def __init__(self, env, label: str, every: Optional[int] = None, total: Optional[int] = None, inner=None):
        self.env, self.label, self.inner = env, label, inner
        self.total = int(total or env.episode_steps)
        self.every = int(every or max(1, self.total // 10))
        self.t0 = time.perf_counter()

    def __call__(self, t, info, intent=None):
        if self.inner is not None:
            self.inner(t, info, intent)
        k = t + 1
        if k % self.every and k != self.total:
            return
        el = time.perf_counter() - self.t0
        rate = el / k
        clock = self.env.epoch_clock_s(t)
        import datetime as _dt
        when = (_dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=int(clock))).strftime("%m-%d %H:%M")
        i = info
        print(f"    [{self.label}] epoch {k:>5}/{self.total} ({k / self.total:5.1%}) sim {when} | "
              f"profit ${i.profit:>10,.0f}  service {i.trips_served / max(i.trips_loaded, 1):6.1%} | "
              f"{rate:5.2f} s/epoch, elapsed {fmt_duration(el)}, ETA {fmt_duration(rate * (self.total - k))}", flush=True)
