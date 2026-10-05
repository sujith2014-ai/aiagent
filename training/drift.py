"""Drift detection for learned capabilities.
Concept drift (same inputs, new labels) is invisible without labels, so detection needs label queries to the environment. Two parts:
 - Cusum: a sequential test per capability (H0: error rate p0, H1: error rate p1); it alarms after a handful of net errors instead of waiting for a full window of samples.
 - ProbeScheduler: a label-query budget spent where the evidence is stalest, so detection does not depend on organic traffic or on organic feedback.
Probes run with the capability forced (no routing), so a routing mistake cannot be mistaken for model drift."""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field


@dataclass
class Cusum:
    p0: float = 0.05
    p1: float = 0.30
    h: float = 4.6                    # ln(99): about 1% false alarms per run of observations under H0
    s: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)

    def add(self, cap: str, wrong: bool) -> bool:
        llr = math.log(self.p1 / self.p0) if wrong else math.log((1 - self.p1) / (1 - self.p0))
        self.s[cap] = max(0.0, self.s.get(cap, 0.0) + llr); self.n[cap] = self.n.get(cap, 0) + 1
        return self.s[cap] >= self.h

    def reset(self, cap: str): self.s.pop(cap, None); self.n.pop(cap, None)


class WindowMonitor:
    """The Phase 14 original: alarm when a window of at least `min_samples` outcomes has accuracy below `threshold` (kept for the ablation)."""
    def __init__(self, window=12, min_samples=8, threshold=0.7):
        import collections
        self.w, self.min, self.thr = window, min_samples, threshold; self.win: dict = {}; self._c = collections
    def add(self, cap: str, wrong: bool) -> bool:
        d = self.win.setdefault(cap, self._c.deque(maxlen=self.w)); d.append(not wrong)
        return len(d) >= self.min and sum(d) / len(d) < self.thr
    def reset(self, cap: str): self.win.pop(cap, None)


class ProbeScheduler:
    """With probability `rate` per arrival, pick the capability probed longest ago (ties broken at random) and ask for `k` labelled examples."""
    def __init__(self, rate: float, k: int = 3, seed: int = 0):
        self.rate, self.k, self.rng = rate, k, random.Random(seed); self.last: dict[str, int] = {}; self.queries = 0

    def pick(self, t: int, caps: list[str]) -> str | None:
        if not caps or self.rng.random() >= self.rate: return None
        cap = min(caps, key=lambda c: (self.last.get(c, -1), self.rng.random())); self.last[cap] = t; self.queries += self.k
        return cap
