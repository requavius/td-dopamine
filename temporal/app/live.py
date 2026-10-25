"""What the accumulator predicts for the decision on screen, for the live model view.

The person decides; the model only predicts. Each start or redo prompt gets the
solution for its (value, cost) under the current (f, k, T0) estimate, and the
frontends draw it: the bounds, the mean drift, the predicted RT densities at each bound,
and sample paths run in real time beside the person's own clock. Nothing here draws
from the global `random` or numpy generators, so a seeded session is unchanged.
"""
import random

import numpy as np

from ..controller import PRIOR
from ..model.accumulator import BOUND, DT, SIGMA, T_DUR, initiation_drift, solution

BIN = 0.05  # seconds per point of the RT densities sent to the frontend


# (f, k, t0, note): the controller's estimate if it has one, else its prior.
def current_estimate(session):
    est = getattr(session.controller, "est", None)
    if est is None:
        return PRIOR[0], PRIOR[1], 0.0, "prior (controller has no estimate yet)"
    return est.f, est.k, est.t0, f"controller estimate from {est.rows} rows"


def _binned(pdf, n):
    pdf = np.asarray(pdf[: len(pdf) - len(pdf) % n], dtype=float)
    return pdf.reshape(-1, n).mean(axis=1)


# The view of one decision, as plain numbers. `decision` is Session.decision_now.
def model_view(session, decision):
    f, k, t0, note = current_estimate(session)
    value, cost = decision["value"], decision["cost"]
    sol = solution(f, k, value, cost, t0)
    n = max(1, round(BIN / DT))
    top = _binned(sol.pdf("_top"), n)
    bottom = _binned(sol.pdf("_bottom"), n)
    p_go, p_no = float(sol.prob("_top")), float(sol.prob("_bottom"))
    return {
        **decision,
        "f": f, "k": k, "t0": t0, "note": note,
        "drift": initiation_drift(value, cost, f, k),
        "sigma": SIGMA, "bound": BOUND, "t_dur": T_DUR, "bin": BIN,
        "p_go": p_go, "p_no": p_no, "p_undecided": float(sol.prob_undecided()),
        "mean_rt": float(sol.mean_decision_time()) if p_go + p_no > 0 else None,
        "top": top.round(5).tolist(), "bottom": bottom.round(5).tolist(),
    }


# Sample paths of the accumulator, stepped by the frontend in real time. Its own generator.
class Paths:
    def __init__(self, view, n=24, dt=0.01, seed=None):
        self.view, self.n, self.dt = view, n, dt
        self.rng = random.Random(seed)
        self.t = 0.0  # seconds since the prompt appeared
        self.x = [0.0] * n
        self.ended = [None] * n  # (time, +1 / -1) once a path reaches a bound
        self.trace = [[(view["t0"], 0.0)] for _ in range(n)]

    # Advance every path to `now` seconds after the prompt.
    def step_to(self, now):
        v = self.view
        now = min(now, v["t_dur"])
        sd = v["sigma"] * self.dt ** 0.5
        while self.t + self.dt <= now:
            self.t += self.dt
            if self.t < v["t0"]:
                continue
            for i in range(self.n):
                if self.ended[i] is not None:
                    continue
                self.x[i] += v["drift"] * self.dt + self.rng.gauss(0.0, sd)
                if abs(self.x[i]) >= v["bound"]:
                    side = 1 if self.x[i] > 0 else -1
                    self.x[i] = side * v["bound"]
                    self.ended[i] = (self.t, side)
                self.trace[i].append((self.t, self.x[i]))

    # (paths ended at the upper bound, at the lower bound) so far.
    def decided(self):
        go = sum(1 for e in self.ended if e and e[1] > 0)
        return go, sum(e is not None for e in self.ended) - go
