# Giving up on a question: quit drift, patience, and the Wald likelihood of the times.

import math
import random

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from config import (
    ACHIEVEMENT_SCALE,
    DELTA_CLIP,
    MAX_REPEAT_ATTEMPTS,
    PATIENCE,
    REPEAT_COST_SCALE,
)
from initiation import BOUND, SIGMA

LAPSE = 0.02          # share of times the model does not describe, kept from dominating the fit
LAPSE_SECONDS = 60.0  # those times are uniform over this window
PATIENCE_BOUNDS = (0.3, 60.0)
# How long solving takes, for simulated people only: median = SOLVE_BASE / p^SOLVE_EXPONENT.
SOLVE_BASE = 4.0
SOLVE_EXPONENT = 0.7
SOLVE_SPREAD = 0.5   # lognormal sigma
P_FLOOR = 1e-3


# (value, cost) of carrying on with a question the person has chance `p` on. `delta` is
# the prediction error carried out of the slot just left: how the last question went, if
# they were shown how it went, moves what this one feels worth.
def conditions(p, stage_amt, delta=0.0):
    p = np.clip(np.asarray(p, dtype=float), P_FLOOR, 1 - P_FLOOR)
    value = ACHIEVEMENT_SCALE * p * (1 - p) / np.asarray(stage_amt, dtype=float) # what solving it adds
    value = value + np.clip(np.asarray(delta, dtype=float), -DELTA_CLIP, DELTA_CLIP)
    cost = REPEAT_COST_SCALE * np.minimum(1.0 / p, MAX_REPEAT_ATTEMPTS) # the effort it takes
    return value, cost


def quit_drift(f, k, value, cost):
    return k * cost - f * value # the start drive, reversed: it builds toward giving up


# log density of giving up at model time t (inverse Gaussian; defective when mu <= 0).
def log_density(t, mu):
    t = np.asarray(t, dtype=float)
    return (math.log(BOUND) - 0.5 * math.log(2 * math.pi) - math.log(SIGMA) - 1.5 * np.log(t)
            - (BOUND - mu * t) ** 2 / (2 * SIGMA**2 * t))


# log P(still going at model time t): an answer after t seconds is right-censored.
def log_survival(t, mu):
    t = np.asarray(t, dtype=float)
    mu = np.asarray(mu, dtype=float)
    scale = SIGMA * np.sqrt(t)
    log_cdf = np.logaddexp(
        log_ndtr((mu * t - BOUND) / scale),
        2 * mu * BOUND / SIGMA**2 + log_ndtr((-mu * t - BOUND) / scale), # the defective mass
    )
    return np.log(np.maximum(-np.expm1(np.minimum(log_cdf, 0.0)), 1e-300))


# Model time until giving up, or inf if they never do.
def draw_quit_time(mu):
    if mu <= 0:
        # They reach the bound at all with probability exp(2 mu B / s^2).
        if random.random() >= math.exp(2 * mu * BOUND / SIGMA**2):
            return math.inf
        mu = -mu # given that they do, the time is distributed as for drift |mu|
    if mu < 1e-9:
        z = random.gauss(0.0, 1.0)
        return (BOUND / SIGMA) ** 2 / max(z * z, 1e-12)  # driftless: Levy distribution
    return float(np.random.wald(BOUND / mu, BOUND**2 / SIGMA**2))


# Seconds a simulated person needs to reach an answer.
def draw_solve_seconds(p):
    median = SOLVE_BASE / max(p, P_FLOOR) ** SOLVE_EXPONENT
    return median * math.exp(random.gauss(0.0, SOLVE_SPREAD))


# One row per question worked on: value, cost, whether they gave up, seconds.
def attempts_to_frame(attempts):
    # `p` is the chance as the environment knew it -- all a fit on a real person has.
    # Records without a time (an exit nobody saw) are left out.
    rows = [(a["p"], a["rt"], bool(a.get("skipped", False)), a.get("stage_amt") or 4,
             float(a.get("delta") or 0.0))
            for a in attempts
            if a.get("rt") is not None and a["rt"] > 0 and a.get("p") is not None]
    df = pd.DataFrame(rows, columns=["p", "rt", "skipped", "stage_amt", "delta"])
    value, cost = conditions(df["p"], df["stage_amt"], df["delta"])
    return pd.DataFrame({"value": value, "cost": cost, "skipped": df["skipped"].to_numpy(dtype=bool),
                         "seconds": df["rt"].to_numpy(dtype=float)})


# -log L of the attempts in `frame` (from attempts_to_frame).
def neg_log_likelihood(f, k, frame, patience=PATIENCE, lapse=LAPSE):
    if len(frame) == 0:
        return 0.0
    mu = quit_drift(f, k, frame["value"].to_numpy(), frame["cost"].to_numpy())
    s = frame["seconds"].to_numpy()
    t = s / patience # seconds into model time
    gave_up = frame["skipped"].to_numpy(dtype=bool)
    with np.errstate(divide="ignore"):
        # log(patience) is the change of variable to a density per second; without it
        # patience could not be fitted.
        model_d = np.log1p(-lapse) + log_density(t, mu) - math.log(patience)
        model_s = np.log1p(-lapse) + log_survival(t, mu)
        lapse_d = np.where(s < LAPSE_SECONDS, np.log(lapse / LAPSE_SECONDS), -np.inf)
        lapse_s = np.log(lapse * np.clip(1.0 - s / LAPSE_SECONDS, 0.0, None))
    # Gave up: the density at that time. Answered: the survival past it.
    ll = np.where(gave_up, np.logaddexp(model_d, lapse_d), np.logaddexp(model_s, lapse_s))
    return float(-ll.sum())


# This person's patience (seconds per unit of model time), with f and k given.
def fit_patience(f, k, frame, bounds=PATIENCE_BOUNDS):
    from scipy.optimize import minimize_scalar

    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    grid = np.linspace(lo, hi, 25) # coarse pass in log space, since patience is a scale
    nll = [neg_log_likelihood(f, k, frame, math.exp(x)) for x in grid]
    i = int(np.argmin(nll))
    step = grid[1] - grid[0]
    res = minimize_scalar(lambda x: neg_log_likelihood(f, k, frame, math.exp(x)), # refine around it
                          bounds=(max(lo, grid[i] - step), min(hi, grid[i] + step)),
                          method="bounded")
    return math.exp(res.x)


# (f, k) from persistence alone: what answer and skip times say without decisions.
def fit_persistence(frame, patience=PATIENCE, bounds=((0.0, 3.0), (0.0, 3.0))):
    from scipy.optimize import minimize

    best = None
    for x0 in ((0.5, 0.5), (1.5, 1.5), (0.5, 2.0), (2.0, 0.5)): # several starts; the surface is flat
        res = minimize(lambda x: neg_log_likelihood(x[0], x[1], frame, patience), x0,
                       bounds=bounds, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return tuple(float(v) for v in best.x)
