"""The accumulator's likelihood in closed form: no Fokker-Planck solve, any (μ, BOUND, T0).

The accumulator (accumulator.build_model) starts at 0 between bounds at ±BOUND, drifts at
a constant μ with noise SIGMA, and stops at T_DUR. Its first-passage density has a
standard series (Navarro & Fuss, 2009), and the chance of reaching neither bound by the
deadline has one too, so a decision's likelihood is a few vectorized exponentials. pyddm
gets the same numbers by solving the Fokker-Planck equation on a grid
(tests/test_fastlik.py checks the two agree).

What pyddm adds on top is added here the same way:
- T0 shifts every passage later; mass shifted past T_DUR becomes undecided.
- The lapse mixture (accumulator.LAPSE_MIX, pyddm's uniform mixture): a decided trial's
  density is (1 − m)·density + (m / 2) / T_DUR for each choice, and the undecided mass is
  scaled by (1 − m).

Densities are per second; pyddm's likelihood is the same up to its time bin, a constant.
"""
import numpy as np

from .model.accumulator import BOUND, LAPSE_MIX, SIGMA, T_DUR

SMALL_TERMS = np.arange(-7, 8)     # small-time series; t / a² stays under ~1 at T_DUR
LARGE_TERMS = np.arange(1, 40, 2)  # odd k, the survival series


# Density of reaching the lower bound at time t (no T0, no lapses). Unit noise after
# dividing μ and the bound separation by SIGMA; the start is midway (w = 1/2).
def _lower_density(t, mu, bound, sigma=SIGMA):
    v = mu / sigma
    a = 2.0 * bound / sigma
    u = np.maximum(t, 1e-12) / a ** 2
    w = 0.5
    k = SMALL_TERMS[:, None]
    series = np.sum((w + 2 * k) * np.exp(-((w + 2 * k) ** 2) / (2 * u)), axis=0)
    f_std = series / np.sqrt(2 * np.pi * u ** 3)
    return np.where(t > 0, f_std * np.exp(-v * a * w - v * v * t / 2) / a ** 2, 0.0)


# Density at the upper bound (GO): the lower one with the drift reversed.
def passage_density(t, mu, upper, bound=BOUND, sigma=SIGMA):
    mu = np.where(upper, -mu, mu)
    return _lower_density(t, mu, bound, sigma)


# P(neither bound by time t), from the eigenfunction series of the absorbed diffusion.
def survival(t, mu, bound=BOUND, sigma=SIGMA):
    v = np.asarray(mu, float) / sigma
    a = 2.0 * bound / sigma
    t = np.maximum(np.asarray(t, float), 1e-9)
    k = LARGE_TERMS[:, None]
    kpa = k * np.pi / a
    sin = np.where(k % 4 == 1, 1.0, -1.0)  # sin(kπ/2) for odd k
    terms = sin * np.exp(-(k * k) * np.pi ** 2 * t / (2 * a * a)) * kpa / (v * v + kpa ** 2)
    # 2·cosh(v·a/2)·exp(-v²t/2), in logs so a large drift cannot overflow
    log_front = np.logaddexp(v * a / 2, -v * a / 2) - v * v * t / 2
    return np.clip((2.0 / a) * np.exp(log_front) * np.sum(terms, axis=0), 0.0, 1.0)


# -log L of decisions. go: 1 upper, 0 lower, -1 undecided; rt in seconds (ignored when
# undecided); mu per decision. Arrays of one length; bound and t0 scalars.
def neg_log_likelihood(go, rt, mu, bound=BOUND, t0=0.0, lapse=LAPSE_MIX, t_dur=T_DUR):
    go = np.asarray(go)
    mu = np.asarray(mu, float)
    decided = go >= 0
    out = 0.0
    if decided.any():
        t = np.asarray(rt, float)[decided] - t0
        dens = passage_density(t, mu[decided], go[decided] == 1, bound)
        out -= np.sum(np.log((1 - lapse) * dens + 0.5 * lapse / t_dur))
    if (~decided).any():
        s = survival(t_dur - t0, mu[~decided], bound)
        out -= np.sum(np.log(np.maximum((1 - lapse) * s, 1e-300)))
    return float(out)


# Each decision's outcome probabilities on a time grid: (upper, lower) mass per bin of
# width dt, and the undecided mass. For expected (Fisher) information.
def outcome_probs(mu, bound=BOUND, t0=0.0, lapse=LAPSE_MIX, t_dur=T_DUR, dt=0.01):
    mu = np.asarray(mu, float)[:, None]
    t = np.arange(dt / 2, t_dur, dt)[None, :]
    up = ((1 - lapse) * passage_density(t - t0, mu, True, bound) + 0.5 * lapse / t_dur) * dt
    lo = ((1 - lapse) * passage_density(t - t0, mu, False, bound) + 0.5 * lapse / t_dur) * dt
    und = (1 - lapse) * survival(t_dur - t0, mu[:, 0], bound)
    return up, lo, und
