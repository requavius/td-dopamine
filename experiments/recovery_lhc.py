"""Recovery of f, k, γ, BOUND and T0 together, from the data the app records.

People are a Latin hypercube over

    f      0.2 to 1.5       reward sensitivity
    k      0.2 to 1.5       effort aversion
    γ      0.6 to 0.97      discount factor
    BOUND  1.0 to 2.0       caution (the model's 1.5 is in the middle)
    T0     0 to 0.4 s       non-decision time

N_PEOPLE of them, drawn once from a seeded hypercube, so every machine draws the same
people and `--people` only says which to run. Each plays the app's task: question types
with Elo ratings, skips on, InfoController steering from the default setting, 300 start
decisions plus their redo decisions. α is 0.05 for the person and the fit; SIGMA 0.7,
ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0.

Stage 1, `simulate`, saves each person's history to results/recovery_lhc/histories/:
every recorded decision (choice, RT, cost, node) and its value under each γ from 0.50 to
0.99 in steps of 0.01, from shadow TD learners at α 0.05 that run on the same rewards, as
the app could replay a person's record. This is the only expensive step, and it is done
once: every analysis below reads the histories.

Stage 2, `fit`, fits (f, k, γ, BOUND, T0) to each history by maximum likelihood with the
closed-form likelihood (temporal.fastlik), γ continuous (value interpolated between the
0.01 steps by a cubic spline), and computes, at the fit, the observed information
(finite-difference Hessian of -log L) and, at the truth, the expected (Fisher)
information. Value is used before rounding to 0.05: the rounding exists so pyddm can
reuse solved conditions, and it would make the likelihood jagged in γ. The person's own
draws used rounded value, so the fit carries that rounding as a small misspecification.

Claims, fixed before any fit (2026-10-28):
    L1  RMSE, bias (mean of recovered − true) and r for each of f, k, γ, BOUND, T0; no
        threshold, the numbers are the result
    L2  calibration: over all fits away from a parameter bound, the standardized errors
        (recovered − true) / SE, with SE from the observed information, have SD within
        0.8 to 1.25 for every parameter, and the true value is inside ±1.96 SE in 90% to
        98% of fits
    L3  the same with SE from the Fisher information at the truth
    L4  none: report the correlation of γ with f from the inverse Fisher information,
        across the hypercube

    python -m experiments.recovery_lhc simulate --people 0-49 --tag server
    python -m experiments.recovery_lhc fit
    python -m experiments.recovery_lhc --replot
"""
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from temporal import fastlik
from temporal.inference import T0_BOUNDS
from temporal.model import accumulator as acc

from .common import Quiet, _design_args, design, out_dir, run_pool, save_conditions, se
from .gamma_profile import run_app
from .misspecification import PersonConstants

D = design("recovery_lhc", people=100, decisions=300, seed=20261028,
           ranges={"f": (0.2, 1.5), "k": (0.2, 1.5), "g": (0.6, 0.97), "bound": (1.0, 2.0),
                   "t0": (0.0, 0.4)})
OUT = out_dir(D.out)
HIST = out_dir(f"{D.out}/histories")
N_PEOPLE = D.people
RANGES = D.ranges
G_FINE = tuple(round(0.50 + 0.01 * i, 2) for i in range(50))  # 0.50 to 0.99
SEED = D.seed


# The hypercube, the same on every machine.
def people():
    unit = qmc.LatinHypercube(d=len(RANGES), seed=SEED).random(N_PEOPLE)
    lo = np.array([r[0] for r in RANGES.values()])
    hi = np.array([r[1] for r in RANGES.values()])
    return [dict(zip(RANGES, row)) for row in lo + unit * (hi - lo)]


def history_path(i):
    return HIST / f"p{i:03d}.npz"


def _simulate(i):
    truth = people()[i]
    acc.clear_caches()
    with Quiet():
        data = run_app(truth["f"], truth["k"], truth["g"], seed=SEED + i, n=D.decisions,
                       person=PersonConstants(acc.SIGMA, truth["bound"]), grid=G_FINE,
                       t0=truth["t0"])
    node = np.array([p.get("node") == "stage" for p in data], dtype=np.int8)
    go = np.array([-1 if p["GO"] is None else p["GO"] for p in data], dtype=np.int8)
    rt = np.array([np.nan if p["RT"] is None else p["RT"] for p in data])
    parts = np.array([[p["parts_by_g"][g] for g in G_FINE] for p in data])  # (n, 50, 2)
    np.savez_compressed(history_path(i), node=node, go=go, rt=rt,
                        cost=np.array([p["cost"] for p in data]),
                        value_rec=np.array([p["value"] for p in data]),
                        feedback=np.array([p.get("feedback", 1) for p in data], dtype=np.int8),
                        base=parts[:, :, 0], delta=parts[:, :, 1], g_grid=np.array(G_FINE),
                        **{f"true_{k}": v for k, v in truth.items()})
    return dict(person=i, decisions=len(data), redos=int(node.sum()),
                undecided=int((go == -1).sum()), **truth)


# ---------------------------------------------------------------- stage 2: fit

NAMES = ("f", "k", "g", "bound", "t0")
BOUNDS = [(0.0, 3.0), (0.0, 3.0), (0.5, 0.99), (0.5, 3.0), T0_BOUNDS]
STEPS = np.array([0.01, 0.01, 0.005, 0.01, 0.005])  # finite-difference steps
APP = {"g": 0.9, "bound": acc.BOUND}  # what the current pipeline pins


class Person:
    """One saved history: its decisions, and value as a smooth function of γ."""

    # `h`: the history's arrays, as saved by _simulate; read from its file when None.
    def __init__(self, i, h=None):
        h = np.load(history_path(i)) if h is None else h
        self.i = i
        self.go, self.rt, self.cost, self.node = h["go"], h["rt"], h["cost"], h["node"]
        self.value = CubicSpline(h["g_grid"], (h["base"] + h["delta"]).T, axis=0)
        self.truth = np.array([float(h[f"true_{n}"]) for n in NAMES])

    def mu(self, th):
        return th[0] * self.value(th[2]) - th[1] * self.cost

    def nll(self, th):
        return fastlik.neg_log_likelihood(self.go, self.rt, self.mu(th), th[3], th[4])

    def fit(self, free=(0, 1, 2, 3, 4), fixed=None):
        th = np.array(fixed if fixed is not None else self.truth, float)
        free = list(free)

        def loss(x):
            th[free] = x
            return self.nll(th)
        res = differential_evolution(loss, [BOUNDS[j] for j in free], seed=self.i, tol=1e-8,
                                     maxiter=2000, polish=False)
        res = minimize(loss, res.x, method="L-BFGS-B", bounds=[BOUNDS[j] for j in free])
        th[free] = res.x
        return th.copy(), float(res.fun)

    # Central-difference Hessian of -log L.
    def hessian(self, th, steps=STEPS):
        n = len(th)
        H = np.zeros((n, n))
        f0 = self.nll(th)
        e = np.eye(n) * steps
        for i in range(n):
            H[i, i] = (self.nll(th + e[i]) - 2 * f0 + self.nll(th - e[i])) / steps[i] ** 2
            for j in range(i):
                H[i, j] = H[j, i] = (self.nll(th + e[i] + e[j]) - self.nll(th + e[i] - e[j])
                                     - self.nll(th - e[i] + e[j]) + self.nll(th - e[i] - e[j])
                                     ) / (4 * steps[i] * steps[j])
        return H

    # Expected information: Σ over decisions and outcomes of ∂p ∂pᵀ / p, outcomes being
    # choice × RT bin (0.01 s) and undecided, derivatives by central differences.
    def fisher(self, th, steps=STEPS):
        def probs(x):
            up, lo, und = fastlik.outcome_probs(self.mu(x), x[3], x[4])
            return np.concatenate([up, lo, und[:, None]], axis=1)
        w = 1.0 / np.maximum(probs(th), 1e-300)
        e = np.eye(len(th)) * steps
        G = np.stack([(probs(th + e[i]) - probs(th - e[i])) / (2 * steps[i])
                      for i in range(len(th))])
        return np.einsum("ino,jno,no->ij", G, G, w)


def standard_errors(M):
    try:
        cov = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.full(len(M), np.nan), np.nan
    d = np.diag(cov)
    if np.any(d <= 0):
        return np.full(len(M), np.nan), np.nan
    return np.sqrt(d), cov[0, 2] / np.sqrt(d[0] * d[2])  # SEs, and corr(f, γ)


def _fit(i):
    p = Person(i)
    est, nll = p.fit()
    se_obs, corr_obs = standard_errors(p.hessian(est))
    se_half, _ = standard_errors(p.hessian(est, STEPS / 2))
    se_fis, corr_fis = standard_errors(p.fisher(p.truth))
    # The current pipeline: γ and BOUND pinned, f, k and T0 fitted.
    pinned = p.truth.copy()
    pinned[2], pinned[3] = APP["g"], APP["bound"]
    app, nll_app = p.fit(free=(0, 1, 4), fixed=pinned)
    row = dict(person=i, decisions=len(p.go), redos=int(p.node.sum()),
               undecided=int((p.go < 0).sum()), nll=nll, nll_app=nll_app,
               corr_obs=corr_obs, corr_fisher=corr_fis)
    for j, n in enumerate(NAMES):
        lo, hi = BOUNDS[j]
        row.update({f"true_{n}": p.truth[j], f"est_{n}": est[j], f"se_obs_{n}": se_obs[j],
                    f"se_half_{n}": se_half[j], f"se_fisher_{n}": se_fis[j],
                    f"at_bound_{n}": bool(min(est[j] - lo, hi - est[j]) < 2 * STEPS[j]),
                    f"app_{n}": app[j]})
    return row


def summarize(df):
    print(f"\n== recovery of f, k, γ, BOUND, T0: {len(df)} people, "
          f"{df.decisions.mean():.0f} decisions each ({df.redos.mean():.0f} redos) ==")
    print(f"{'':>6} {'RMSE':>7} {'bias':>16} {'r':>6}   {'SD z obs':>8} {'cover obs':>9} "
          f"{'SD z Fisher':>11} {'cover Fisher':>12} {'n':>4}")
    for n in NAMES:
        err = df[f"est_{n}"] - df[f"true_{n}"]
        r = np.corrcoef(df[f"est_{n}"], df[f"true_{n}"])[0, 1]
        ok = ~df[[f"at_bound_{m}" for m in NAMES]].any(axis=1) & df[f"se_obs_{n}"].notna()
        z_obs = (err / df[f"se_obs_{n}"])[ok]
        z_fis = (err / df[f"se_fisher_{n}"])[ok]
        print(f"{n:>6} {np.sqrt(np.mean(err ** 2)):7.3f} {err.mean():+8.3f} ± {se(err):.3f} {r:6.3f}   "
              f"{z_obs.std():8.2f} {(z_obs.abs() < 1.96).mean():9.0%} "
              f"{z_fis.std():11.2f} {(z_fis.abs() < 1.96).mean():12.0%} {ok.sum():4d}")
    print("L2/L3 pass if every SD z is in 0.8 to 1.25 and every coverage in 90% to 98%.")
    for col, name in (("corr_fisher", "Fisher, at the truth"), ("corr_obs", "observed, at the fit")):
        c = df[col].dropna()
        print(f"corr(f, γ), {name}: median {c.median():+.2f}, "
              f"5th to 95th percentile {c.quantile(0.05):+.2f} to {c.quantile(0.95):+.2f}")
    stab = (df.se_half_g / df.se_obs_g).dropna()
    print(f"SE of γ with half the finite-difference step / with the step: median {stab.median():.3f}")
    print("\nthe current pipeline (γ 0.9 and BOUND 1.5 pinned; f, k, T0 fitted) on the same data:")
    for n in ("f", "k", "t0"):
        e_full = df[f"est_{n}"] - df[f"true_{n}"]
        e_app = df[f"app_{n}"] - df[f"true_{n}"]
        print(f"{n:>6} RMSE {np.sqrt(np.mean(e_app ** 2)):.3f} (all free: "
              f"{np.sqrt(np.mean(e_full ** 2)):.3f}), bias {e_app.mean():+.3f} ± {se(e_app):.3f}")


def span(s):
    lo, hi = (int(x) for x in s.split("-"))
    return list(range(lo, hi + 1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", choices=("simulate", "fit"), default=None)
    p.add_argument("--replot", action="store_true", help="summarize the saved fits")
    p.add_argument("--people", type=span, default=list(range(N_PEOPLE)), help="for example 0-49")
    p.add_argument("--tag", default="local", help="names this machine's file")
    _design_args(p)
    a = p.parse_args()
    if a.stage == "simulate":
        save_conditions(D)
        todo = [i for i in a.people if not history_path(i).exists()]
        run_pool(_simulate, todo, OUT / f"simulated_{a.tag}.csv", fresh=True,
                 label=lambda r: f"person {r['person']}: {r['decisions']} decisions")
    elif a.stage == "fit":
        todo = [i for i in a.people if history_path(i).exists()]
        df = run_pool(_fit, todo, OUT / "fits.csv",
                      label=lambda r: f"person {r['person']}")
        df.sort_values("person").to_csv(OUT / "fits.csv", index=False)
    if a.stage == "fit" or a.replot:
        summarize(pd.read_csv(OUT / "fits.csv"))
