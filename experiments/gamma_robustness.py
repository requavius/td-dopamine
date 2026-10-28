"""Does the γ estimate survive the other constants being wrong? Run if gamma_profile passes.

experiments.gamma_profile recovers γ from a person who shares every other constant with
the fit. Here each person differs from the fit in one more thing, and the fit isn't told:

    a        the TD learning rate: 0.01, 0.025, 0.1, 0.2 (the fit's learners use 0.05)
    bound    the accumulator's BOUND: 1.0, 2.0 (the fit uses 1.5)
    reward   what a set is worth to them: the fraction right ("accuracy"), where the fit
             assumes right answers weighted by 1 − p ("achievement")

Everything else is gamma_profile's: the app's task, InfoController, 300 start decisions
plus their redo decisions, people (f, k) (0.8, 0.8), (1.2, 0.4), (0.6, 1.2), the same
grid, (f, k) fitted at each grid γ, the lowest -log L the estimate. True γ 0.8 and 0.95,
either side of the app's 0.9. Constants of the fit: γ grid, α 0.05, BOUND 1.5, SIGMA 0.7,
ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0, T0 0. Seeds are gamma_profile's
plus 1 000 per level, so no run repeats one of its.

Claims, fixed before any run (2026-10-28):
    R1  at every level of every factor, the estimate is the true γ or its grid neighbour
        in at least 80% of runs (gamma_profile's P1, per level)
    R2  none: report the bias in f and k at the estimate, against at γ = 0.9

A level that fails R1 says that constant can pass for a different γ, and γ cannot be
reported for a person without measuring it.

    python -m experiments.gamma_robustness --factors a --reps 0,1 --tag server
    python -m experiments.gamma_robustness --replot
"""
import argparse

import pandas as pd

from temporal.model import accumulator as acc

from .common import Quiet, out_dir, run_pool, se
from .gamma_profile import PEOPLE, floats, per_run, profile, run_app, seed_of
from .misspecification import PersonConstants

OUT = out_dir("gamma_robustness")
TRUE_G = (0.8, 0.95)
LEVELS = {"a": (0.01, 0.025, 0.1, 0.2), "bound": (1.0, 2.0), "reward": ("accuracy",)}
RUN = ["factor", "level", "f_true", "k_true", "g_true", "rep"]


def _one(task):
    factor, level, f, k, g_true, rep = task
    index = [str(x) for x in LEVELS[factor]].index(str(level))
    seed = seed_of(f, k, g_true, rep) + 1000 * (1 + index + 10 * list(LEVELS).index(factor))
    kw = {"a": {"a": level}, "bound": {"person": PersonConstants(acc.SIGMA, level)},
          "reward": {"reward": level}}[factor]
    acc.clear_caches()
    with Quiet():
        rows = profile(run_app(f, k, g_true, seed, **kw))
    return [dict(factor=factor, level=str(level), f_true=f, k_true=k, g_true=g_true, rep=rep,
                 seed=seed, **r) for r in rows]


def summarize(df):
    runs, grid = per_run(df, RUN)
    print(f"\n== γ under a wrong constant: {len(runs)} runs, grid {grid} ==")
    print(f"{'factor':>6} {'level':>8} {'true γ':>6} {'runs':>4} {'R1':>5} {'γ̂ (each run)':>28} "
          f"{'f bias at γ̂':>14} {'at 0.9':>12} {'k bias at γ̂':>14}")
    for (factor, level, g_true), r in runs.groupby(["factor", "level", "g_true"]):
        hats = ",".join(f"{x:g}" for x in r.g_hat)
        print(f"{factor:>6} {level:>8} {g_true:>6g} {len(r):>4} {r.p1.mean():>5.0%} {hats:>28} "
              f"{r.f_bias.mean():+7.0f} ± {se(r.f_bias):2.0f}% {r.f_bias_app.mean():+5.0f} ± "
              f"{se(r.f_bias_app):2.0f}% {r.k_bias.mean():+7.0f} ± {se(r.k_bias):2.0f}%")
    print("\nR1 by level (claim: ≥ 80% at every level):")
    by = runs.groupby(["factor", "level"]).p1.mean()
    print(by.map("{:.0%}".format).to_string())
    print(f"R1 {'holds' if (by >= 0.8).all() else 'does not hold'}")
    return runs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="merge and summarize the saved results")
    p.add_argument("--factors", default=",".join(LEVELS), help="which of a, bound, reward")
    p.add_argument("--gammas", type=floats, default=TRUE_G, help="true γ values to run")
    p.add_argument("--reps", type=lambda s: tuple(int(x) for x in s.split(",")), default=(0, 1))
    p.add_argument("--tag", default="local", help="names this machine's file, runs_<tag>.csv")
    a = p.parse_args()
    if not a.replot:
        tasks = [(fa, lv, f, k, g, rep) for fa in a.factors.split(",") for lv in LEVELS[fa]
                 for g in a.gammas for f, k in PEOPLE for rep in a.reps]
        run_pool(_one, tasks, OUT / f"runs_{a.tag}.csv", fresh=True,
                 label=lambda rows: f"{rows[0]['factor']} {rows[0]['level']} γ {rows[0]['g_true']}")
    files = sorted(OUT.glob("runs_*.csv"))
    summarize(pd.concat([pd.read_csv(f, dtype={"level": str}) for f in files], ignore_index=True))
