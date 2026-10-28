"""Where does the information about γ come from? A diagnostic for gamma_profile.

A wrong γ mostly rescales value, and a rescaling is absorbed by f. What can identify γ is
the part of how value changes with γ that is not a rescaling. Value at a start is
V(0) + Σδ (the learned value plus the error carried out of the last set); at a redo it is
the slot's share (1/4)·γ^remaining plus the error carried out of the slot. The hypothesis
from 2026-10-28: γ is identified mainly because it changes V(0) far more than the carried
error, so it sets the balance between the two, and f cannot absorb a balance.

The same people and seeds as gamma_profile (so the data are its runs, decision for
decision), and two measurements per run:

1. Profiles from parts of the data: (f, k) fitted at each grid γ from start decisions
   alone and from redo decisions alone, beside gamma_profile's fit to both.
2. Information about γ with f and k profiled out, at the truth. A decision's drift is
   f·v − k·c, so the fit can absorb any change in value along v or c. What is left of
   dv/dγ after a weighted projection onto (v, c), with weights I_μ (the Fisher
   information of one decision about its drift, controller.i_mu_table), is what
   identifies γ: J = f² Σ I_μ r². J is computed for dv/dγ and for each part alone
   (V(0), carried error, slot share). dv/dγ is a central difference between the true
   γ's grid neighbours, on the unsnapped parts. ½·J·h² predicts the rise in -log L a step
   h away, reported against the profile's own rise as a check of the approximation.

Constants: α 0.05, BOUND 1.5, SIGMA 0.7, ACHIEVEMENT_SCALE 4, continuing bootstrap,
T_DUR 7.0, T0 0.

Claims, fixed before any run (2026-10-28):
    S1  J from the carried error alone exceeds J from V(0) alone in at least 80% of runs
    S2  none: report P1 and P2 of the start-only and redo-only profiles

    python -m experiments.gamma_sources --gammas 0.8,0.95 --reps 0 --tag server
    python -m experiments.gamma_sources --replot
"""
import argparse

import numpy as np
import pandas as pd

from temporal.controller import MU_GRID, i_mu_table
from temporal.model import accumulator as acc

from .common import Quiet, out_dir, run_pool, se
from .gamma_profile import GRID, PEOPLE, TRUE_G, floats, per_run, profile, run_app, seed_of

OUT = out_dir("gamma_sources")
RUN = ["part", "f_true", "k_true", "g_true", "rep"]


# f² Σ w r², r what is left of y after the w-weighted projection onto the columns of X.
def profiled_information(y, X, w, f):
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    r = y - X @ beta
    return float(f * f * np.sum(w * r * r))


def information(data, f, k, g_true):
    i = GRID.index(g_true)
    lo, hi = GRID[max(i - 1, 0)], GRID[min(i + 1, len(GRID) - 1)]
    rows = []
    for p in data:
        a, d = p["parts_by_g"][g_true]
        (a_lo, d_lo), (a_hi, d_hi) = p["parts_by_g"][lo], p["parts_by_g"][hi]
        start = p.get("node") != "stage"
        da, dd = (a_hi - a_lo) / (hi - lo), (d_hi - d_lo) / (hi - lo)
        rows.append((a + d, p["cost"], da if start else 0.0, dd, 0.0 if start else da))
    v, c, d_v0, d_delta, d_slot = np.array(rows).T
    w = np.interp(f * v - k * c, MU_GRID, i_mu_table())
    X = np.column_stack([v, c])
    return {"J": profiled_information(d_v0 + d_delta + d_slot, X, w, f),
            "J_v0": profiled_information(d_v0, X, w, f),
            "J_delta": profiled_information(d_delta, X, w, f),
            "J_slot": profiled_information(d_slot, X, w, f), "h": hi - lo}


def _one(task):
    f, k, g_true, rep = task
    seed = seed_of(f, k, g_true, rep)
    acc.clear_caches()
    with Quiet():
        data = run_app(f, k, g_true, seed)
        starts = [p for p in data if p.get("node") != "stage"]
        redos = [p for p in data if p.get("node") == "stage"]
        info = information(data, f, k, g_true)
        out = []
        for part, sub in (("both", data), ("starts", starts), ("redos", redos)):
            out += [dict(part=part, f_true=f, k_true=k, g_true=g_true, rep=rep, seed=seed,
                         **info, **r) for r in profile(sub)]
    return out


def summarize(df):
    runs, grid = per_run(df, RUN)
    print(f"\n== where γ is identified: {runs[runs.part == 'both'].shape[0]} runs ==")
    print("profiles from parts of the data:")
    for part, r in runs.groupby("part"):
        print(f"  {part:>6}: P1 {r.p1.mean():.0%}, P2 {r.p2.mean():.0%} of {len(r)} runs")

    info = df[df.part == "both"].drop_duplicates(RUN[1:])
    share = info[["J_v0", "J_delta", "J_slot"]].div(info.J_v0 + info.J_delta + info.J_slot, axis=0)
    print("\nprofiled information about γ at the truth, each part alone, as a share of their sum:")
    print(pd.concat([info[RUN[1:]], share.round(2)], axis=1).groupby("g_true")
          [["J_v0", "J_delta", "J_slot"]].mean().round(2).to_string())
    s1 = (info.J_delta > info.J_v0).mean()
    print(f"S1 (carried error > V(0) in ≥ 80% of runs): {s1:.0%}, "
          f"{'holds' if s1 >= 0.8 else 'does not hold'}")

    # The approximation, checked: ½·J·h² against the profile's rise one grid step away.
    both = df[df.part == "both"]
    rise = []
    for key, r in both.groupby(RUN[1:]):
        g_true = key[2]
        i = grid.index(g_true)
        at = r.set_index("g").nll
        steps = [grid[j] for j in (i - 1, i + 1) if 0 <= j < len(grid)]
        observed = np.mean([at[g] - at[g_true] for g in steps])
        h = np.mean([abs(g - g_true) for g in steps])
        rise.append(dict(observed=observed, predicted=0.5 * r.J.iloc[0] * h * h))
    rise = pd.DataFrame(rise)
    print(f"\nrise in -log L one grid step from the truth: observed {rise.observed.mean():.2f} ± "
          f"{se(rise.observed):.2f}, predicted from J {rise.predicted.mean():.2f} ± "
          f"{se(rise.predicted):.2f}")
    return runs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="merge and summarize the saved results")
    p.add_argument("--gammas", type=floats, default=TRUE_G, help="true γ values to run")
    p.add_argument("--reps", type=lambda s: tuple(int(x) for x in s.split(",")), default=(0,))
    p.add_argument("--tag", default="local", help="names this machine's file, runs_<tag>.csv")
    a = p.parse_args()
    if not a.replot:
        tasks = [(f, k, g, rep) for g in a.gammas for f, k in PEOPLE for rep in a.reps]
        run_pool(_one, tasks, OUT / f"runs_{a.tag}.csv", fresh=True,
                 label=lambda rows: f"f {rows[0]['f_true']} k {rows[0]['k_true']} γ {rows[0]['g_true']}")
    files = sorted(OUT.glob("runs_*.csv"))
    summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
