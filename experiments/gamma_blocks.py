"""Does switching feedback in long blocks pin γ down? Run if gamma_profile fails.

Hiding feedback moves a set's reward to its end without changing how much it pays, so
whether a person starts less after hidden sets than after shown ones depends on γ, and f
cancels out of the comparison. Under InfoController feedback switches only when it
serves f and k. Here it switches on a schedule.

Two arms, paired on person and seed, no controller, one fixed setting
(gamma_profile.BLOCK_ENV: aim 60% right, no scheduled review, redo prompts):
    blocks   feedback on for 25 start decisions, off for 25, and so on (6 blocks of each)
    always   feedback on throughout
With feedback off there are no redo prompts, so the blocks arm has about half the redo
decisions of the always arm. The comparison is the net of the two: what a schedule a
person could actually be given buys.

Everything else is gamma_profile's: the app's task, 300 start decisions plus their redo
decisions, people (f, k) (0.8, 0.8), (1.2, 0.4), (0.6, 1.2), true γ 0.7, 0.8, 0.9, 0.95,
the same grid and fit. Constants: α 0.05, BOUND 1.5, SIGMA 0.7, ACHIEVEMENT_SCALE 4,
continuing bootstrap, T_DUR 7.0, T0 0. The simulated person is paid the end-of-set reward
with feedback hidden, as the app's score has shown it since 2026-10-28.

Sharpness of a run is how much -log L rises from the true γ to the best grid γ more than
0.1 away from it: min over |γ − γ_true| > 0.1 of nll(γ) − nll(γ_true).

Claims, fixed before any run (2026-10-28):
    B1  blocks arm: gamma_profile's P1 and P2, each in at least 80% of runs
    B2  sharpness is higher in the blocks arm than the always arm, paired by person and
        seed, beyond 2 SE

    python -m experiments.gamma_blocks --gammas 0.7,0.8 --reps 0,1 --tag server
    python -m experiments.gamma_blocks --replot
"""
import argparse

import pandas as pd

from temporal.model import accumulator as acc

from .common import Quiet, out_dir, run_pool, se
from .gamma_profile import N_APP, PEOPLE, TRUE_G, floats, per_run, profile, run_app, seed_of

OUT = out_dir("gamma_blocks")
BLOCK = 25
ARMS = {"blocks": BLOCK, "always": N_APP + 1}  # a block longer than the run never switches
RUN = ["arm", "f_true", "k_true", "g_true", "rep"]


def _one(task):
    arm, f, k, g_true, rep = task
    seed = seed_of(f, k, g_true, rep) + 7000  # the same in both arms
    acc.clear_caches()
    with Quiet():
        rows = profile(run_app(f, k, g_true, seed, blocks=ARMS[arm]))
    return [dict(arm=arm, f_true=f, k_true=k, g_true=g_true, rep=rep, seed=seed, **r)
            for r in rows]


def sharpness(df):
    out = []
    for key, r in df.groupby(RUN):
        info = dict(zip(RUN, key))
        at_true = r[r.g == info["g_true"]].nll.iloc[0]
        far = r[(r.g - info["g_true"]).abs() > 0.1 + 1e-9]
        out.append(dict(info, sharpness=(far.nll - at_true).min()))
    return pd.DataFrame(out)


def summarize(df):
    runs, grid = per_run(df, RUN)
    print(f"\n== γ with feedback in blocks of {BLOCK}: {len(runs)} runs, grid {grid} ==")
    for arm, r in runs.groupby("arm"):
        print(f"{arm:>6}: P1 {r.p1.mean():.0%}, P2 {r.p2.mean():.0%} of {len(r)} runs; "
              f"feedback on {r.feedback_share.mean():.0%} of starts, {r.redos.mean():.0f} redos; "
              f"f bias at γ̂ {r.f_bias.mean():+.0f} ± {se(r.f_bias):.0f}%")
    b = runs[runs.arm == "blocks"]
    print(f"B1 {'holds' if b.p1.mean() >= 0.8 and b.p2.mean() >= 0.8 else 'does not hold'}")

    s = sharpness(df).pivot_table(index=RUN[1:], columns="arm", values="sharpness").dropna()
    diff = s["blocks"] - s["always"]
    print(f"\nsharpness (Δ -log L to the best γ more than 0.1 from the truth), {len(s)} pairs:")
    print(s.groupby("g_true").mean().round(1).to_string())
    print(f"blocks − always: {diff.mean():+.2f} ± {se(diff):.2f}")
    print(f"B2 {'holds' if diff.mean() > 2 * se(diff) else 'does not hold'}")
    return runs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="merge and summarize the saved results")
    p.add_argument("--gammas", type=floats, default=TRUE_G, help="true γ values to run")
    p.add_argument("--reps", type=lambda s: tuple(int(x) for x in s.split(",")), default=(0, 1))
    p.add_argument("--tag", default="local", help="names this machine's file, runs_<tag>.csv")
    a = p.parse_args()
    if not a.replot:
        tasks = [(arm, f, k, g, rep) for arm in ARMS for g in a.gammas for f, k in PEOPLE
                 for rep in a.reps]
        run_pool(_one, tasks, OUT / f"runs_{a.tag}.csv", fresh=True,
                 label=lambda rows: f"{rows[0]['arm']} γ {rows[0]['g_true']}")
    files = sorted(OUT.glob("runs_*.csv"))
    summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
