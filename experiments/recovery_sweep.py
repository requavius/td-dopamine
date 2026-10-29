"""Recovery from one person, and what dropping timed-out trials does to it.

One simulated person per dataset, mid cold-start cost, scheduled review, 1200 recorded
decisions split across four learners after 0, 200, 500 and 1000 warm-up decisions
(`recovery.collect_reseeded`), single-diff task, 4 x 4 grid x 3 reps (48 fits), seeded per
grid point and rep. Each dataset is fitted twice: timed-out trials kept as undecided (the
pipeline) and dropped. Current model (V4): gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap,
T_DUR 7.0 s, T0 pinned at 0.

README: "Recovery from one person" and "Censored trials are data". Paper Figures 4 and S2.

    python -m experiments.recovery_sweep            # run, then summarize
    python -m experiments.recovery_sweep --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from temporal.inference import fit
from temporal.recovery import RESEED_WARMUPS, collect_reseeded, seed_point

from .common import LEVELS, Quiet, args, design, out_dir, run_pool

D = design("recovery_sweep", levels=LEVELS, reps=3, decisions=1200, warmups=RESEED_WARMUPS)
OUT = out_dir(D.out)
REPS = D.reps


def _one(task):
    f_true, k_true, rep = task
    seed = seed_point(f_true, k_true, rep)
    with Quiet():
        df = collect_reseeded(f_true, k_true, D.decisions, D.warmups)
        f_kept, k_kept = fit(df)
        dropped = df.copy()
        dropped.attrs["undecided"] = {}
        f_drop, k_drop = fit(dropped)
    return dict(f_true=f_true, k_true=k_true, rep=rep, seed=seed, n_decided=len(df),
                n_undecided=sum(df.attrs["undecided"].values()),
                f_kept=f_kept, k_kept=k_kept, f_drop=f_drop, k_drop=k_drop)


def rmse(est, truth):
    return float(np.sqrt(np.mean((np.asarray(est) - np.asarray(truth)) ** 2)))


# 95% percentile bootstrap over the fits.
def rmse_interval(est, truth, n=10_000, seed=0):
    err = np.asarray(est) - np.asarray(truth)
    rng = np.random.default_rng(seed)
    boots = [np.sqrt(np.mean(rng.choice(err, len(err)) ** 2)) for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])


def summarize(res):
    print(f"== recovery from one person ({len(res)} fits, timed-out trials kept) ==")
    for p in ("f", "k"):
        lo, hi = rmse_interval(res[f"{p}_kept"], res[f"{p}_true"])
        print(f"{p} RMSE {rmse(res[f'{p}_kept'], res[f'{p}_true']):.3f} [{lo:.3f}, {hi:.3f}]")
    print("\nworst grid points:")
    for p in ("f", "k"):
        sq = (res[f"{p}_kept"] - res[f"{p}_true"]) ** 2
        g = np.sqrt(sq.groupby([res.f_true, res.k_true]).mean())
        top = g.sort_values(ascending=False).head(3)
        print(f"  {p}: " + ", ".join(f"f {f}, k {k}: {v:.3f}" for (f, k), v in top.items()))

    total = res.n_undecided.sum() + res.n_decided.sum()
    print(f"\n== censored trials: {res.n_undecided.sum():,} of {total:,} decisions "
          f"({res.n_undecided.sum() / total:.1%}) timed out ==")
    for col, name in (("kept", "kept as undecided"), ("drop", "dropped")):
        parts = []
        for p in ("f", "k"):
            est, truth = res[f"{p}_{col}"], res[f"{p}_true"]
            parts.append(f"{p} bias {100 * ((est - truth) / truth).mean():+.1f}% "
                         f"RMSE {rmse(est, truth):.3f}")
        print(f"{name:>18}: " + "  ".join(parts))


if __name__ == "__main__":
    a = args(__doc__, D)
    if a.replot:
        res = pd.read_csv(OUT / "runs.csv")
    else:
        tasks = [(f, k, r) for f in D.levels for k in D.levels for r in range(REPS)]
        res = run_pool(_one, tasks, OUT / "runs.csv",
                       label=lambda r: f"f {r['f_true']} k {r['k_true']} rep {r['rep']}")
        res = res.sort_values(["f_true", "k_true", "rep"])
        res.to_csv(OUT / "runs.csv", index=False)
    summarize(res)
