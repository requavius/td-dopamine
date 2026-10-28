"""Intervals for two saved results, from their runs. No simulation.

Both analyses were added on 2026-10-28, after the results were known. They are
descriptive: they put an interval on an effect, and do not replace the claims each
experiment fixed before its run (the 2·SE rules in experiments.controllers and
experiments.task_choice).

1. Controllers (results/controllers/): the same 32 people and seeds
   under every controller. The information controller minus each other controller, for
   f RMSE and k RMSE at 100 and 300 ticks, total reward and P(start). A bootstrap
   resamples people with replacement, keeping each person's controllers together, and
   recomputes each RMSE on the resample; 10,000 resamples, 95% percentile intervals.
   γ was known and correct (0.9) for every person and every fit.

2. Feedback (results/task_choice/): five people, six targets,
   feedback shown and hidden on the same 16 seeds. Shown minus hidden in P(start):
   - per person: the mean over targets and seeds, with an interval over the 16 seeds
     (each seed averaged over the six targets first), t with 15 df;
   - across the five people: the mean of the five per-person effects, t with 4 df. Five
     values are too few for a bootstrap (126 distinct resamples);
   - for reference, a one-sample t-test on the 30 person × target means, which treats
     cells of one person as independent.

    python -m experiments.intervals
"""
import numpy as np
import pandas as pd
from scipy import stats

from .common import RESULTS, se

B = 10_000
CONTROLLERS = ("fixed", "random", "reward_scheduled", "reward")
NAMES = {"fixed": "default setting", "random": "random", "reward_scheduled": "reward",
         "reward": "reward (redo)", "information": "information (redo)"}


def controllers():
    df = pd.read_csv(RESULTS / "controllers" / "runs.csv")
    key = ["f_true", "k_true", "seed"]
    wide = {arm: g.sort_values(key).reset_index(drop=True) for arm, g in df.groupby("arm")}
    people = wide["information"][key]
    for g in wide.values():  # every controller has the same people in the same order
        assert g[key].equals(people)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(people), size=(B, len(people)))

    def rmse(arm, p, n, rows=None):
        d = wide[arm]
        err = (d[f"{p}_{n}"] - d[f"{p}_true"]).to_numpy() ** 2
        return np.sqrt(err.mean()) if rows is None else np.sqrt(err[rows].mean(axis=1))

    def mean(arm, col, rows=None):
        x = wide[arm][col].to_numpy()
        return x.mean() if rows is None else x[rows].mean(axis=1)

    print(f"== information controller minus each other, {len(people)} paired people, "
          f"{B:,} bootstrap resamples, 95% intervals ==")
    measures = [(f"{p} RMSE, {n} ticks", lambda arm, rows=None, p=p, n=n: rmse(arm, p, n, rows))
                for p in ("k", "f") for n in (100, 300)]
    measures += [("total reward", lambda arm, rows=None: mean(arm, "reward", rows)),
                 ("P(start)", lambda arm, rows=None: mean(arm, "p_go", rows))]
    for label, fn in measures:
        print(f"\n{label}: information {fn('information'):.3f}")
        for other in CONTROLLERS:
            d = fn("information") - fn(other)
            boot = fn("information", idx) - fn(other, idx)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  − {NAMES[other]:<16} {d:+8.3f}  [{lo:+.3f}, {hi:+.3f}]")


def feedback():
    df = pd.read_csv(RESULTS / "task_choice" / "runs.csv")
    shown = df[df.part == "a"].set_index(["f", "k", "target", "seed"]).p_go
    hidden = df[df.part == "a_hidden"].set_index(["f", "k", "target", "seed"]).p_go
    diff = (shown - hidden).dropna().rename("d").reset_index()

    print("\n== feedback shown − hidden, P(start), five people × six targets × 16 paired seeds ==")
    people = []
    for (f, k), g in diff.groupby(["f", "k"]):
        by_seed = g.groupby("seed").d.mean()  # each seed averaged over targets
        m, e = by_seed.mean(), se(by_seed)
        t = stats.t.ppf(0.975, len(by_seed) - 1)
        people.append(m)
        print(f"  f {f}, k {k}: {m:+.4f}  [{m - t * e:+.4f}, {m + t * e:+.4f}] over {len(by_seed)} seeds")
    people = np.array(people)
    m, e = people.mean(), se(people)
    t = stats.t.ppf(0.975, len(people) - 1)
    print(f"across the five people: {m:+.4f}  [{m - t * e:+.4f}, {m + t * e:+.4f}] (t, 4 df), "
          f"{(people > 0).sum()} of 5 above zero")
    cells = diff.groupby(["f", "k", "target"]).d.mean()
    res = stats.ttest_1samp(cells, 0.0)
    print(f"for reference, a one-sample test on the 30 cells: {cells.mean():+.4f}, "
          f"t = {res.statistic:.2f}, p = {res.pvalue:.1e} (cells of one person are not "
          f"independent, so this p is too small)")


if __name__ == "__main__":
    controllers()
    feedback()
