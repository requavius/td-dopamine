"""Does showing feedback protect effort-averse people from lockout?

Lockout: a run in which the person almost never starts again, defined before the run as
P(start next set) < 0.10 over the start decisions recorded in the run (300 ticks). The
task of task_choice (question-type task, skips on, one setting held for each run, review
0.25, skipped questions dropped, no controller). Current model (V4): gamma 0.9,
ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s. Four effort-averse people, targets 75, 60 and 45%,
feedback shown and hidden, 64 seeds paired across feedback (4000-4063).

Claims, fixed before the run (paired on person, target and seed):
  L1  Pooled over people and targets, lockout is more common with feedback hidden: the mean
      paired difference in the lockout indicator (hidden - shown) is above zero by more
      than 2 SE.
  L2  It holds for each person separately if that person's pooled difference is above
      zero by more than 2 SE.
Reported without a claim: lockouts per person and target, and the start rate in runs that
did not lock out.

README: the "Lockout" bullet.

    python -m experiments.lockout            # run, then summarize
    python -m experiments.lockout --replot   # summarize the saved results
"""
import pandas as pd

from .common import args, out_dir, run_pool, se
from .task_choice import one_run

OUT = out_dir("lockout")
PEOPLE = ((0.6, 1.0), (0.6, 1.2), (1.0, 1.2), (1.0, 1.5))
TARGETS = (0.75, 0.6, 0.45)
SEEDS = range(4000, 4064)
LOCKOUT = 0.10


def summarize(df):
    df = df.assign(locked=(df.p_go < LOCKOUT).astype(int))
    w = df.pivot_table(index=["f", "k", "target", "seed"], columns="feedback", values="locked")
    d = w[False] - w[True]
    print(f"== L1: lockout, hidden - shown, pooled: {d.mean():+.3f} ± {se(d):.3f} "
          f"({d.mean() / se(d):.1f} SE) -> {'HOLDS' if d.mean() > 2 * se(d) else 'does not hold'}")
    print(f"   lockout rate: shown {w[True].mean():.1%}, hidden {w[False].mean():.1%} "
          f"({int(w[True].sum())} and {int(w[False].sum())} of {len(w)} runs each)")
    print("\n== L2: per person ==")
    for (f, k), g in d.groupby(level=["f", "k"]):
        ww = w.xs((f, k), level=["f", "k"])
        print(f"f {f} k {k}: hidden - shown {g.mean():+.3f} ± {se(g):.3f} "
              f"-> {'holds' if g.mean() > 2 * se(g) else 'does not hold'}  "
              f"(lockouts: shown {int(ww[True].sum())}, hidden {int(ww[False].sum())} of {len(ww)})")
    print("\n== lockouts of 64 per person and target, shown / hidden (no claim) ==")
    for (f, k, t), g in w.groupby(level=["f", "k", "target"]):
        print(f"f {f} k {k} target {t:.0%}: {int(g[True].sum())} / {int(g[False].sum())}")
    print("\n== start rate in runs that did not lock out, shown - hidden (no claim) ==")
    alive = df[df.locked == 0].groupby(["f", "k", "feedback"]).p_go.mean().unstack()
    print((alive[True] - alive[False]).round(3).to_string())
    if "away" in df:
        print("\n== share of ticks spent away, and start decisions recorded per run (no claim) ==")
        print(df.groupby(["f", "k", "feedback"])[["away", "decisions"]].mean().round(2).to_string())
        w = df.pivot_table(index=["f", "k", "target", "seed"], columns="feedback", values="away")
        d = w[False] - w[True]
        print(f"away, hidden - shown, pooled: {d.mean():+.3f} ± {se(d):.3f}")


if __name__ == "__main__":
    a = args(__doc__)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        tasks = [("lockout", f, k, t, fb, s) for f, k in PEOPLE for t in TARGETS
                 for fb in (True, False) for s in SEEDS]
        df = run_pool(one_run, tasks, OUT / "runs.csv", chunksize=16)
    summarize(df)
