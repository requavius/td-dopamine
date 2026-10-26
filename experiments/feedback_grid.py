"""Does showing feedback raise engagement for everyone, and how do f and k shape it?

The task of task_choice (question-type task, skips on, one setting held for each run of 300
ticks, review 0.25, skipped questions dropped, no controller) for the 16 people of the
4 x 4 grid, at all six targets, feedback shown and hidden, 16 seeds paired across targets
and feedback. Current model (V4): gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap,
T_DUR 7.0 s. Engagement = P(start next set), over the start decisions recorded.

Claims, fixed before the run (paired on seed):
  F1  "Showing feedback never lowers engagement" holds if, across the 96 person x target
      cells, shown - hidden is below zero beyond 2 SE in at most 4 cells and beyond 3 SE
      in none.
  F2  "Showing feedback raises engagement" holds for a person if shown - hidden is above
      zero beyond 2 SE at every target where that person starts at least 10% of the time
      with feedback shown.
Everything else is reported without a claim.

README: the "Across people", "By difficulty, over all 16 people" and "f and k set the two
ends of the engagement curve" bullets.

    python -m experiments.feedback_grid            # run (about 40 min), then summarize
    python -m experiments.feedback_grid --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from temporal.controller import TARGETS

from .common import LEVELS, args, out_dir, run_pool, se
from .task_choice import one_run, plateau

OUT = out_dir("feedback_grid")
SEEDS = 16


def summarize(df):
    cells = []
    for (f, k, t), g in df.groupby(["f", "k", "target"]):
        w = g.pivot(index="seed", columns="feedback", values="p_go")
        d = w[True] - w[False]
        cells.append(dict(f=f, k=k, target=t, shown=w[True].mean(), hidden=w[False].mean(),
                          diff=d.mean(), se=se(d)))
    c = pd.DataFrame(cells)
    c["z"] = c["diff"] / c["se"].replace(0, np.nan)
    below2, below3 = int((c.z < -2).sum()), int((c.z < -3).sum())
    print(f"== F1: of {len(c)} person x target cells, shown - hidden below zero beyond 2 SE: "
          f"{below2}, beyond 3 SE: {below3} -> {'HOLDS' if below2 <= 4 and below3 == 0 else 'FAILS'}; "
          f"{int((c['diff'] > 0).sum())} positive")

    holds = 0
    for _, g in c.groupby(["f", "k"]):
        live = g[g.shown >= 0.10]
        holds += bool(len(live)) and bool((live.z > 2).all())
    print(f"== F2: holds for {holds} of 16 people")

    print("\n== pooled over targets, per person: P(start) shown - hidden (no claim) ==")
    raised = []
    for (f, k), g in df.groupby(["f", "k"]):
        w = g.pivot_table(index="seed", columns="feedback", values="p_go", aggfunc="mean")
        d = w[True] - w[False]
        beyond = d.mean() > 2 * se(d)
        raised += [d.mean()] if beyond else []
        print(f"f {f} k {k}: {d.mean():+.3f} ± {se(d):.3f}{'  *' if beyond else ''}")
    print(f"raised beyond 2 SE for {len(raised)} of 16, by {min(raised):.3f} to {max(raised):.3f}")

    print("\n== by difficulty, all 16 people (no claim) ==")
    for col, fmt in (("p_go", "{:+.3f} ± {:.3f}"), ("reward", "{:+.1f} ± {:.1f}")):
        parts = []
        for t in TARGETS:
            w = df[df.target == t].pivot_table(index=["f", "k", "seed"], columns="feedback", values=col)
            d = w[True] - w[False]
            parts.append(f"{t:.0%} " + fmt.format(d.mean(), se(d)) + ("*" if abs(d.mean()) > 2 * se(d) else ""))
        print(f"{col:>7} ({len(d)} pairs per target): " + "  ".join(parts))

    print("\n== the two ends of the engagement curve, feedback shown (no claim) ==")
    shown = df[df.feedback].groupby(["f", "k", "target"]).p_go.mean()
    for p, level in (("f", 0.2), ("f", 1.5), ("k", 0.2), ("k", 1.5)):
        for t in (0.9, 0.2):
            x = shown.xs((level, t), level=(p, "target"))
            print(f"{p} = {level}, target {t:.0%}: P(start) {x.min():.2f} to {x.max():.2f}")
    bests = []
    for (f, k), g in df[df.feedback].groupby(["f", "k"]):
        w = g.pivot(index="seed", columns="target", values="p_go")[list(TARGETS)]
        best, near = plateau(w)
        bests.append(best)
        print(f"f {f} k {k}: best target {best:.0%}, plateau " + ", ".join(f"{t:.0%}" for t in near))
    print(f"best target ranges {min(bests):.0%} to {max(bests):.0%}")
    c.to_csv(OUT / "cells.csv", index=False)


if __name__ == "__main__":
    a = args(__doc__)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        tasks = [("grid", f, k, t, fb, 3000 + s) for f in LEVELS for k in LEVELS for t in TARGETS
                 for fb in (True, False) for s in range(SEEDS)]
        df = run_pool(one_run, tasks, OUT / "runs.csv", chunksize=4)
    summarize(df)
