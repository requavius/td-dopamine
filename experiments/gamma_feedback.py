"""Does what showing feedback does depend on γ?

Feedback is timing, not reward: shown, a slot's achievement lands when the outcome is
shown; hidden, the set's achievement lands in one lump at the end. Discounting is the only
thing that makes when reward lands matter, so the shown - hidden difference should depend
on γ. Figure 7 and feedback_grid measure it only at γ = 0.9.

task_choice's task and people (question-type task, skips on, one setting held for each run
of 300 ticks, review 0.25, skipped questions dropped, no controller), at all six targets,
γ 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, feedback shown and hidden, 32 seeds paired across γ,
target and feedback. Current model (V4): ACHIEVEMENT_SCALE 4, continuing bootstrap,
T_DUR 7.0 s. There are no redo prompts in this task, so every decision is a start.
γ = 1 is left out: with the continuing bootstrap a value never stops growing.

Claims, fixed before the run (2026-10-28, paired on seed):
  G1  the effect depends on γ for a person if the shown - hidden difference in P(start),
      averaged over the six targets, differs between γ 0.5 and γ 0.99 by more than 2 SE.
  G2  none: report reward and the share of ticks away by γ.

Paper Figure S6 (`python -m paper figS6`).

    python -m experiments.gamma_feedback            # run, then summarize
    python -m experiments.gamma_feedback --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from temporal.controller import TARGETS, Environment, run
from temporal.model.params import UserParams

from . import task_choice
from .common import args, design, out_dir, run_pool, se

D = design("gamma_feedback", ticks=task_choice.D.paper["ticks"],
           review=task_choice.D.paper["review"], people=task_choice.D.paper["people"],
           gammas=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99), seeds=32)
OUT = out_dir(D.out)
N, REVIEW, PEOPLE = D.ticks, D.review, D.people
GAMMAS = D.gammas
SEEDS = D.seeds


def one_run(task):
    f, k, g, target, feedback, seed = task
    sim = run(UserParams(f=f, k=k, g=g), env=Environment(target, REVIEW, feedback, False),
              n_ticks=N, seed=seed)
    st = sim.state
    starts = st.initiation_passages
    return dict(f=f, k=k, g=g, target=target, feedback=feedback, seed=seed,
                reward=st.total_reward, sets=st.episodes, decisions=len(starts),
                p_go=float(np.mean([p["chose"] == 1 for p in starts])) if starts else np.nan,
                away=st.away / N)


def tasks():
    return [(f, k, g, t, fb, 3000 + s) for f, k in PEOPLE for g in GAMMAS for t in TARGETS
            for fb in (True, False) for s in range(SEEDS)]


# shown - hidden for each (person, γ, target, seed), one row per pair.
def paired(df):
    key = ["f", "k", "g", "target", "seed"]
    on = df[df.feedback].set_index(key)
    off = df[~df.feedback].set_index(key)
    cols = ["p_go", "reward", "away"]
    return (on[cols] - off[cols]).reset_index()


def summarize(df):
    d = paired(df)
    print(f"== P(start), shown - hidden, mean over the six targets, {SEEDS} paired seeds ==")
    print(f"{'f, k':>10}  " + "  ".join(f"γ {g:<11}" for g in GAMMAS) + f"  G1 ({max(GAMMAS)} vs {min(GAMMAS)})")
    for f, k in PEOPLE:
        p = d[(d.f == f) & (d.k == k)]
        w = p.groupby(["g", "seed"]).p_go.mean().unstack("g")  # seed x γ, mean over targets
        cells = [f"{w[g].mean():+.3f}±{se(w[g]):.3f}" for g in GAMMAS]
        c = w[max(GAMMAS)] - w[min(GAMMAS)]
        verdict = "depends" if abs(c.mean()) > 2 * se(c) else "within 2 SE"
        print(f"{f:>4}, {k:<4}  " + "  ".join(f"{x:<13}" for x in cells)
              + f"  {c.mean():+.3f}±{se(c):.3f} {verdict}")
    for col, name, fmt in (("reward", "total reward", "+.1f"),
                           ("away", "share away", "+.3f")):
        print(f"\n== {name}, shown - hidden, mean over targets (G2, no claim) ==")
        for f, k in PEOPLE:
            p = d[(d.f == f) & (d.k == k)]
            w = p.groupby(["g", "seed"])[col].mean().unstack("g")
            print(f"{f:>4}, {k:<4}  " + "  ".join(f"{w[g].mean():{fmt}}" for g in GAMMAS))


if __name__ == "__main__":
    a = args(__doc__, D)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        df = run_pool(one_run, tasks(), OUT / "runs.csv", chunksize=4)
    summarize(df)
