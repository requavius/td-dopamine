"""Which difficulty keeps a person engaged and earns most, and what showing feedback does.

Question-type task (`controller.run`), skips on, one setting held for each run of 300
ticks (a tick is a start decision in a session, or a chance to come back while away),
review rate 0.25, skipped questions dropped, no controller. Current model (V4): gamma
0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s.

    a         five people, every target, feedback shown, 16 seeds paired across targets
    a_hidden  the same people, targets and seeds with feedback hidden
    b         f = k = 0.8, every target, feedback shown and hidden, 96 paired seeds

Claims, fixed before the run (paired on seed):
  A  a person's peak target is "determined" if its reward beats both neighbouring targets
     by more than 2 SE; otherwise the peak is reported as not determined.
  B  the feedback crossover is "established" if shown - hidden is above zero beyond 2 SE at
     some target and below zero beyond 2 SE at a harder one; otherwise "suggested" (sign
     flip within 2 SE) or "absent".

README: "Which difficulty and whether to show feedback" (the three tables). Paper Figures 7
and S4.

    python -m experiments.task_choice            # run, then summarize
    python -m experiments.task_choice --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from temporal.controller import TARGETS, Environment, run
from temporal.model.params import UserParams

from .common import args, design, out_dir, run_pool, se

D = design("task_choice", ticks=300, review=0.25,
           people=((0.6, 1.2), (1.0, 1.0), (0.8, 0.8), (0.4, 0.4), (1.2, 0.4)),
           seeds_a=16, person_b=(0.8, 0.8), seeds_b=96)
OUT = out_dir(D.out)
N = D.ticks
REVIEW = D.review
PEOPLE = D.people
SEEDS_A = D.seeds_a
SEEDS_B = D.seeds_b


# One run of one fixed setting, N ticks long. Shared with feedback_grid and lockout.
# P(start) is the share of recorded start decisions on which they chose to start, slow
# choices included; `away` is the share of ticks spent out of the task.
def one_run(task):
    part, f, k, target, feedback, seed = task
    sim = run(UserParams(f=f, k=k), env=Environment(target, REVIEW, feedback, False),
              n_ticks=N, seed=seed)
    st = sim.state
    starts = st.initiation_passages
    return dict(part=part, f=f, k=k, target=target, feedback=feedback, seed=seed,
                reward=st.total_reward, sets=st.episodes, decisions=len(starts),
                p_go=float(np.mean([p["chose"] == 1 for p in starts])) if starts else 0.0,
                away=st.away / N)


def tasks():
    out = [(part, f, k, t, fb, 1000 + s) for part, fb in (("a", True), ("a_hidden", False))
           for f, k in PEOPLE for t in TARGETS for s in range(SEEDS_A)]
    out += [("b", *D.person_b, t, fb, 2000 + s) for t in TARGETS for fb in (True, False)
            for s in range(SEEDS_B)]
    return out


# The best target, and every target whose paired difference from it is within 2 SE.
def plateau(wide):
    m = wide.mean()
    best = m.idxmax()
    near = [t for t in TARGETS if t == best or (wide[best] - wide[t]).mean() <= 2 * se(wide[best] - wide[t])]
    return best, near


def paired(df, col):
    w = df.pivot(index="seed", columns="feedback", values=col)
    d = w[True] - w[False]
    return d.mean(), se(d)


def summarize(df):
    a = df[df.part == "a"]
    heads = "  ".join(f"{t:>5.0%}" for t in TARGETS)
    print(f"== engagement, P(start next set), feedback shown, {SEEDS_A} seeds ==")
    print(f"{'f, k':>10}  {heads}   targets within 2 SE of the best")
    for f, k in PEOPLE:
        w = a[(a.f == f) & (a.k == k)].pivot(index="seed", columns="target", values="p_go")[list(TARGETS)]
        best, near = plateau(w)
        print(f"{f:>4}, {k:<4}  " + "  ".join(f"{w[t].mean():5.2f}" for t in TARGETS)
              + "   " + ", ".join(f"{t:.0%}" for t in near))

    print(f"\n== share of the {N} ticks spent away, the same runs (no claim) ==")
    print(f"{'f, k':>10}  {heads}")
    for f, k in PEOPLE:
        w = a[(a.f == f) & (a.k == k)].pivot(index="seed", columns="target", values="away")[list(TARGETS)]
        print(f"{f:>4}, {k:<4}  " + "  ".join(f"{w[t].mean():5.2f}" for t in TARGETS))

    print("\n== total reward, the same runs (claim A) ==")
    print(f"{'f, k':>10}  {heads}   peak")
    for f, k in PEOPLE:
        w = a[(a.f == f) & (a.k == k)].pivot(index="seed", columns="target", values="reward")[list(TARGETS)]
        best = w.mean().idxmax()
        i = TARGETS.index(best)
        beats = [(w[best] - w[TARGETS[j]]).mean() > 2 * se(w[best] - w[TARGETS[j]])
                 for j in (i - 1, i + 1) if 0 <= j < len(TARGETS)]
        print(f"{f:>4}, {k:<4}  " + "  ".join(f"{w[t].mean():5.1f}" for t in TARGETS)
              + f"   {best:.0%}, {'determined' if all(beats) else 'not determined'}")

    b = df[df.part == "b"]
    print(f"\n== feedback shown - hidden, f, k = {D.person_b}, {SEEDS_B} paired seeds (claim B) ==")
    diffs = {}
    for col, name in (("p_go", "P(start)"), ("away", "share away"), ("reward", "total reward")):
        cells = []
        for t in TARGETS:
            m, e = paired(b[b.target == t], col)
            if col == "reward":
                diffs[t] = (m, e)
            mark = "*" if abs(m) > 2 * e else " "
            cells.append(f"{t:.0%} {m:+.3f} ± {e:.3f}{mark}" if col != "reward" else
                         f"{t:.0%} {m:+.1f} ± {e:.1f}{mark}")
        print(f"{name:>12}: " + "  ".join(cells))
    print("(* beyond 2 SE)")
    pos = [t for t in TARGETS if diffs[t][0] > 2 * diffs[t][1]]
    neg = [t for t in TARGETS if diffs[t][0] < -2 * diffs[t][1]]
    flip = any(m > 0 for m, _ in diffs.values()) and any(m < 0 for m, _ in diffs.values())
    verdict = ("established" if pos and neg and min(neg) < max(pos) else
               "suggested (sign flips, not beyond 2 SE on both sides)" if flip else "absent")
    print(f"reward crossover: {verdict}")


if __name__ == "__main__":
    a = args(__doc__, D)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        df = run_pool(one_run, tasks(), OUT / "runs.csv", chunksize=4)
    summarize(df)
