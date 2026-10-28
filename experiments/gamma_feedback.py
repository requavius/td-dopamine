"""Does the discount factor γ change what showing feedback does to engagement?

With feedback shown, each right answer's reward lands when it is answered; with it
hidden, the set pays in one lump at its end. Value discounts slot j of a set by γ^j, so
hiding feedback should cost a steep discounter (low γ) more than a patient one. A seen
miss also produces a negative error that an unseen one does not, which pushes the other
way, and the continuing bootstrap spreads value over later sets, so the net effect is
the question.

task_choice's design with γ varied: question-type task (`controller.run`), skips on, one
setting held for each run of 300 ticks, review rate 0.25, skipped questions dropped, no
redo prompts, no controller. ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s.

    people     task_choice's five (f, k)
    γ          0.6, 0.7, 0.8, 0.9, 0.95 (the person's; α 0.05)
    targets    0.75 (the default) and 0.45 (misses are common)
    feedback   shown and hidden
    seeds      96, the same across γ, targets, feedback and people, so every difference
               pairs by seed

d(γ) = P(start, shown) − P(start, hidden), P(start) being the share of recorded start
decisions on which they chose to start (task_choice.one_run). Each seed's d(γ) is the
mean over the five people and two targets.

Claims, fixed before the run (2026-10-28):
    E1  d(0.7) − d(0.95) > 0 beyond 2 SE across seeds
    E2  the slope of d on γ (least squares over the five γ, per seed) is negative beyond
        2 SE across seeds
    E3  none: the same contrasts for total reward and the share of ticks away; d(γ) per
        target and per person; and baseline P(start) (feedback shown) at each γ

Lower γ lowers value, and with it engagement, everywhere, so low-γ people may rarely
start and leave little room for any difference. That can only shrink d at low γ, so it
works against E1 and E2 and cannot manufacture a pass. A fail is read as inconclusive,
not as evidence against, if baseline P(start) at γ 0.7 is below 0.15; the effect
relative to baseline, d / P(start, shown), is also reported, with no claim. No arm
rescales f to match baseline engagement across γ.

The simulator pays an end-of-set reward when feedback is hidden; the app shows no
end-of-set score yet. A human version of this result needs that score screen first.

    python -m experiments.gamma_feedback            # run, then summarize
    python -m experiments.gamma_feedback --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from temporal.controller import Environment, run
from temporal.model.params import UserParams

from .common import args, out_dir, run_pool, se
from .task_choice import N, PEOPLE, REVIEW

OUT = out_dir("gamma_feedback")
GAMMAS = (0.6, 0.7, 0.8, 0.9, 0.95)
TARGETS = (0.75, 0.45)
SEEDS = 96


# task_choice.one_run with the person's γ.
def one_run(task):
    f, k, g, target, feedback, seed = task
    sim = run(UserParams(f=f, k=k, g=g), env=Environment(target, REVIEW, feedback, False),
              n_ticks=N, seed=seed)
    st = sim.state
    starts = st.initiation_passages
    return dict(f=f, k=k, g=g, target=target, feedback=feedback, seed=seed,
                reward=st.total_reward, sets=st.episodes, decisions=len(starts),
                p_go=float(np.mean([p["chose"] == 1 for p in starts])) if starts else 0.0,
                away=st.away / N)


def tasks():
    return [(f, k, g, t, fb, 3000 + s) for g in GAMMAS for f, k in PEOPLE for t in TARGETS
            for fb in (True, False) for s in range(SEEDS)]


# shown − hidden for each (seed, γ), averaged over the cells left in `df`: seeds × γ.
def effect(df, col):
    w = df.pivot_table(index=["seed", "g", "f", "k", "target"], columns="feedback", values=col)
    d = (w[True] - w[False]).rename("d").reset_index()
    return d.groupby(["seed", "g"]).d.mean().unstack("g")[list(GAMMAS)]


def slopes(d):
    x = np.array(GAMMAS) - np.mean(GAMMAS)
    return (d.to_numpy() - d.to_numpy().mean(axis=1, keepdims=True)) @ x / (x @ x)


def contrasts(d):
    c = d[0.7] - d[0.95]
    s = slopes(d)
    return (c.mean(), se(c)), (s.mean(), se(s))


def row(d, fmt):
    return "  ".join(fmt(d[g].mean(), se(d[g])) for g in GAMMAS)


def summarize(df):
    heads = "  ".join(f"{'γ ' + str(g):>15}" for g in GAMMAS)
    pg = lambda m, e: f"{m:+.3f} ± {e:.3f}"  # noqa: E731
    shown = df[df.feedback]
    base = shown.groupby(["seed", "g"]).p_go.mean().unstack("g")[list(GAMMAS)]
    print(f"== baseline P(start), feedback shown, mean of {len(PEOPLE)} people × {len(TARGETS)} targets, "
          f"{SEEDS} seeds ==\n{'':>12}  {heads}")
    print(f"{'P(start)':>12}  " + "  ".join(f"{base[g].mean():15.3f}" for g in GAMMAS))

    print(f"\n== d(γ) = shown − hidden, paired by seed ==\n{'':>12}  {heads}")
    d = effect(df, "p_go")
    print(f"{'P(start)':>12}  {row(d, pg)}")
    rel = d / base
    print(f"{'÷ baseline':>12}  {row(rel, pg)}")
    for col, fmt in (("reward", lambda m, e: f"{m:+.1f} ± {e:.1f}"), ("away", pg)):
        print(f"{col:>12}  {row(effect(df, col), fmt)}")

    print("\nper target, P(start):")
    for t in TARGETS:
        print(f"{t:>12.0%}  {row(effect(df[df.target == t], 'p_go'), pg)}")
    print("per person, P(start):")
    for f, k in PEOPLE:
        print(f"{f:>5}, {k:<5}  {row(effect(df[(df.f == f) & (df.k == k)], 'p_go'), pg)}")

    (c, c_se), (s, s_se) = contrasts(d)
    floor = base[0.7].mean() < 0.15
    print(f"\nE1: d(0.7) − d(0.95) = {c:+.4f} ± {c_se:.4f}: "
          f"{'pass' if c > 2 * c_se else 'inconclusive (floor)' if floor else 'fail'}")
    print(f"E2: slope of d on γ = {s:+.4f} ± {s_se:.4f} per unit γ: "
          f"{'pass' if s < -2 * s_se else 'inconclusive (floor)' if floor else 'fail'}")
    print("E3 (no claim), the same contrasts:")
    for col in ("reward", "away"):
        (c, c_se), (s, s_se) = contrasts(effect(df, col))
        print(f"{col:>12}: d(0.7) − d(0.95) {c:+.4f} ± {c_se:.4f}, slope {s:+.4f} ± {s_se:.4f}")
    for t in TARGETS:
        (c, c_se), (s, s_se) = contrasts(effect(df[df.target == t], "p_go"))
        print(f"{'target ' + format(t, '.0%'):>12}: d(0.7) − d(0.95) {c:+.4f} ± {c_se:.4f}, "
              f"slope {s:+.4f} ± {s_se:.4f}")


if __name__ == "__main__":
    a = args(__doc__)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        df = run_pool(one_run, tasks(), OUT / "runs.csv", chunksize=4)
    summarize(df)
