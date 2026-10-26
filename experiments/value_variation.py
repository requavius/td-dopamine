"""How much of the variation in value is prediction error, and how value and cost correlate.

At a start decision value = V(0) + the last set's prediction error, carried at weight 1
(`conditions.value_at_choice_point`). From each start decision's record, the carried error
is `carried_delta(last_set_delta)` and V(0) is the recorded value minus it (value is
recorded snapped to 0.05). Question-type task (`controller.run`), skips on. Current model (V4):
gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s.

    probed  the four people of the persistence test, 2 seeds, 700 start decisions,
            settings switched by random probes every 25 decisions; f and k fitted from
            the start and redo decisions
    held    f = k = 0.8, 400 start decisions at the default setting (target 0.75, review
            0.25), feedback shown and hidden, 8 seeds

Reported without a claim. "Share of value's sd from the error" is sd(error) / sd(value).

README: "Prediction error is a large part of the variation in value" and the carry table under
"How value and cost together decide what can be measured".

    python -m experiments.value_variation            # run, then summarize
    python -m experiments.value_variation --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from temporal.controller import Controller, Environment, run
from temporal.inference import fit, passages_to_frame
from temporal.model.conditions import carried_delta
from temporal.model.params import UserParams
from temporal.recovery import PERSISTENCE_GRID

from .common import Quiet, args, out_dir, run_pool

OUT = out_dir("value_variation")
PROBED_N, HELD_N = 700, 400
HELD_SEEDS = range(8)


def _one(task):
    part, f, k, feedback, seed = task
    with Quiet():
        if part == "probed":
            ctl = Controller(explore=1.0, seed=seed, known=(f, k))
            sim = run(UserParams(f=f, k=k), controller=ctl, n_decisions=PROBED_N, seed=seed)
        else:
            sim = run(UserParams(f=f, k=k), env=Environment(feedback=feedback),
                      n_decisions=HELD_N, seed=seed)
        st = sim.state
        starts = st.initiation_passages
        value = np.array([p["value"] for p in starts])
        cost = np.array([p["cost"] for p in starts])
        delta = np.array([carried_delta(p["last_set_delta"]) for p in starts])
        v0 = value - delta
        out = dict(part=part, f_true=f, k_true=k, feedback=feedback, seed=seed,
                   decisions=len(starts), sd_v0=v0.std(), sd_delta=delta.std(),
                   sd_value=value.std(), corr_v0_cost=np.corrcoef(v0, cost)[0, 1],
                   corr_delta_cost=np.corrcoef(delta, cost)[0, 1],
                   corr_value_cost=np.corrcoef(value, cost)[0, 1])
        if part == "probed":
            out["f_hat"], out["k_hat"] = fit(passages_to_frame(starts + st.first_passage))
    return out


def summarize(df):
    p = df[df.part == "probed"]
    print(f"== probed: {len(p)} fits, {PROBED_N} start decisions each ==")
    print(f"f RMSE {np.sqrt(((p.f_hat - p.f_true) ** 2).mean()):.3f}, "
          f"k RMSE {np.sqrt(((p.k_hat - p.k_true) ** 2).mean()):.3f}")
    print(f"sd of value {p.sd_value.min():.2f} to {p.sd_value.max():.2f}; "
          f"corr(value, cost) {p.corr_value_cost.min():.2f} to {p.corr_value_cost.max():.2f}")
    share = p.sd_delta / p.sd_value
    print(f"share of value's sd from the error: {share.mean():.0%} ({share.min():.0%} to {share.max():.0%})")

    h = df[df.part == "held"]
    print(f"\n== held: f = k = 0.8, {HELD_N} start decisions, means over {len(HELD_SEEDS)} seeds ==")
    print(f"{'':>13} corr(V(0), cost)  corr(error, cost)  corr(value, cost)  sd of value, V(0) -> with error")
    for fb in (True, False):
        g = h[h.feedback == fb]
        print(f"{'feedback on' if fb else 'feedback off':>13} {g.corr_v0_cost.mean():16.2f}  "
              f"{g.corr_delta_cost.mean():17.2f}  {g.corr_value_cost.mean():17.2f}  "
              f"{g.sd_v0.mean():.3f} -> {g.sd_value.mean():.3f}")


if __name__ == "__main__":
    a = args(__doc__)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        tasks = [("probed", f, k, True, s) for f, k in PERSISTENCE_GRID for s in (0, 1)]
        tasks += [("held", 0.8, 0.8, fb, s) for fb in (True, False) for s in HELD_SEEDS]
        df = run_pool(_one, tasks, OUT / "runs.csv")
        df = df.sort_values(["part", "f_true", "k_true", "feedback", "seed"])
        df.to_csv(OUT / "runs.csv", index=False)
    summarize(df)
