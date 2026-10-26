"""Two checks on value: how often the carried error hits DELTA_CLIP, and what the
continuing bootstrap does to value's size and spread.

Question-type task (`controller.run`), skips on, no controller. Current model (V4):
gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s.

    clip       four people, f, k = 0.8/0.8, 1.2/0.4, 0.6/1.2, 1.2/1.2, 2 seeds, 300 start
               decisions, the default setting (target 0.75, review 0.25) and the same
               target with redo prompts instead of review, feedback shown and hidden. The
               share of start decisions whose last-set error, and of redo decisions whose
               slot error, lies beyond DELTA_CLIP; and how much the start node's effort cost
               (review load) adds and varies, as EFFORT_WEIGHT * review load.
    bootstrap  f = k = 0.8, the default setting, 8 seeds, 400 start decisions, the same
               runs with value.CONTINUING on and off: mean and sd of value at the start
               decisions.

Reported without a claim. Both were measured by hand on 2026-10-24 under V3, with no
script; this one replaces them.

README: the DELTA_CLIP and effort-cost bullets under "Known limitations", and "Why it
matters for measurement" under "Value bootstrap".

    python -m experiments.value_checks            # run (minutes), then summarize
    python -m experiments.value_checks --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from temporal.controller import Environment, run
from temporal.model import value
from temporal.model.conditions import DELTA_CLIP, EFFORT_WEIGHT
from temporal.model.params import UserParams
from temporal.recovery import PERSISTENCE_GRID

from .common import args, out_dir, run_pool

OUT = out_dir("value_checks")
CLIP_N, BOOT_N = 300, 400


def _one(task):
    part, f, k, redo, feedback, continuing, seed = task
    value.CONTINUING = continuing  # each task sets it, so a reused worker cannot carry it over
    env = Environment(0.75, 0.0 if redo else 0.25, feedback, False, redo)
    sim = run(UserParams(f=f, k=k), env=env, n_decisions=CLIP_N if part == "clip" else BOOT_N,
              seed=seed)
    value.CONTINUING = True
    starts, redos = sim.state.initiation_passages, sim.state.first_passage
    load = np.array([p["cost_effort"] / EFFORT_WEIGHT - 1.0 for p in starts])
    v = np.array([p["value"] for p in starts])
    return dict(part=part, f_true=f, k_true=k, redo=redo, feedback=feedback,
                continuing=continuing, seed=seed, decisions=len(starts), redo_decisions=len(redos),
                start_clipped=float(np.mean([abs(p["last_set_delta"]) > DELTA_CLIP for p in starts])),
                redo_clipped=(float(np.mean([abs(r["slot_delta"]) > DELTA_CLIP for r in redos]))
                              if redos else np.nan),
                effort_mean=float(EFFORT_WEIGHT * load.mean()), effort_sd=float(EFFORT_WEIGHT * load.std()),
                value_mean=float(v.mean()), value_sd=float(v.std()))


def tasks():
    out = [("clip", f, k, redo, fb, True, s) for f, k in PERSISTENCE_GRID
           for redo in (False, True) for fb in (True, False) for s in (0, 1)]
    out += [("bootstrap", 0.8, 0.8, False, True, c, s) for c in (True, False) for s in range(8)]
    return out


def summarize(df):
    c = df[df.part == "clip"]
    print(f"== clip: share beyond DELTA_CLIP = {DELTA_CLIP}, by setting (range over the 4 people, "
          "pooled over 2 seeds) ==")
    for redo in (False, True):
        for fb in (True, False):
            g = c[(c.redo == redo) & (c.feedback == fb)].groupby(["f_true", "k_true"])
            s = g.start_clipped.mean()
            r = g.redo_clipped.mean()
            redo_part = (f", redo decisions {r.min():.1%} to {r.max():.1%}" if redo and fb
                         else "")
            print(f"{'redo prompts' if redo else 'review 0.25':>12}, feedback "
                  f"{'shown' if fb else 'hidden'}: start decisions {s.min():.1%} to {s.max():.1%}"
                  + redo_part)
    plain = c[~c.redo]
    print(f"effort cost at the start node, default setting: review load adds "
          f"{plain.effort_mean.mean():.3f} on average, sd {plain.effort_sd.min():.3f} to "
          f"{plain.effort_sd.max():.3f} within a run")

    b = df[df.part == "bootstrap"]
    print("\n== bootstrap: value at the start decisions, f = k = 0.8, 8 seeds ==")
    for cont in (True, False):
        g = b[b.continuing == cont]
        print(f"{'continuing' if cont else 'terminal':>10}: mean {g.value_mean.mean():.2f}, "
              f"sd {g.value_sd.mean():.2f}")
    on, off = b[b.continuing], b[~b.continuing]
    print(f"sd ratio continuing / terminal: {on.value_sd.mean() / off.value_sd.mean():.2f}")


if __name__ == "__main__":
    a = args(__doc__)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        df = run_pool(_one, tasks(), OUT / "runs.csv")
        df = df.sort_values(["part", "f_true", "k_true", "redo", "feedback", "continuing", "seed"])
        df.to_csv(OUT / "runs.csv", index=False)
    summarize(df)
