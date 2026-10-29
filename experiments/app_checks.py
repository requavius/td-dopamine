"""Three checks on what the app's controller (controller.AppController) relies on.

Question-type task (`controller.run`), skips on, no controller unless said. Current model
(V4): γ 0.9, BOUND 1.5, SIGMA 0.7, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s.

    fit    the closed-form fit (inference.fit_closed_form) against pyddm's
           (inference.fit with T0), on records with no countdown. Four people, f, k =
           0.8/0.8, 1.2/0.4, 0.6/1.2, 1.2/1.2, T0 0.2 s, 2 seeds, 150 start decisions plus
           their redo decisions, aim 45% right with redo prompts. Each fit's estimate, and
           both points' -log L under the closed form.
    lam    λ by the closed-form fit, with a 3 s countdown on every redo prompt. f 0.9, k 0.7,
           λ 0.4, 0.6, 0.8 and 1.0, 2 seeds, 150 start decisions plus their redo decisions,
           the same setting. The fit holds γ and BOUND at the truth, as the app's does.
    plan   seconds per AppController plan over its 1104 settings, after 80 start decisions
           of the `lam` setting at λ 0.6 (seed 2): the first plan with empty caches, then
           two more. Machine-dependent; run on its own, after the pool.

Reported without a claim. README: "The app's controller".

    python -m experiments.app_checks            # run (a minute or two), then summarize
    python -m experiments.app_checks --replot   # summarize the saved results
"""
import contextlib
import io
import time

import numpy as np
import pandas as pd

from temporal import fastlik
from temporal.controller import APP_CANDIDATES, AppController, Environment, run
from temporal.inference import decision_arrays, fit, fit_closed_form, passages_to_frame
from temporal.model import accumulator as acc
from temporal.model.params import UserParams
from temporal.recovery import PERSISTENCE_GRID

from .common import args, out_dir, run_pool

OUT = out_dir("app_checks")
N = 150
REDO_ENV = Environment(0.45, 0.0, True, False, True)
TIMED_ENV = Environment(0.45, 0.0, True, False, True, pressure=True, countdown=3.0)


def _closed_nll(passages, f, k, t0):
    go, rt, v, c, _ = decision_arrays(passages)
    return fastlik.neg_log_likelihood(go, rt, f * v - k * c, t0=t0)


def _one(task):
    part, f, k, lam, seed = task
    env = TIMED_ENV if part == "lam" else REDO_ENV
    with contextlib.redirect_stderr(io.StringIO()):
        sim = run(UserParams(f=f, k=k, t0=0.2 if part == "fit" else 0.0, pressure=lam),
                  env=env, n_decisions=N, seed=seed)
    ps = sim.state.initiation_passages + sim.state.first_passage
    cf = fit_closed_form(ps)
    row = dict(part=part, f_true=f, k_true=k, lam_true=lam, seed=seed, decisions=len(ps),
               timed=sum("deadline" in p for p in ps), f_cf=cf[0], k_cf=cf[1], t0_cf=cf[2],
               lam_cf=cf[3])
    if part == "fit":
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            py = fit(passages_to_frame(ps), t0=True)
        row.update(f_py=py[0], k_py=py[1], t0_py=py[2],
                   nll_gap=_closed_nll(ps, *py) - _closed_nll(ps, *cf[:3]))
    return row


def tasks():
    out = [("fit", f, k, 1.0, s) for f, k in PERSISTENCE_GRID for s in (0, 1)]
    out += [("lam", 0.9, 0.7, lam, s) for lam in (0.4, 0.6, 0.8, 1.0) for s in (0, 1)]
    return out


def plan_times():
    with contextlib.redirect_stderr(io.StringIO()):
        sim = run(UserParams(f=0.9, k=0.7, pressure=0.6), env=TIMED_ENV, n_decisions=80, seed=2)
    acc.clear_caches()
    ctl = AppController(seed=3)
    out = []
    for i in range(3):
        t = time.perf_counter()
        with contextlib.redirect_stderr(io.StringIO()):
            ctl.plan(sim.state, UserParams(f=0.0, k=0.0))
        out.append(dict(part="plan", seed=i, seconds=time.perf_counter() - t,
                        settings=len(APP_CANDIDATES)))
    return out


def summarize(df):
    fitted = df[df.part == "fit"]
    print("== fit: closed form minus pyddm, over 4 people x 2 seeds ==")
    for name in ("f", "k", "t0"):
        d = fitted[f"{name}_cf"] - fitted[f"{name}_py"]
        rel = d / fitted[f"{name}_py"]
        print(f"{name:>3}: mean {d.mean():+.3f}, largest {d.abs().max():.3f}"
              + (f" ({rel.abs().max():.1%} of pyddm's)" if name != "t0" else ""))
    print(f"-log L under the closed form, pyddm's point minus its own: median "
          f"{fitted.nll_gap.median():.2f}, largest {fitted.nll_gap.max():.2f}")

    lam = df[df.part == "lam"]
    print("\n== lam: closed-form fit with a 3 s countdown on every redo prompt, 2 seeds ==")
    for true, g in lam.groupby("lam_true"):
        print(f"λ {true:.1f}: fitted {', '.join(f'{x:.2f}' for x in g.lam_cf)} "
              f"from {g.timed.mean():.0f} timed of {g.decisions.mean():.0f} decisions")
    err = lam.lam_cf - lam.lam_true
    print(f"λ RMSE {np.sqrt((err ** 2).mean()):.3f}, bias {err.mean():+.3f}; "
          f"f RMSE {np.sqrt(((lam.f_cf - lam.f_true) ** 2).mean()):.3f}, "
          f"k RMSE {np.sqrt(((lam.k_cf - lam.k_true) ** 2).mean()):.3f}")

    plan = df[df.part == "plan"]
    print(f"\n== plan: {int(plan.settings.iloc[0])} settings, seconds per plan: "
          + ", ".join(f"{s:.1f}" for s in plan.seconds) + " (first with empty caches) ==")


if __name__ == "__main__":
    a = args(__doc__)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        df = run_pool(_one, tasks(), OUT / "runs.csv")
        df = pd.concat([df, pd.DataFrame(plan_times())], ignore_index=True)
        df.to_csv(OUT / "runs.csv", index=False)
    summarize(df)
