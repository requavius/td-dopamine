"""Paper 5.2: what each part of the fit contributes, alone and in combination, on the
same data. Current model (V4): gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap,
T_DUR 7.0 s, confidence off.

Data: the 32 people of paper 5.4 (4 x 4 grid of f, k in {0.2, 0.6, 1.0, 1.5}, 2 seeds),
question-type task, skips on, 300 start decisions, settings chosen by InfoController over
all 96 (the app's controller), never shown the truth. True non-decision time 0.3 s, true
patience 10 s. f and k are fitted from start decisions up to the 300th and the redo
decisions before it; answer and skip times from the whole run.

Each dataset is fitted 8 ways, every combination of three choices:
  timeouts   kept as undecided (the pipeline) | dropped
  T0         fitted in [0, 0.5] s (the pipeline) | pinned at 0
  times      not used (the pipeline) | answer and skip times added to the likelihood
             at a fixed 10 s patience, which is the true patience here (their best case)

Claims, fixed before the run (paired on person, squared error and signed bias):
  C1  a choice matters alone if switching only it from the pipeline worsens squared error
      in f or k beyond 2 SE.
  C2  two choices interact if the bias from switching both differs from the sum of the
      biases from switching each alone, beyond 2 SE (bias adds if effects are independent).
Everything else is reported without a claim.

README: "What each part of the fit contributes, alone and together". Paper Figure S3.

    python -m experiments.fit_factorial            # run (about 1 h), then summarize
    python -m experiments.fit_factorial --replot   # summarize the saved results
"""
import itertools

import numpy as np
import pandas as pd

from temporal.controller import Environment, InfoController, run
from temporal.inference import fit, passages_to_frame
from temporal.model.params import UserParams
from temporal.model.persistence import attempts_to_frame

from .common import LEVELS, Quiet, args, design, out_dir, run_pool, se

D = design("fit_factorial", levels=LEVELS, reps=2, decisions=300, t0_true=0.3)
OUT = out_dir(D.out)
REPS = D.reps
N = D.decisions
T0_TRUE = D.t0_true
FACTORS = {"timeouts": ("kept", "dropped"), "t0": ("fitted", "pinned"), "times": ("no", "yes")}
PIPELINE = ("kept", "fitted", "no")


def _one(task):
    f, k, seed = task
    with Quiet():
        return _fits(f, k, seed)


def _fits(f, k, seed):
    ctl = InfoController(seed=seed)
    sim = run(UserParams(f=f, k=k, t0=T0_TRUE), controller=ctl, env=Environment(),
              n_decisions=N, seed=seed)
    starts, redos = sim.state.initiation_passages, sim.state.first_passage
    upto = starts[N - 1]["timestep"]
    df = passages_to_frame(starts[:N] + [r for r in redos if r["timestep"] <= upto])
    attempts = attempts_to_frame(sim.state.persistence)
    rows = []
    for timeouts, t0, times in itertools.product(*FACTORS.values()):
        data = df.copy()
        data.attrs["undecided"] = dict(df.attrs.get("undecided", {})) if timeouts == "kept" else {}
        out = fit(data, attempts=attempts if times == "yes" else None, t0=(t0 == "fitted"))
        rows.append(dict(f_true=f, k_true=k, seed=seed, timeouts=timeouts, t0=t0, times=times,
                         f_hat=out[0], k_hat=out[1], t0_hat=out[2] if len(out) > 2 else 0.0,
                         decided=len(df), undecided=int(sum(df.attrs.get("undecided", {}).values())),
                         questions=len(attempts), skips=int(attempts["skipped"].sum())))
    return rows


def summarize(df):
    df = df.assign(ef=(df.f_hat - df.f_true) ** 2, ek=(df.k_hat - df.k_true) ** 2,
                   bf=100 * (df.f_hat - df.f_true) / df.f_true, bk=100 * (df.k_hat - df.k_true) / df.k_true)
    key = ["f_true", "k_true", "seed"]
    cells = {c: g.set_index(key).sort_index() for c, g in df.groupby(["timeouts", "t0", "times"])}
    base = cells[PIPELINE]
    print(f"== {len(base)} people, {N} start decisions, true T0 {T0_TRUE} s ==")
    print(f"{'timeouts':>9} {'T0':>7} {'times':>6} | {'f RMSE':>7} {'k RMSE':>7} | {'f bias %':>9} {'k bias %':>9}")
    order = sorted(cells, key=lambda c: sum(a != b for a, b in zip(c, PIPELINE)))
    for c in order:
        g = cells[c]
        print(f"{c[0]:>9} {c[1]:>7} {c[2]:>6} | {np.sqrt(g.ef.mean()):7.3f} {np.sqrt(g.ek.mean()):7.3f} | "
              f"{g.bf.mean():+9.1f} {g.bk.mean():+9.1f}")

    print("\n== C1: switching one choice from the pipeline (squared error, switched - pipeline) ==")
    singles = {}
    for i, (name, levels) in enumerate(FACTORS.items()):
        c = list(PIPELINE)
        c[i] = levels[1]
        c = tuple(c)
        singles[name] = c
        for p in ("f", "k"):
            d = cells[c][f"e{p}"] - base[f"e{p}"]
            verdict = "matters" if d.mean() > 2 * se(d) else "not beyond 2 SE"
            print(f"{name:>9} {p}: {d.mean():+.4f} +- {se(d):.4f}  -> {verdict}")

    print("\n== C2: interactions on bias (both switched - sum of each alone), percentage points ==")
    names = list(FACTORS)
    for a, b in itertools.combinations(range(3), 2):
        c = list(PIPELINE)
        c[a], c[b] = FACTORS[names[a]][1], FACTORS[names[b]][1]
        c = tuple(c)
        for p in ("f", "k"):
            both = cells[c][f"b{p}"] - base[f"b{p}"]
            alone = (cells[singles[names[a]]][f"b{p}"] - base[f"b{p}"]) + \
                    (cells[singles[names[b]]][f"b{p}"] - base[f"b{p}"])
            d = both - alone
            verdict = "interact" if abs(d.mean()) > 2 * se(d) else "additive within 2 SE"
            print(f"{names[a]} x {names[b]} {p}: both {both.mean():+.1f}, sum of single "
                  f"{alone.mean():+.1f}, difference {d.mean():+.1f} +- {se(d):.1f} -> {verdict}")
    print(f"\nT0 recovered in the pipeline: {base.t0_hat.mean():.2f} +- {base.t0_hat.std():.2f} s "
          f"(true {T0_TRUE}); timed out {base.undecided.sum() / (base.undecided.sum() + base.decided.sum()):.1%}; "
          f"skips per person {base.skips.mean():.0f}")


if __name__ == "__main__":
    a = args(__doc__, D)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        tasks = [(f, k, s) for f in D.levels for k in D.levels for s in range(REPS)]
        df = run_pool(_one, tasks, OUT / "runs.csv",
                      label=lambda rows: f"f {rows[0]['f_true']} k {rows[0]['k_true']} seed {rows[0]['seed']}")
    summarize(df)
