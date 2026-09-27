import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from config import REENTRY_COST_HIGH, REENTRY_COST_LOW, REENTRY_COST_MID, UserParams
from controller import Controller
from controller import run as run_person
from inference import assert_row_floor, fit, passages_to_frame
from initiation import build_model
from persistence import attempts_to_frame, fit_patience, fit_persistence
from plots import ASSETS, plot_recovery
from temporal_difference_model import Simulation

# Crossed 2x2: value and cost vary orthogonally, which is what separates f from k.
DESIGN = [(v, c) for v in (0.3, 1.0) for c in (0.3, 1.0)]

GRID_LEVELS = (0.2, 0.6, 1.0, 1.5)
GRID = [(f, k) for f in GRID_LEVELS for k in GRID_LEVELS]

# People for the persistence test: middling, eager, reluctant, and strongly driven.
PERSISTENCE_GRID = [(0.8, 0.8), (1.2, 0.4), (0.6, 1.2), (1.2, 1.2)]
PERSISTENCE_SOURCES = ("decisions", "persistence", "joint")

# (f, k) and true patience: the same person patient and at a glance, and two others.
PATIENCE_GRID = [
    ((0.8, 0.8), 10.0),
    ((0.8, 0.8), 2.0),
    ((1.2, 0.4), 5.0),
    ((0.6, 1.2), 20.0),
]


def simulate_dataset(f_true, k_true, design=DESIGN, n_per_cell=200):
    model = build_model({"f_val": f_true, "k_val": k_true})

    rows = []
    undecided = {}
    for value, cost in design:
        sol = model.solve(conditions={"value": value, "cost": cost})
        samp = sol.sample(n_per_cell)
        for rt in samp.choice_upper:
            rows.append((value, cost, 1, float(rt)))
        for rt in samp.choice_lower:
            rows.append((value, cost, 0, float(rt)))
        undecided[(value, cost)] = n_per_cell - len(samp.choice_upper) - len(
            samp.choice_lower
        )

    df = pd.DataFrame(rows, columns=["value", "cost", "GO", "RT"])
    df.attrs["undecided"] = undecided
    return df


def collect_coupled(f_true, k_true, n_decisions=700, cost_profiles=None,
                    review_mode="scheduled", review_rate=0.25, diff_spread=0.0):
    if cost_profiles is None:
        cost_profiles = (REENTRY_COST_LOW, REENTRY_COST_HIGH)

    passages = []
    for cost in cost_profiles:
        sim = Simulation(UserParams(f=f_true, k=k_true), reentry_cost=cost,
                         review_mode=review_mode, review_rate=review_rate,
                         diff_spread=diff_spread)
        sim.run(max_decisions=n_decisions)
        passages.extend(sim.state.initiation_passages)

    return passages_to_frame(passages)

# One learner, one cold-start cost. The cost contrast comes from within the run: it swings
# warm-to-cold as sessions start and end, and moves with each episode's review load.
def collect_single(f_true, k_true, n_decisions=1200, reentry_cost=REENTRY_COST_MID,
                   include_stage=True, review_mode="scheduled", review_rate=0.25,
                   diff_spread=0.0, t0=0.0):
    sim = Simulation(UserParams(f=f_true, k=k_true, t0=t0), reentry_cost=reentry_cost,
                     review_mode=review_mode, review_rate=review_rate,
                     diff_spread=diff_spread)
    sim.run(max_decisions=n_decisions)

    passages = list(sim.state.initiation_passages)
    if include_stage:
        # The same accumulator on the same (f, k), so its draws pool straight in.
        # Only fires in review_mode="choice"; otherwise first_passage is empty.
        passages += sim.state.first_passage
    return passages_to_frame(passages)


# Warm-up decisions, discarded, before each learner is recorded.
RESEED_WARMUPS = (0, 200, 500, 1000)


# collect_single's data, split across learners at different points on the curve: each
# discards its warm-up decisions, then is recorded for an equal share of `n_decisions`.
def collect_reseeded(f_true, k_true, n_decisions=1200, warmups=RESEED_WARMUPS,
                     reentry_cost=REENTRY_COST_MID, review_mode="scheduled", review_rate=0.25,
                     diff_spread=0.0):
    per = n_decisions // len(warmups)
    passages = []
    for w in warmups:
        sim = Simulation(UserParams(f=f_true, k=k_true), reentry_cost=reentry_cost,
                         review_mode=review_mode, review_rate=review_rate,
                         diff_spread=diff_spread)
        sim.run(max_decisions=w + per)
        recorded = sim.state.initiation_passages[w:]
        passages += recorded
        if recorded:
            since = recorded[0]["timestep"]
            passages += [p for p in sim.state.first_passage if p["timestep"] >= since]
    return passages_to_frame(passages)


# One person measured as the interface measures a human. Every controller update is an
# exploration probe: the only source of sets with feedback hidden, which is what makes
# the two ways a set's reward can be timed both appear in the data.
def _probed_person(f_true, k_true, n_decisions, seed):
    probe = Controller(explore=1.0, seed=seed, known=(f_true, k_true))
    return run_person(UserParams(f=f_true, k=k_true), controller=probe,
                      n_decisions=n_decisions, seed=seed)


def _sweep(grid, reps, generate, out_path, title):
    records = []
    total = len(grid) * reps
    for i, ((f_true, k_true), rep) in enumerate(
        itertools.product(grid, range(reps)), start=1
    ):
        df = generate(f_true, k_true)
        assert_row_floor(df)
        f_hat, k_hat = fit(df)
        records.append((f_true, k_true, f_hat, k_hat))
        print(
            f"[{i}/{total}] rep {rep + 1} rows={len(df)} "
            f"f {f_true:.2f} -> {f_hat:.2f} | k {k_true:.2f} -> {k_hat:.2f}",
            flush=True,
        )

    res = pd.DataFrame(records, columns=["f_true", "k_true", "f_hat", "k_hat"])
    res.attrs["stats"] = plot_recovery(res, out_path, title=title, reps=reps)
    return res


def recovery(grid, reps=5, n_per_cell=400, out_path=None):
    return _sweep(
        grid,
        reps,
        lambda f, k: simulate_dataset(f, k, DESIGN, n_per_cell),
        Path(out_path) if out_path else ASSETS / "initiation_recovery.png",
        "Initiation DDM parameter recovery",
    )


def recovery_single(grid, reps=5, n_decisions=1200, out_path=None):
    return _sweep(
        grid,
        reps,
        lambda f, k: collect_single(f, k, n_decisions),
        Path(out_path) if out_path else ASSETS / "initiation_recovery_single.png",
        "Initiation DDM parameter recovery (one learner, endogenous cost)",
    )


def recovery_reseeded(grid, reps=5, n_decisions=1200, out_path=None):
    return _sweep(
        grid,
        reps,
        lambda f, k: collect_reseeded(f, k, n_decisions),
        Path(out_path) if out_path else ASSETS / "initiation_recovery_reseeded.png",
        "Initiation DDM parameter recovery (one person, four points on the learning curve)",
    )


def recovery_coupled(grid, reps=5, n_decisions=700, out_path=None):
    return _sweep(
        grid,
        reps,
        lambda f, k: collect_coupled(f, k, n_decisions),
        Path(out_path) if out_path else ASSETS / "initiation_recovery_coupled.png",
        "Initiation DDM parameter recovery (value from TD learner)",
    )


# f and k from decisions alone, from answer and skip times alone, and from both, on
# human-sized data.
def recovery_persistence(grid=PERSISTENCE_GRID, reps=3, n_decisions=300):
    records = []
    total = len(grid) * reps
    for i, ((f, k), rep) in enumerate(itertools.product(grid, range(reps)), start=1):
        sim = _probed_person(f, k, n_decisions, seed=rep)
        df = passages_to_frame(sim.state.initiation_passages + sim.state.first_passage)
        attempts = attempts_to_frame(sim.state.persistence)
        estimates = { # the same person's f and k from three sources
            "decisions": fit(df),
            "persistence": fit_persistence(attempts),
            "joint": fit(df, attempts=attempts),
        }
        skips = int(attempts["skipped"].sum())
        records.append((f, k, *(x for s in PERSISTENCE_SOURCES for x in estimates[s]),
                        len(df), len(attempts), skips))
        print(f"[{i}/{total}] rows={len(df)} questions={len(attempts)} skips={skips}  "
              + "  ".join(f"{s}: f {f:.2f}->{e[0]:.2f} k {k:.2f}->{e[1]:.2f}"
                          for s, e in estimates.items()), flush=True)

    res = pd.DataFrame(records, columns=[
        "f_true", "k_true", *(f"{p}_{s}" for s in PERSISTENCE_SOURCES for p in ("f", "k")),
        "rows", "questions", "skips"])
    res.attrs["stats"] = {
        s: {p: float(np.sqrt(((res[f"{p}_{s}"] - res[f"{p}_true"]) ** 2).mean())) for p in ("f", "k")}
        for s in PERSISTENCE_SOURCES
    }
    return res


# True non-decision times for the T0 test, in seconds.
T0_GRID = (0.0, 0.2, 0.3, 0.5)


# What assuming `T0 = 0` costs: one learner per run, generated with a true non-decision
# time, then fitted twice -- `T0` pinned at 0, and `T0` free.
def recovery_t0(grid=T0_GRID, f_true=0.8, k_true=0.8, reps=3, n_decisions=1200):
    records = []
    total = len(grid) * reps
    for i, (t0_true, rep) in enumerate(itertools.product(grid, range(reps)), start=1):
        df = collect_single(f_true, k_true, n_decisions, t0=t0_true)
        f_pin, k_pin = fit(df)
        f_free, k_free, t0_hat = fit(df, t0=True)
        records.append((t0_true, rep, len(df), f_pin, k_pin, f_free, k_free, t0_hat))
        print(f"[{i}/{total}] true T0 {t0_true:.2f} rows={len(df)}  pinned: f {f_pin:.2f} k {k_pin:.2f}"
              f"  free: f {f_free:.2f} k {k_free:.2f} T0 {t0_hat:.2f}", flush=True)

    res = pd.DataFrame(records, columns=["t0_true", "rep", "rows", "f_pinned", "k_pinned",
                                         "f_free", "k_free", "t0_hat"])
    res.attrs["stats"] = {
        f"T0 {t0:.2f}": {
            "f pinned error": float((g.f_pinned - f_true).mean() / f_true),
            "k pinned error": float((g.k_pinned - k_true).mean() / k_true),
            "f free error": float((g.f_free - f_true).mean() / f_true),
            "k free error": float((g.k_free - k_true).mean() / k_true),
            "T0 recovered": float(g.t0_hat.mean()),
        }
        for t0, g in res.groupby("t0_true")
    }
    return res


# Patience as the controller estimates it, fitted twice -- with the true f and k and with
# the fitted ones -- to separate its own error from what f and k pass on. Error is a factor.
def recovery_patience(grid=PATIENCE_GRID, reps=3, n_decisions=300):
    records = []
    total = len(grid) * reps
    for i, (((f, k), patience), rep) in enumerate(itertools.product(grid, range(reps)), start=1):
        params = UserParams(f=f, k=k, patience=patience)
        probe = Controller(explore=1.0, seed=rep, known=(f, k))
        sim = run_person(params, controller=probe, n_decisions=n_decisions, seed=rep)
        df = passages_to_frame(sim.state.initiation_passages + sim.state.first_passage)
        attempts = attempts_to_frame(sim.state.persistence)
        f_hat, k_hat = fit(df)
        gave_up = int(attempts["skipped"].sum())
        given_true = fit_patience(f, k, attempts) # patience' own error
        given_fit = fit_patience(f_hat, k_hat, attempts) # plus what f and k pass on
        records.append((f, k, patience, f_hat, k_hat, given_true, given_fit, len(attempts), gave_up))
        print(f"[{i}/{total}] questions={len(attempts)} given up={gave_up}  f {f:.2f}->{f_hat:.2f} "
              f"k {k:.2f}->{k_hat:.2f}  patience {patience:.0f}s -> {given_true:.1f}s with true f, k, "
              f"{given_fit:.1f}s with fitted", flush=True)

    res = pd.DataFrame(records, columns=["f_true", "k_true", "patience_true", "f_hat", "k_hat",
                                         "patience_given_true", "patience_given_fit",
                                         "questions", "gave_up"])

    def factor(col):
        # RMSE in log space, read back as a multiplicative error.
        return float(np.exp(np.sqrt((np.log(res[col] / res["patience_true"]) ** 2).mean())))

    res.attrs["stats"] = {"given true f, k": factor("patience_given_true"),
                          "given fitted f, k": factor("patience_given_fit")}
    return res
