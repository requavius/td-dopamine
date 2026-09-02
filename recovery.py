import itertools
from pathlib import Path

import pandas as pd

from config import REENTRY_COST_HIGH, REENTRY_COST_LOW, UserParams
from inference import assert_row_floor, fit, passages_to_frame
from initiation import build_model
from plots import ASSETS, plot_recovery
from temporal_difference_model import Simulation

# Crossed 2x2: value and cost vary orthogonally. This is what makes f and k
# Separately identifiable. Neither can explain the other's cell pattern
DESIGN = [(v, c) for v in (0.3, 1.0) for c in (0.3, 1.0)]

GRID_LEVELS = (0.2, 0.6, 1.0, 1.5)
GRID = [(f, k) for f in GRID_LEVELS for k in GRID_LEVELS]


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


def collect_coupled(f_true, k_true, n_decisions=700, cost_profiles=None):
    if cost_profiles is None:
        cost_profiles = (REENTRY_COST_LOW, REENTRY_COST_HIGH)

    passages = []
    for cost in cost_profiles:
        sim = Simulation(UserParams(f=f_true, k=k_true), reentry_cost=cost)
        sim.run(max_decisions=n_decisions)
        passages.extend(sim.state.initiation_passages)

    return passages_to_frame(passages)

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


def recovery_coupled(grid, reps=5, n_decisions=700, out_path=None):
    return _sweep(
        grid,
        reps,
        lambda f, k: collect_coupled(f, k, n_decisions),
        Path(out_path) if out_path else ASSETS / "initiation_recovery_coupled.png",
        "Initiation DDM parameter recovery (value from TD learner)",
    )
