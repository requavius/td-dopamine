"""Fitting (f, k) to decisions, with undecided trials kept as data."""
from collections import Counter

import numpy as np
import pandas as pd
import pyddm
from scipy.optimize import minimize

from . import fastlik
from .model.accumulator import BOUND, T_DUR, build_model
from .model.persistence import neg_log_likelihood

ROW_FLOOR = 400
# `value` already carries the prediction error out of the last set, so the conditions
# are the whole of the drift (accumulator.initiation_drift).
CONDITIONS = ("value", "cost")
# Fitted non-decision time. The upper end stays under the fastest decision in the data.
T0_BOUNDS = (0.0, 0.5)
# λ, the share of BOUND a person keeps on a redo prompt under a countdown
# (model/pressure.py): its prior, N(mean, sd), and its range. The prior is
# experiments.controller_all's; with no countdown in the record λ stays at its mean.
LAM_PRIOR = (0.7, 0.2)
LAM_BOUNDS = (0.2, 1.5)


# Decided trials as rows; undecided ones as counts per condition.
def passages_to_frame(passages):
    rows = []
    undecided = Counter()
    for p in passages:
        cond = tuple(float(p.get(c) or 0.0) for c in CONDITIONS)
        # No bound reached by the deadline: a timeout, or a person slower than T_DUR.
        if p["GO"] is None or p["RT"] is None or p["RT"] > T_DUR:
            undecided[cond] += 1
        else:
            rows.append((*cond, p["GO"], p["RT"]))

    df = pd.DataFrame(rows, columns=[*CONDITIONS, "GO", "RT"])
    df.attrs["undecided"] = dict(undecided)
    return df


DEADLINE_TOLERANCE = 0.02  # past this share of decisions over T_DUR, move the deadline


# Decision times against the deadline, for start decisions, redo decisions and both.
def audit_rts(passages, deadline=None):
    deadline = T_DUR if deadline is None else deadline
    groups = {"start": [], "stage": []}
    for p in passages:
        if p.get("RT") is not None:
            groups["stage" if p.get("node") == "stage" else "start"].append(float(p["RT"]))
    groups["all"] = groups["start"] + groups["stage"]
    out = {}
    for name, rts in groups.items():
        rts = np.array(rts)
        if len(rts) == 0:
            out[name] = {"n": 0}
            continue
        # share_over is the share of choices the fit never sees, being read as undecided.
        out[name] = {"n": len(rts), "min": rts.min(), "median": float(np.median(rts)),
                     "q90": float(np.quantile(rts, 0.9)), "max": rts.max(),
                     "over": int((rts > deadline).sum()), "share_over": float((rts > deadline).mean())}
    out["deadline"] = deadline
    return out


def _condition_columns(df):
    return [c for c in df.columns if c not in ("GO", "RT")]


def _add_undecided(sample, undecided, names):
    empty = np.array([])
    for key, count in undecided.items():
        if count <= 0:
            continue
        # An undecided trial has no RT, so only its conditions are repeated.
        conditions = {name: (empty, empty, np.repeat(value, count))
                      for name, value in zip(names, key)}
        sample = sample + pyddm.Sample(
            choice_upper=empty,
            choice_lower=empty,
            undecided=count,
            **conditions,
        )
    return sample


def build_sample(df):
    sample = pyddm.Sample.from_pandas_dataframe(
        df, rt_column_name="RT", choice_column_name="GO"
    )
    # Undecided trials are data: fitting decided ones alone inflates drift.
    return _add_undecided(sample, df.attrs.get("undecided", {}), _condition_columns(df))


# pyddm's likelihood loss plus the persistence likelihood of `attempts`.
def _with_persistence(attempts):
    class LossWithPersistence(pyddm.LossLikelihood):
        name = "Negative log likelihood, decisions + persistence"

        def loss(self, model):
            drift = model.parameters()["drift"]
            # Both depend on the same f and k, so the two -log L add.
            return (super().loss(model)
                    + neg_log_likelihood(float(drift["f_val"]), float(drift["k_val"]), attempts))

    return LossWithPersistence


# (f, k), plus T0 with `t0`. Two parameters is the whole model: how the last set moves
# this person is carried by `value`, not by terms of its own.
# `attempts` (persistence.attempts_to_frame) are fitted jointly with the decisions.
def fit(df, attempts=None, t0=False):
    parameters = {"f_val": (0.0, 3.0), "k_val": (0.0, 3.0)}
    if t0:
        parameters["T0"] = T0_BOUNDS
    model = build_model(parameters, nondecision="T0" if t0 else 0)
    data = df[[*CONDITIONS, "GO", "RT"]].copy()
    data.attrs["undecided"] = df.attrs.get("undecided", {})
    if attempts is not None and len(attempts):
        model.fit(build_sample(data), lossfunction=_with_persistence(attempts), verbose=False)
    else:
        model.fit(build_sample(data), verbose=False)

    drift_params = model.parameters()["drift"]
    out = tuple(float(drift_params[n]) for n in ("f_val", "k_val"))
    if t0:
        out += (float(model.parameters()["overlay"]["nondectime"]),)
    return out


# Decision records as arrays for the closed-form likelihood: go (1, 0, or -1 when no bound
# was reached by the deadline), rt, value, cost, and each one's countdown in seconds (NaN
# for none). A click slower than the deadline is undecided, as in passages_to_frame.
def decision_arrays(passages):
    deadline = np.array([p.get("deadline") or np.nan for p in passages], dtype=float)
    t_dur = np.where(np.isnan(deadline), T_DUR, deadline)
    rt = np.array([np.nan if p["RT"] is None else p["RT"] for p in passages], dtype=float)
    go = np.array([-1 if p["GO"] is None or not r <= t else p["GO"]
                   for p, r, t in zip(passages, rt, t_dur)], dtype=np.int8)
    value = np.array([float(p.get("value") or 0.0) for p in passages])
    cost = np.array([float(p.get("cost") or 0.0) for p in passages])
    return go, rt, value, cost, deadline


# (f, k, T0, λ) by maximum likelihood in closed form (temporal.fastlik), from decision
# records. A redo prompt under a countdown is read at bound λ·BOUND and its own deadline,
# every other decision at BOUND and T_DUR. λ alone has a prior (LAM_PRIOR). With no
# countdown in the record this is `fit(..., t0=True)` with the closed form in place of
# pyddm's solution, which agrees with it within 0.2% of the density's peak.
def fit_closed_form(passages):
    go, rt, value, cost, deadline = decision_arrays(passages)
    timed = ~np.isnan(deadline)
    t_dur = np.where(timed, deadline, T_DUR)

    def loss(th):
        f, k, t0, lam = th
        bound = BOUND * np.where(timed, lam, 1.0)
        nll = fastlik.neg_log_likelihood(go, rt, f * value - k * cost, bound, t0, t_dur=t_dur)
        return nll + 0.5 * ((lam - LAM_PRIOR[0]) / LAM_PRIOR[1]) ** 2

    bounds = [(0.0, 3.0), (0.0, 3.0), T0_BOUNDS, LAM_BOUNDS]
    starts = [(0.8, 0.8, 0.1, LAM_PRIOR[0]), (1.6, 1.6, 0.1, LAM_PRIOR[0]),
              (0.4, 0.4, 0.3, 1.0)]
    best = min((minimize(loss, x, method="L-BFGS-B", bounds=bounds) for x in starts),
               key=lambda r: r.fun)
    return tuple(float(x) for x in best.x)


def assert_row_floor(df):
    assert len(df) >= ROW_FLOOR, (
        f"dataset too small: {len(df)} rows (need >= {ROW_FLOOR})"
    )


def sanity_check(df):
    assert_row_floor(df)

    table = df.groupby(["value", "cost"]).agg(
        pGO=("GO", "mean"),
        medRT=("RT", "median"),
    )
    print(table)
    return table


def sanity_check_coupled(df):
    assert_row_floor(df)

    # High/low value within one cost level, since value here is continuous.
    def band(group):
        cut = group["value"].median()
        if (group["value"] >= cut).all():
            cut = group["value"].mean() # the median sat on the modal value
        return np.where(group["value"] > cut, "high", "low")

    banded = pd.concat(
        [g.assign(value_band=band(g)) for _, g in df.groupby("cost", sort=True)]
    )

    table = banded.groupby(["cost", "value_band"]).agg(
        pGO=("GO", "mean"),
        medRT=("RT", "median"),
        meanV=("value", "mean"),
        n=("GO", "size"),
    )
    print(table)
    return table
