from collections import Counter

import numpy as np
import pandas as pd
import pyddm

from initiation import T_DUR, build_model
from persistence import neg_log_likelihood

ROW_FLOOR = 400
# `value` already carries the prediction error out of the last set, so the conditions
# are the whole of the drift (initiation.initiation_drift).
CONDITIONS = ("value", "cost")
# Fitted non-decision time. The upper end stays under the fastest decision in the data.
T0_BOUNDS = (0.0, 0.5)


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
