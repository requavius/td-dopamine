from collections import Counter

import numpy as np
import pandas as pd
import pyddm

from initiation import build_model

ROW_FLOOR = 400


def passages_to_frame(passages):
    rows = [
        (p["value"], p["cost"], p["GO"], p["RT"])
        for p in passages
        if p["GO"] is not None
    ]
    undecided = Counter(
        (p["value"], p["cost"]) for p in passages if p["GO"] is None
    )

    df = pd.DataFrame(rows, columns=["value", "cost", "GO", "RT"])
    df.attrs["undecided"] = dict(undecided)
    return df


def _add_undecided(sample, undecided):
    empty = np.array([])
    for (value, cost), count in undecided.items():
        if count <= 0:
            continue
        sample = sample + pyddm.Sample(
            choice_upper=empty,
            choice_lower=empty,
            undecided=count,
            value=(empty, empty, np.repeat(value, count)),
            cost=(empty, empty, np.repeat(cost, count)),
        )
    return sample


def build_sample(df):
    sample = pyddm.Sample.from_pandas_dataframe(
        df, rt_column_name="RT", choice_column_name="GO"
    )
    return _add_undecided(sample, df.attrs.get("undecided", {}))


def fit(df):
    model = build_model({"f_val": (0.0, 3.0), "k_val": (0.0, 3.0)})
    model.fit(build_sample(df), verbose=False)

    drift_params = model.parameters()["drift"]
    return float(drift_params["f_val"]), float(drift_params["k_val"])


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

    def band(group):
        cut = group["value"].median()
        # Fall back to the mean when the median sits on the modal value, which
        # would otherwise put everything on one side of the split.
        if (group["value"] >= cut).all():
            cut = group["value"].mean()
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
