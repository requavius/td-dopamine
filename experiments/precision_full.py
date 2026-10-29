"""How precisely f and k are measured once γ, the bound and T0 are free as well.

experiments.precision gives the expected SD of f and k from the Fisher information with
only f and k free. Here the same quantity is computed with all five parameters free, from
the 100 saved histories of experiments.recovery_lhc (no simulation): the app's task under
InfoController, 300 start decisions plus their redo decisions per person. For each person
the Fisher information at the truth (recovery_lhc.Person.fisher: choice × 0.01 s RT bin and
undecided, closed-form likelihood) is computed from start decisions only, and from start
and redo decisions together, and inverted with
    2 free   f and k; γ, BOUND and T0 held at the truth
    5 free   f, k, γ, BOUND and T0
SD = sqrt(diag(I⁻¹)). Medians and 5th to 95th percentiles over the 100 people are
reported. No claim was fixed before the run; the numbers are descriptive.

Not the same design as experiments.precision, whose people play a fixed setting with f, k
on the 4 × 4 grid and γ 0.9: here settings are chosen by InfoController, and γ, BOUND
and T0 vary across people. α 0.05, SIGMA 0.7, ACHIEVEMENT_SCALE 4, continuing bootstrap,
T_DUR 7.0.

    python -m experiments.precision_full            # compute, then summarize
    python -m experiments.precision_full --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from .common import RESULTS, args, design, out_dir, run_pool
from .recovery_lhc import NAMES, Person

# `histories`: the results folder of the recovery_lhc run whose histories are read.
D = design("precision_full", histories="recovery_lhc")
OUT = out_dir(D.out)
HIST = RESULTS / D.histories / "histories"


def _sds(M, free):
    try:
        cov = np.linalg.inv(M[np.ix_(free, free)])
    except np.linalg.LinAlgError:
        return [np.nan] * len(free)
    d = np.diag(cov)
    return list(np.sqrt(d)) if np.all(d > 0) else [np.nan] * len(free)


def one(i):
    p = Person(i, np.load(HIST / f"p{i:03d}.npz"))
    row = dict(person=i, starts=int((p.node == 0).sum()), redos=int(p.node.sum()),
               **{f"true_{n}": p.truth[j] for j, n in enumerate(NAMES)})
    full = (p.go, p.rt, p.cost, p.node, p.value)
    for data, keep in (("starts", p.node == 0), ("both", np.ones(len(p.go), bool))):
        p.go, p.rt, p.cost, p.node = (x[keep] for x in full[:4])
        p.value = lambda g, v=full[4], keep=keep: v(g)[keep]
        M = p.fisher(p.truth)
        p.go, p.rt, p.cost, p.node, p.value = full
        for label, free in (("2", [0, 1]), ("5", [0, 1, 2, 3, 4])):
            for j, sd in zip(free, _sds(M, free)):
                row[f"sd_{NAMES[j]}_{data}_{label}"] = sd
    return row


def summarize(df):
    def q(col):
        x = df[col].dropna()
        lo, mid, hi = np.percentile(x, [5, 50, 95])
        return f"{mid:.3f} [{lo:.3f}, {hi:.3f}]"
    print(f"== expected SD from the Fisher information at the truth, {len(df)} people, "
          f"median [5th, 95th percentile] ==")
    print(f"{'decisions':<22} {'free':<6} {'SD of f':<24} {'SD of k':<24} {'SD of γ':<24}")
    for data, name in (("starts", "start decisions"), ("both", "starts and redos")):
        for label in ("2", "5"):
            g = q(f"sd_g_{data}_5") if label == "5" else "-"
            print(f"{name:<22} {label:<6} {q(f'sd_f_{data}_{label}'):<24} "
                  f"{q(f'sd_k_{data}_{label}'):<24} {g:<24}")
    print(f"\nstart decisions per person: {df.starts.median():.0f}; redo decisions: "
          f"{df.redos.median():.0f} median ({df.redos.min()} to {df.redos.max()})")
    for data in ("starts", "both"):
        r = (df[f"sd_k_{data}_5"] / df[f"sd_k_{data}_2"]).median()
        print(f"{data}: freeing γ, BOUND and T0 multiplies the SD of k by {r:.2f} (median)")


if __name__ == "__main__":
    a = args(__doc__, D)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        people = sorted(int(h.stem[1:]) for h in HIST.glob("p*.npz"))
        df = run_pool(one, people, OUT / "runs.csv")
    summarize(df)
