"""Profile-likelihood intervals for the five-parameter recovery, fits at a bound included.

experiments.recovery_lhc gives each fit standard errors from the observed information
(the Hessian at the estimate). Those make symmetric (Wald) intervals, and they have two
problems there: T0's covered the truth in 89.2% of fits where 95% is nominal, and the 7
fits that sat at a parameter bound (6 at T0 = 0, one at γ = 0.5) have no valid Hessian,
so they were left out.

A profile-likelihood interval needs neither. For parameter j, the profile is
P_j(x) = min over the other four of -log L with θ_j = x. The 95% interval is every x with
P_j(x) ≤ -log L at the estimate + 1.92 (Wilks: twice the drop is χ² with 1 df), clipped
to the parameter's range, so a fit at T0 = 0 gets an interval [0, upper]. It can be
asymmetric, as an interval near a bound should be.

How it is computed. The same 100 histories and likelihood as recovery_lhc (closed form,
γ continuous by a cubic spline), starting from its estimates (results/recovery_lhc/
fits.csv). From the estimate, x steps outward by STEP_SE times the parameter's observed SE
(or FALLBACK when there is none), up to the parameter's range; at each step the other four
are re-fitted by L-BFGS-B from the previous step's solution. The crossing of the 1.92 line
is placed by linear interpolation between the last two steps. If a profile point falls
below the estimate's -log L, the lowest value found is used as the minimum instead, and
the run is counted.

Claims, fixed before the run (2026-10-28):
    Q1  for every parameter, the profile intervals contain the truth in 90% to 98% of all
        100 fits, the 7 at a bound included
    Q2  T0's profile intervals contain the truth in at least 90% of the 93 fits away from
        a bound (its Wald intervals: 89.2%)
    Q3  none: report the median width against the Wald interval's (2 × 1.96 × SE)

    python -m experiments.lhc_profile
    python -m experiments.lhc_profile --replot
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .common import RESULTS, args, design, out_dir, run_pool
from .recovery_lhc import BOUNDS, NAMES, Person

# `histories`: the results folder of the recovery_lhc run whose histories and fits are read.
D = design("lhc_profile", histories="recovery_lhc")
OUT = out_dir(D.out)
LHC = RESULTS / D.histories
LR_95 = 1.92
STEP_SE = 0.4
FALLBACK = np.array([0.02, 0.02, 0.01, 0.02, 0.01])
MAX_STEPS = 40


def _profile_side(p, est, nll0, j, step, direction):
    """Walk parameter j from the estimate in `direction` until -log L rises by LR_95 or the
    range ends. Returns (limit, lowest -log L seen)."""
    lo, hi = BOUNDS[j]
    others = [m for m in range(5) if m != j]
    th = est.copy()
    prev_x, prev_y = est[j], nll0
    lowest = nll0
    for _ in range(MAX_STEPS):
        x = float(np.clip(prev_x + direction * step, lo, hi))
        th[j] = x

        def loss(z, th=th):
            t = th.copy()
            t[others] = z
            return p.nll(t)
        res = minimize(loss, th[others], method="L-BFGS-B", bounds=[BOUNDS[m] for m in others])
        th[others] = res.x
        y = float(res.fun)
        lowest = min(lowest, y)
        if y - nll0 >= LR_95:
            frac = (nll0 + LR_95 - prev_y) / (y - prev_y) if y != prev_y else 1.0
            return prev_x + frac * (x - prev_x), lowest
        if x in (lo, hi):
            return x, lowest
        prev_x, prev_y = x, y
    return prev_x, lowest


def _one(task):
    i, est, se_obs = task
    p = Person(i, np.load(LHC / "histories" / f"p{i:03d}.npz"))
    nll0 = p.nll(est)
    row = dict(person=i, nll=nll0)
    lowest = nll0
    for j, n in enumerate(NAMES):
        step = STEP_SE * se_obs[j] if np.isfinite(se_obs[j]) and se_obs[j] > 0 else FALLBACK[j]
        lo, low_l = _profile_side(p, est, nll0, j, step, -1)
        hi, low_h = _profile_side(p, est, nll0, j, step, +1)
        lowest = min(lowest, low_l, low_h)
        row.update({f"lo_{n}": lo, f"hi_{n}": hi})
    row["below_estimate"] = nll0 - lowest
    return row


def summarize(df):
    fits = pd.read_csv(LHC / "fits.csv").set_index("person")
    df = df.set_index("person").join(fits, rsuffix="_fit")
    edge = df[[f"at_bound_{n}" for n in NAMES]].any(axis=1)
    print(f"== profile-likelihood 95% intervals: {len(df)} fits, {edge.sum()} at a bound ==")
    print(f"{'':>6} {'cover, all':>11} {'cover, away':>12} {'Wald, away':>11} "
          f"{'median width':>13} {'Wald width':>11}")
    for n in NAMES:
        inside = (df[f"lo_{n}"] <= df[f"true_{n}"]) & (df[f"true_{n}"] <= df[f"hi_{n}"])
        wald = ((df[f"est_{n}"] - df[f"true_{n}"]).abs() < 1.96 * df[f"se_obs_{n}"])
        width = df[f"hi_{n}"] - df[f"lo_{n}"]
        wald_w = 2 * 1.96 * df[f"se_obs_{n}"]
        print(f"{n:>6} {inside.mean():>10.1%} {inside[~edge].mean():>12.1%} "
              f"{wald[~edge].mean():>11.1%} {width.median():>13.3f} {wald_w[~edge].median():>11.3f}")
    print("Q1 passes if every 'cover, all' is 90% to 98%; Q2 if T0's 'cover, away' is ≥ 90%.")
    print(f"fits where profiling found a lower -log L than the estimate: "
          f"{(df.below_estimate > 1e-3).sum()} (largest {df.below_estimate.max():.3f})")
    for n in NAMES:
        bad = df[~((df[f"lo_{n}"] <= df[f"true_{n}"]) & (df[f"true_{n}"] <= df[f"hi_{n}"]))]
        if len(bad):
            print(f"  {n} missed: " + ", ".join(
                f"{i} (true {r[f'true_{n}']:.3f}, [{r[f'lo_{n}']:.3f}, {r[f'hi_{n}']:.3f}])"
                for i, r in bad.iterrows()))


if __name__ == "__main__":
    a = args(__doc__, D)
    if a.replot:
        df = pd.read_csv(OUT / "intervals.csv")
    else:
        fits = pd.read_csv(LHC / "fits.csv")
        tasks = [(int(r.person), np.array([r[f"est_{n}"] for n in NAMES]),
                  np.array([r[f"se_obs_{n}"] for n in NAMES])) for _, r in fits.iterrows()]
        df = run_pool(_one, tasks, OUT / "intervals.csv").sort_values("person")
        df.to_csv(OUT / "intervals.csv", index=False)
    summarize(df)
