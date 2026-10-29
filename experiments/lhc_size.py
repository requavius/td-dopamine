"""How much data the five-parameter fit needs, and what pooling people buys.

experiments.recovery_lhc fits f, k, γ, BOUND and T0 from 300 start decisions plus their
redo decisions (722 decisions on average). A real person may give far fewer. This reads
the same 100 saved histories and refits each from part of its data, and then fits all 100
people together with a population distribution.

Part of the data. The saved histories hold the 300 start decisions in order, then the
redo decisions in order, without the time of each, so where the N-th start falls among
the redos is not known. A history cut at N start decisions keeps the first N starts and
the first round(N / 300 × redos) redo decisions. N is 50, 100, 200 and 300.

Two estimates at each N:
    ML       each person alone: maximum likelihood (recovery_lhc.Person.fit)
    pooled   empirical Bayes. Parameters are mapped to an unbounded scale (log f, log k,
             logit((γ − 0.5) / 0.49), log BOUND, logit(T0 / 0.5)), where the 100 people
             are taken to be normal with mean m and covariance S. m and S are fitted by
             expectation-maximization with a Laplace approximation (as in Huys et al.,
             2011): each person's MAP under N(m, S) and the inverse Hessian there, Σ_i;
             then m = mean of the MAPs and S = mean of Σ_i + (MAP_i − m)(MAP_i − m)ᵀ;
             repeated until m moves less than 0.001 (at most MAX_ITER rounds). The pooled
             estimate is each person's MAP, its 95% interval MAP ± 1.96 √Σ_i on the
             unbounded scale, mapped back.
The people were drawn uniformly from a box, so the normal population is itself an
approximation, as it would be for real people. The population is fitted to the same people
it is used on, as it would be for a sample of participants.

Claims, fixed before the run (2026-10-28), paired over the 100 people:
    H1  at N = 100, squared error pooled − ML is below zero beyond 2 SE for f
    H2  the same for k
    H3  at N = 100, the pooled 95% intervals contain the truth in 90% to 98% of people,
        for every parameter
    H4  none: RMSE and r of both estimates at every N, and the pooled coverage at every N

    python -m experiments.lhc_size
    python -m experiments.lhc_size --replot
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .common import RESULTS, WORKERS, args, design, out_dir, run_pool, se
from .recovery_lhc import NAMES, Person

# `histories`: the results folder of the recovery_lhc run whose histories are read.
D = design("lhc_size", histories="recovery_lhc", sizes=(50, 100, 200, 300))
OUT = out_dir(D.out)
HIST = RESULTS / D.histories / "histories"
SIZES = D.sizes
MAX_ITER = 40
H_X = 0.01  # finite-difference step on the unbounded scale
EPS = 1e-4


def to_x(th):
    f, k, g, b, t0 = th
    u = np.clip((g - 0.5) / 0.49, EPS, 1 - EPS)
    v = np.clip(t0 / 0.5, EPS, 1 - EPS)
    return np.array([np.log(max(f, EPS)), np.log(max(k, EPS)), np.log(u / (1 - u)),
                     np.log(b), np.log(v / (1 - v))])


def to_th(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return np.array([np.exp(x[0]), np.exp(x[1]), 0.5 + 0.49 * s[2], np.exp(x[3]), 0.5 * s[4]])


def cut(i, n):
    h = dict(np.load(HIST / f"p{i:03d}.npz"))
    node = h["node"].astype(bool)
    starts, redos = np.flatnonzero(~node), np.flatnonzero(node)
    keep = np.concatenate([starts[:n], redos[:int(round(n / len(starts) * len(redos)))]])
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(node) else v)
           for k, v in h.items()}
    return Person(i, h=out)


def _ml(task):
    i, n = task
    p = cut(i, n)
    est, nll = p.fit()
    return dict(person=i, n=n, rows=len(p.go), nll=nll,
                **{f"true_{m}": p.truth[j] for j, m in enumerate(NAMES)},
                **{f"ml_{m}": est[j] for j, m in enumerate(NAMES)})


def _hessian(fn, x, h=H_X):
    d = len(x)
    H = np.zeros((d, d))
    f0 = fn(x)
    e = np.eye(d) * h
    for a in range(d):
        H[a, a] = (fn(x + e[a]) - 2 * f0 + fn(x - e[a])) / h ** 2
        for b in range(a):
            H[a, b] = H[b, a] = (fn(x + e[a] + e[b]) - fn(x + e[a] - e[b]) - fn(x - e[a] + e[b])
                                 + fn(x - e[a] - e[b])) / (4 * h * h)
    return H


# One person's E-step: MAP under N(m, S) on the unbounded scale, and its inverse Hessian.
def _map(task):
    i, n, x0, m, S_inv = task
    p = cut(i, n)

    def loss(x):
        d = x - m
        return p.nll(to_th(x)) + 0.5 * float(d @ S_inv @ d)
    res = minimize(loss, x0, method="L-BFGS-B")
    H = _hessian(loss, res.x)
    w, V = np.linalg.eigh(H)
    cov = (V / np.maximum(w, 1e-6)) @ V.T  # a non-positive direction gets a wide variance
    return i, res.x, cov


def pooled(n, ml):
    people = list(ml.index)
    x = np.array([to_x(ml.loc[i, [f"ml_{m}" for m in NAMES]].to_numpy(float)) for i in people])
    m = x.mean(axis=0)
    S = np.cov(x.T) + np.eye(5) * 0.01
    with Pool(WORKERS) as pool:
        for it in range(MAX_ITER):
            S_inv = np.linalg.inv(S)
            res = pool.map(_map, [(i, n, x[j], m, S_inv) for j, i in enumerate(people)])
            x = np.array([r[1] for r in res])
            covs = np.array([r[2] for r in res])
            m_new = x.mean(axis=0)
            S = covs.mean(axis=0) + (x - m_new).T @ (x - m_new) / len(x)
            moved = float(np.abs(m_new - m).max())
            m = m_new
            print(f"N {n}, round {it + 1}: the population mean moved {moved:.4f}", flush=True)
            if moved < 1e-3:
                break
    rows = []
    for j, i in enumerate(people):
        sd = np.sqrt(np.diag(covs[j]))
        po, lo, hi = to_th(x[j]), to_th(x[j] - 1.96 * sd), to_th(x[j] + 1.96 * sd)
        rows.append(dict(person=i, rounds=it + 1, **{f"po_{m_}": po[k] for k, m_ in enumerate(NAMES)},
                         **{f"lo_{m_}": lo[k] for k, m_ in enumerate(NAMES)},
                         **{f"hi_{m_}": hi[k] for k, m_ in enumerate(NAMES)}))
    pop = dict(n=n, **{f"mean_{m_}": v for m_, v in zip(NAMES, to_th(m))},
               **{f"sd_x_{m_}": v for m_, v in zip(NAMES, np.sqrt(np.diag(S)))})
    return pd.DataFrame(rows).set_index("person"), pop


def run(people):
    ml = run_pool(_ml, [(i, n) for n in SIZES for i in people], OUT / "ml.csv")
    parts, pops = [], []
    for n in SIZES:
        g = ml[ml.n == n].set_index("person").sort_index()
        po, pop = pooled(n, g)
        parts.append(g.join(po))
        pops.append(pop)
    pd.DataFrame(pops).to_csv(OUT / "population.csv", index=False)
    df = pd.concat(parts).reset_index()
    df.to_csv(OUT / "fits.csv", index=False)
    return df


def summarize(df):
    print(f"== the five-parameter fit from part of the data: {df.person.nunique()} people ==")
    for n, g in df.groupby("n"):
        print(f"\nN = {n} start decisions ({g.rows.mean():.0f} decisions on average)")
        print(f"{'':>6} {'ML RMSE':>8} {'pooled RMSE':>12} {'ML r':>6} {'pooled r':>9} "
              f"{'pooled cover':>13} {'Δ sq. error (pooled − ML)':>27}")
        for m in NAMES:
            e_ml = g[f"ml_{m}"] - g[f"true_{m}"]
            e_po = g[f"po_{m}"] - g[f"true_{m}"]
            d = e_po ** 2 - e_ml ** 2
            cover = ((g[f"lo_{m}"] <= g[f"true_{m}"]) & (g[f"true_{m}"] <= g[f"hi_{m}"])).mean()
            print(f"{m:>6} {np.sqrt(np.mean(e_ml ** 2)):8.3f} {np.sqrt(np.mean(e_po ** 2)):12.3f} "
                  f"{np.corrcoef(g[f'ml_{m}'], g[f'true_{m}'])[0, 1]:6.3f} "
                  f"{np.corrcoef(g[f'po_{m}'], g[f'true_{m}'])[0, 1]:9.3f} {cover:13.0%} "
                  f"{d.mean():+15.4f} ± {se(d):.4f}")
    g = df[df.n == 100]
    for claim, m in (("H1", "f"), ("H2", "k")):
        d = (g[f"po_{m}"] - g[f"true_{m}"]) ** 2 - (g[f"ml_{m}"] - g[f"true_{m}"]) ** 2
        print(f"{claim} ({m}, N = 100): {d.mean():+.4f} ± {se(d):.4f}, "
              f"{'holds' if d.mean() < -2 * se(d) else 'does not hold'}")
    cover = [((g[f"lo_{m}"] <= g[f"true_{m}"]) & (g[f"true_{m}"] <= g[f"hi_{m}"])).mean()
             for m in NAMES]
    print(f"H3 (N = 100, pooled coverage 90 to 98% for every parameter): "
          f"{', '.join(f'{m} {c:.0%}' for m, c in zip(NAMES, cover))}; "
          f"{'holds' if all(0.9 <= c <= 0.98 for c in cover) else 'does not hold'}")


if __name__ == "__main__":
    a = args(__doc__, D)
    if a.replot:
        summarize(pd.read_csv(OUT / "fits.csv"))
    else:
        people = sorted(int(h.stem[1:]) for h in HIST.glob("p*.npz"))
        summarize(run(people))
