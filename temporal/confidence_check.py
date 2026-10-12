"""Checks for modelling confidence (model/confidence.py), with pass conditions fixed here.

    python -m temporal.confidence_check recovery   # step 2: can it be estimated?
    python -m temporal.confidence_check compare    # step 3: does the controller decide better?

recovery: simulated people with known confidence play 300 start decisions under a fixed
setting with redo prompts (the kind the measuring controller chooses). Pass: people with
none are called "none" (P(none) >= 0.5) at least 95% of the time; for eta = 1 people of
a modelled rule, the right hypothesis gets the most weight at least 80% of the time.
f and k come from the unchanged fit, so their recovery is the current pipeline's.

compare: InfoController with and without confidence on the same people and seeds, 300
start decisions, nothing given; the 20-setting subset of the regression test and a refit
every 150 decided rows (COMPARE_REFIT), to keep it to about an hour. Pass: reward with confidence - without > 2 SE (paired);
people with none not worse beyond 2 SE; f, k RMSE not worse beyond 2 SE.
"""
import argparse
import math
from multiprocessing import Pool

import pandas as pd

from .controller import (CANDIDATES, Environment, InfoController, confidence_posterior,
                         estimate, run)
from .model import confidence as conf
from .model.params import UserParams

PEOPLE_FK = ((0.6, 1.0), (1.0, 0.6), (0.8, 0.8))
TYPES = {"none": conf.NONE, "elo 0.5": ("elo", 0.5, 0.75), "elo 1.0": ("elo", 1.0, 0.75),
         "count 1.0/0.65": ("count", 1.0, 0.65), "count 0.5/0.85": ("count", 0.5, 0.85),
         "asym 1.0 (unmodelled)": ("asym", 1.0, 0.75)}
REDO_ENV = Environment(0.6, 0.0, True, False, True)
# The regression test's subset: 8 scheduled settings and every redo-prompt one.
SUBSET = CANDIDATES[::12] + CANDIDATES[-12:]
# Refit less often than the app (15) so a comparison run takes an hour, not four; both
# controllers use it.
COMPARE_REFIT = 150


def _params(f, k, h):
    return UserParams(f=f, k=k, eta=h[1], conf_rule=h[0], conf_ref=h[2])


def _recover(args):
    name, h, f, k, seed = args
    sim = run(_params(f, k, h), env=REDO_ENV, n_decisions=300, seed=seed)
    est = estimate(sim.state)
    post = confidence_posterior(sim.state, est)
    best = max(post, key=lambda x: x[1])[0]
    return {"type": name, "f": f, "k": k, "seed": seed, "f_hat": est.f, "k_hat": est.k,
            "p_none": next(w for hh, w, _ in post if hh == conf.NONE),
            "map_right": best == h, "skips": sim.state.skips,
            "redos": len(sim.state.first_passage)}


def recovery(seeds=4):
    jobs = [(n, h, f, k, 100 + s) for n, h in TYPES.items() for f, k in PEOPLE_FK
            for s in range(seeds)]
    with Pool() as pool:
        rows = pool.map(_recover, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv("confidence_recovery.csv", index=False)
    print(df.groupby("type").agg(n=("seed", "size"), p_none=("p_none", "mean"),
                                 called_none=("p_none", lambda x: (x >= 0.5).mean()),
                                 map_right=("map_right", "mean"),
                                 f_err=("f_hat", "mean"), skips=("skips", "mean"))
          .to_string())
    none = df[df.type == "none"]
    ok_none = (none.p_none >= 0.5).mean() >= 0.95
    strong = df[df.type.isin(["elo 1.0", "count 1.0/0.65"])]
    ok_strong = strong.map_right.mean() >= 0.80
    print(f"\nnone called none: {(none.p_none >= 0.5).mean():.2f} (pass >= 0.95): "
          f"{'PASS' if ok_none else 'FAIL'}")
    print(f"eta=1 modelled, right hypothesis: {strong.map_right.mean():.2f} (pass >= 0.80): "
          f"{'PASS' if ok_strong else 'FAIL'}")
    df["f_sq"] = (df.f_hat - df.f) ** 2
    df["k_sq"] = (df.k_hat - df.k) ** 2
    print(df.groupby("type")[["f_sq", "k_sq"]].mean().pow(0.5).rename(
        columns={"f_sq": "f RMSE", "k_sq": "k RMSE"}).to_string())


def _compare(args):
    name, h, f, k, seed, with_conf = args
    ctl = InfoController(candidates=SUBSET, seed=seed, confidence=with_conf,
                         refit_every=COMPARE_REFIT)
    sim = run(_params(f, k, h), controller=ctl, n_decisions=300, seed=seed)
    est = ctl.est
    p_none = (next(w for hh, w, _ in est.confidence if hh == conf.NONE)
              if est is not None and est.confidence else float("nan"))
    return {"type": name, "f": f, "k": k, "seed": seed, "conf": with_conf,
            "reward": sim.state.total_reward, "sets": sim.state.episodes,
            "f_hat": est.f if est else float("nan"), "k_hat": est.k if est else float("nan"),
            "p_none": p_none}


def se(x):
    return x.std(ddof=1) / math.sqrt(len(x))


def compare(seeds=2):
    jobs = [(n, h, f, k, 200 + s, wc) for n, h in TYPES.items() for f, k in PEOPLE_FK
            for s in range(seeds) for wc in (False, True)]
    with Pool() as pool:
        rows = pool.map(_compare, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv("confidence_compare.csv", index=False)
    key = ["type", "f", "k", "seed"]
    a = df[~df.conf].set_index(key)
    b = df[df.conf].set_index(key)
    d = (b.reward - a.reward)
    print(f"reward without {a.reward.mean():.1f}, with {b.reward.mean():.1f}, "
          f"paired diff {d.mean():+.1f} ± {se(d):.1f}  ({len(d)} people)")
    for t, g in d.groupby(level="type"):
        print(f"  {t:24s} {g.mean():+7.1f} ± {se(g):.1f}  (n={len(g)})")
    dn = d.xs("none", level="type")
    fa = (a.f_hat - a.index.get_level_values("f")) ** 2
    fb = (b.f_hat - b.index.get_level_values("f")) ** 2
    ka = (a.k_hat - a.index.get_level_values("k")) ** 2
    kb = (b.k_hat - b.index.get_level_values("k")) ** 2
    print(f"\nP1 reward gain > 2 SE: {'PASS' if d.mean() > 2 * se(d) else 'FAIL'}")
    print(f"P2 none not worse beyond 2 SE: {dn.mean():+.1f} ± {se(dn):.1f}: "
          f"{'PASS' if dn.mean() > -2 * se(dn) else 'FAIL'}")
    for nm, x, y in (("f", fa, fb), ("k", ka, kb)):
        dd = y - x
        print(f"P3 {nm} squared error with - without {dd.mean():+.4f} ± {se(dd):.4f}: "
              f"{'PASS' if dd.mean() < 2 * se(dd) else 'FAIL'}"
              f"  (RMSE {math.sqrt(x.mean()):.3f} -> {math.sqrt(y.mean()):.3f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("check", choices=("recovery", "compare"))
    p.add_argument("--seeds", type=int, default=None)
    a = p.parse_args()
    if a.check == "recovery":
        recovery(a.seeds or 4)
    else:
        compare(a.seeds or 2)


if __name__ == "__main__":
    main()
