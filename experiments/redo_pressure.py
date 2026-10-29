"""Time pressure on the redo prompt: can the person's response to it be measured?

Under time pressure people lower their response threshold (Forstmann et al., 2008). A
pressured redo prompt (temporal/model/pressure.py) shows a 3 s countdown and closes,
without a redo, if the person has not chosen by then. Their bound on it is
`pressure` × BOUND, with `pressure` (λ) theirs; their drift is unchanged. λ is a new
parameter of the person: how much caution they give up when hurried. Fitting it needs
redo decisions with and without the countdown, which share a drift and differ in bound,
as in a speed-accuracy design.

People: a seeded Latin hypercube over f and k 0.2 to 1.5, γ 0.6 to 0.97, BOUND 1.0 to
2.0, T0 0 to 0.4 s and λ 0.4 to 1.0. Each plays the app's task (question types with Elo
ratings, skips on, InfoController steering as the app runs it) for 300 start decisions
plus their redo decisions, twice on the same seed:
    none     no countdown, as now
    blocks   the countdown on every redo prompt in alternate blocks of 25 start
             decisions, starting without it (the controller does not know of it)
α 0.05; SIGMA 0.7, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s without the
countdown and 3.0 s with it. Value at any γ from shadow learners, as in
experiments.recovery_lhc. Fits, by maximum likelihood with the closed-form likelihood
(differential evolution, then L-BFGS-B):
    none     f, k, γ, BOUND, T0 (recovery_lhc.Person.fit)
    blocks   the same and λ: a countdown decision has bound λ · BOUND and deadline 3 s
    blocks, λ ignored   five parameters, a countdown decision keeps BOUND and is read
             with its 3 s deadline: what a fit that knows the countdown but not the
             person's response would give

Claims, fixed before any run (2026-10-28):
    T1  in the blocks arm λ is recovered: r(λ̂, λ) at least 0.8 and RMSE at most 0.10
    T2  squared error in k, blocks − none, paired, is not above zero beyond 2 SE
    T3  squared error in BOUND, blocks − none, paired, is below zero beyond 2 SE
    T4  none: f, γ and T0 the same way; the bias with λ ignored; redo decisions, the
        share taken and the share timed out, by λ

    python -m experiments.redo_pressure --people 0-31 --tag server
    python -m experiments.redo_pressure --replot
"""
import argparse
import random

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from temporal import fastlik
from temporal.controller import InfoController, apply
from temporal.model import accumulator as acc
from temporal.model.params import UserParams
from temporal.model.pressure import REDO_DEADLINE
from temporal.model.questions import Ratings, synthetic_type_diffs

from .common import Quiet, _design_args, design, out_dir, run_pool, save_conditions, se
from .gamma_profile import CACHE_CAP, GridShadow
from .misspecification import PersonConstants
from .recovery_lhc import BOUNDS, G_FINE, NAMES, Person, span

# `ranges` in the hypercube's order; "lam" is λ, the share of BOUND kept under a countdown.
D = design("redo_pressure", people=64, seed=20261032, decisions=300, block=25,
           deadline=REDO_DEADLINE,
           ranges={"f": (0.2, 1.5), "k": (0.2, 1.5), "g": (0.6, 0.97), "bound": (1.0, 2.0),
                   "t0": (0.0, 0.4), "lam": (0.4, 1.0)})
OUT = out_dir(D.out)
N_PEOPLE = D.people
SEED = D.seed
N_STARTS = D.decisions
BLOCK = D.block
DEADLINE = D.deadline
ARMS = ("none", "blocks")
RANGES = D.ranges
NAMES6 = (*NAMES, "lam")
BOUNDS6 = [*BOUNDS, (0.2, 1.5)]


def people():
    unit = qmc.LatinHypercube(d=len(RANGES), seed=SEED).random(N_PEOPLE)
    lo = np.array([r[0] for r in RANGES.values()])
    hi = np.array([r[1] for r in RANGES.values()])
    return [dict(zip(RANGES, row)) for row in lo + unit * (hi - lo)]


class PressurePerson(Person):
    """Person, with redo decisions under a countdown read at bound λ · BOUND and a 3 s
    deadline. θ = (f, k, γ, BOUND, T0, λ); with `ignore_lam` λ is held at 1."""

    def __init__(self, i, h, pressured, ignore_lam=False):
        super().__init__(i, h=h)
        self.pressured = pressured
        self.ignore_lam = ignore_lam
        self.t_dur = np.where(pressured, DEADLINE, acc.T_DUR)

    def nll(self, th):
        lam = 1.0 if self.ignore_lam or len(th) < 6 else th[5]
        bound = th[3] * np.where(self.pressured, lam, 1.0)
        mu = th[0] * self.value(th[2]) - th[1] * self.cost
        return fastlik.neg_log_likelihood(self.go, self.rt, mu, bound, th[4], t_dur=self.t_dur)

    def fit6(self):
        res = differential_evolution(self.nll, BOUNDS6, seed=self.i, tol=1e-8, maxiter=2000,
                                     polish=False)
        res = minimize(self.nll, res.x, method="L-BFGS-B", bounds=BOUNDS6)
        return res.x, float(res.fun)


def simulate(arm, i):
    truth = people()[i]
    seed = SEED + i
    random.seed(seed)
    np.random.seed(seed)
    ctl = InfoController(seed=seed)
    start = ctl.env
    ratings = Ratings()
    person = PersonConstants(acc.SIGMA, truth["bound"])
    sim = GridShadow(UserParams(f=truth["f"], k=truth["k"], g=truth["g"], t0=truth["t0"],
                                pressure=truth["lam"]),
                     G_FINE, reentry_cost=0.6, review_rate=start.review_rate,
                     review_mode=start.review_mode, target_p=start.target_p,
                     feedback=start.feedback, ratings=ratings, skips=True,
                     return_skipped=start.return_skipped)
    st = sim.state
    st.type_diffs = np.array(synthetic_type_diffs(1.0), dtype=float)
    st.ability = 0.0
    st.in_session = True
    with person():
        sim.run_episode()
    while len(st.initiation_passages) < N_STARTS:
        with person():
            sim.tick()
        person.trim()
        if len(acc._SOLUTION_CACHE) > CACHE_CAP:
            acc.clear_caches()
        if ctl.due(st):
            ctl.plan(sim.app_view(ratings), UserParams(f=truth["f"], k=truth["k"]))
            apply(sim, ctl.env)
        on = arm == "blocks" and (len(st.initiation_passages) // BLOCK) % 2 == 1
        sim.redo_deadline = DEADLINE if on else None
    upto = st.initiation_passages[N_STARTS - 1]["timestep"]
    return truth, (st.initiation_passages[:N_STARTS]
                   + [r for r in st.first_passage if r["timestep"] <= upto])


def _history(records):
    node = np.array([p.get("node") == "stage" for p in records], dtype=np.int8)
    go = np.array([-1 if p["GO"] is None else p["GO"] for p in records], dtype=np.int8)
    rt = np.array([np.nan if p["RT"] is None else p["RT"] for p in records])
    parts = np.array([[p["parts_by_g"][g] for g in G_FINE] for p in records])
    pressured = np.array(["deadline" in p for p in records])
    h = dict(node=node, go=go, rt=rt, cost=np.array([p["cost"] for p in records]),
             base=parts[:, :, 0], delta=parts[:, :, 1], g_grid=np.array(G_FINE),
             **{f"true_{n}": 0.0 for n in NAMES})
    return h, pressured


def _one(task):
    arm, i = task
    acc.clear_caches()
    with Quiet():
        truth, records = simulate(arm, i)
        h, pressured = _history(records)
        redos = [p for p in records if p.get("node") == "stage"]
        under = [p for p in redos if "deadline" in p]
        row = dict(arm=arm, person=i, **{f"true_{n}": truth[n] for n in NAMES6},
                   decisions=len(records), redos=len(redos), pressured=len(under),
                   taken=float(np.mean([p["chose"] for p in redos])) if redos else np.nan,
                   taken_pressured=float(np.mean([p["chose"] for p in under])) if under else np.nan,
                   timed_out=float(np.mean([p["GO"] is None for p in under])) if under else np.nan,
                   p_go=float(np.mean([p["chose"] for p in records if p.get("node") != "stage"])))
        if arm == "none":
            est, nll = Person(i, h=h).fit()
            row.update({f"est_{n}": est[j] for j, n in enumerate(NAMES)}, nll=nll)
        else:
            est, nll = PressurePerson(i, h, pressured).fit6()
            row.update({f"est_{n}": est[j] for j, n in enumerate(NAMES6)}, nll=nll)
            est5, nll5 = PressurePerson(i, h, pressured, ignore_lam=True).fit()
            row.update({f"ign_{n}": est5[j] for j, n in enumerate(NAMES)}, nll_ignored=nll5)
    return row


def summarize(df):
    none = df[df.arm == "none"].set_index("person").sort_index()
    blk = df[df.arm == "blocks"].set_index("person").sort_index()
    both = none.index.intersection(blk.index)
    none, blk = none.loc[both], blk.loc[both]
    print(f"== a countdown on the redo prompt: {len(both)} paired people, {N_STARTS} start "
          f"decisions plus redos ==")
    err = blk.est_lam - blk.true_lam
    r = np.corrcoef(blk.est_lam, blk.true_lam)[0, 1]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    print(f"T1 λ: RMSE {rmse:.3f}, bias {err.mean():+.3f} ± {se(err):.3f}, r {r:.3f} "
          f"({'holds' if r >= 0.8 and rmse <= 0.10 else 'does not hold'})")
    print(f"\n{'':>6} {'RMSE none':>10} {'RMSE blocks':>12} {'Δ sq. error (blocks − none)':>30} "
          f"{'bias, λ ignored':>18}")
    for n in NAMES:
        e0, e1 = none[f"est_{n}"] - none[f"true_{n}"], blk[f"est_{n}"] - blk[f"true_{n}"]
        d = e1 ** 2 - e0 ** 2
        ign = (blk[f"ign_{n}"] - blk[f"true_{n}"]) / blk[f"true_{n}"].where(blk[f"true_{n}"] > 0.02) * 100
        print(f"{n:>6} {np.sqrt(np.mean(e0 ** 2)):10.3f} {np.sqrt(np.mean(e1 ** 2)):12.3f} "
              f"{d.mean():+20.4f} ± {se(d):.4f} {ign.mean():+12.1f} ± {se(ign.dropna()):.1f}%")
    for claim, n, below in (("T2", "k", False), ("T3", "bound", True)):
        d = ((blk[f"est_{n}"] - blk[f"true_{n}"]) ** 2 - (none[f"est_{n}"] - none[f"true_{n}"]) ** 2)
        ok = d.mean() < -2 * se(d) if below else d.mean() < 2 * se(d)
        print(f"{claim} {n}: {d.mean():+.4f} ± {se(d):.4f} ({'holds' if ok else 'does not hold'})")
    print(f"λ ignored, -log L worse than with λ by: median {(blk.nll_ignored - blk.nll).median():.1f}")
    print("\nby true λ (blocks arm): redos, share taken under the countdown, share timed out")
    bins = pd.cut(blk.true_lam, [0.4, 0.55, 0.7, 0.85, 1.0], include_lowest=True)
    for b, g in blk.groupby(bins, observed=True):
        print(f"  λ {b}: {g.redos.mean():.0f} redos ({g.pressured.mean():.0f} under the countdown), "
              f"taken {g.taken_pressured.mean():.2f}, timed out {g.timed_out.mean():.2f} "
              f"({len(g)} people)")
    print(f"redos per person, none / blocks: {none.redos.mean():.0f} / {blk.redos.mean():.0f}; "
          f"share taken {none.taken.mean():.2f} / {blk.taken.mean():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="merge and summarize the saved results")
    p.add_argument("--people", type=span, default=list(range(N_PEOPLE)), help="for example 0-31")
    p.add_argument("--tag", default="local", help="names this machine's file, runs_<tag>.csv")
    _design_args(p)
    a = p.parse_args()
    if not a.replot:
        save_conditions(D)
        tasks = [(arm, i) for i in a.people for arm in ARMS]
        run_pool(_one, tasks, OUT / f"runs_{a.tag}.csv", fresh=True,
                 label=lambda r: f"{r['arm']} person {r['person']}")
    files = sorted(OUT.glob("runs_*.csv"))
    summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
