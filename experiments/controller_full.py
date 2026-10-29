"""Planning with fitted γ and bound: a controller that measures all five parameters.

InfoController, the app's controller, fits f, k and T0 with γ held at 0.9 and BOUND at
1.5, rolls out settings with those values, and scores a setting by how much it would
sharpen (f, k). FullInfoController, here, does what README "Next steps" 8 asks:

- **Estimate.** f, k, γ, BOUND and T0 together, by maximum a posteriori with the
  closed-form likelihood (temporal.fastlik). Every recorded decision's value at any γ
  comes from shadow TD learners at α 0.05 on the rewards the model assumes, one per γ from
  0.50 to 0.99 in steps of 0.01 (gamma_profile.GridShadow), joined by a cubic spline, as
  in experiments.recovery_lhc. The prior is PRIOR_MEAN ± PRIOR_SDS, independent normals;
  the posterior is approximated as normal at the MAP (Laplace, finite-difference Hessian).
- **Rollouts.** Each of DRAWS draws from that posterior is rolled out as a person with its
  own γ (the shadow learner nearest it supplies the value weights and carried errors) and
  its own bound. Rollout decisions are drawn from the closed-form likelihood at that bound
  (RolloutBound), not from pyddm, so a new bound costs no Fokker-Planck solve.
- **Score.** The rise in log det of the Fisher information about all five parameters.
  One decision's outcome (choice, RT bin, undecided) depends on (μ, BOUND, T0), with
  μ = f·v(γ) − k·c, so its information about (f, k, γ, BOUND, T0) is Jᵀ I(μ, BOUND, T0) J
  with J's rows (v, −c, f·∂v/∂γ, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1). I comes from
  fastlik on a grid of μ, by central differences. ∂v/∂γ is from the spline for recorded
  decisions, and from two extra shadow learners at γ ± 0.02 in a rollout.
- The candidates, the 25-tick rollouts, the 6 draws, the safety floor, the plan every 25
  start decisions and the re-fit every 15 decided rows are InfoController's.

The experiment. 128 people from a seeded Latin hypercube over the ranges of
experiments.recovery_lhc (f and k 0.2 to 1.5, γ 0.6 to 0.97, BOUND 1.0 to 2.0, T0 0 to
0.4 s), a different seed from it. Each plays the app's task (question types with Elo
ratings, skips on, cold-start cost 0.6) for 300 ticks twice, with the same seed: once
steered by InfoController as the app runs it, once by FullInfoController. α 0.05 for
person, shadows and both controllers; SIGMA 0.7, ACHIEVEMENT_SCALE 4, continuing
bootstrap, T_DUR 7.0. At 100, 200 and 300 ticks the data recorded so far are fitted the
same way in both arms: all five parameters by maximum likelihood (recovery_lhc.Person.fit).

Claims, fixed before any run (2026-10-28), paired on person, at 300 ticks:
    F1  squared error in k, full − current, is below zero beyond 2 SE
    F2  the same for f
    F3  the same for γ
    F4  P(start next set), full − current, is not below zero beyond 2 SE
    F5  none: report BOUND and T0 errors, reward, time away, decisions, redo decisions,
        and the share of start decisions made after a set with feedback shown, by true γ

    python -m experiments.controller_full --people 0-63 --tag server
    python -m experiments.controller_full --replot
"""
import argparse
import contextlib
import copy
import dataclasses
import random

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.stats import qmc

from temporal import fastlik
from temporal.controller import (DRAWS, MIN_ROWS, MU_GRID, Estimate, InfoController, Update,
                                 apply, decided_rows, estimate_patience, logdet, measured)
from temporal.model import accumulator as acc
from temporal.model import persistence
from temporal.model.accumulator import snap_value
from temporal.model.params import UserParams
from temporal.model.questions import Ratings, synthetic_type_diffs

from . import recovery_lhc
from .common import Quiet, _design_args, design, out_dir, run_pool, save_conditions, se
from .gamma_profile import CACHE_CAP, GridShadow
from .misspecification import PersonConstants
from .recovery_lhc import BOUNDS, G_FINE, NAMES, Person, span

# The run lasts until the last checkpoint. The seed differs from recovery_lhc's (20261028).
D = design("controller_full", people=128, seed=20261029, checkpoints=(100, 200, 300),
           ranges=recovery_lhc.D.paper["ranges"])
OUT = out_dir(D.out)
N_PEOPLE = D.people
SEED = D.seed
RANGES = D.ranges
CHECKPOINTS = D.checkpoints
N_TICKS = CHECKPOINTS[-1]
ARMS = ("current", "full")

PRIOR_MEAN = np.array([0.8, 0.8, 0.85, 1.5, 0.2])  # f, k, γ, BOUND, T0
PRIOR_SDS = np.array([0.5, 0.5, 0.1, 0.3, 0.15])
DRAW_LO = np.array([0.05, 0.05, 0.5, 0.8, 0.0])
DRAW_HI = np.array([2.5, 2.5, 0.99, 2.5, 0.5])
H_G = 0.02         # γ step for ∂v/∂γ in a rollout
BOUND_STEP = 0.05  # rollout bounds are snapped to this, so their draw tables are reused
STEPS = np.array([0.01, 0.01, 0.005, 0.01, 0.005])


def people():
    unit = qmc.LatinHypercube(d=len(RANGES), seed=SEED).random(N_PEOPLE)
    lo = np.array([r[0] for r in RANGES.values()])
    hi = np.array([r[1] for r in RANGES.values()])
    return [dict(zip(RANGES, row)) for row in lo + unit * (hi - lo)]


class RolloutBound:
    """Decisions at a bound of the rollout's own, drawn from the closed-form likelihood.

    Inside `with self(bound):` accumulator._sampler builds its draw table from
    fastlik.outcome_probs instead of a pyddm solution, and the give-up process uses the same
    bound. Tables are cached per (f, k, value, cost, T0, bound) across plans."""

    CAP = 5000  # tables kept, about 22 kB each

    def __init__(self):
        self.cache = {}

    def _table(self, key, bound):
        f, k, value, cost, t0 = key
        drift = acc.initiation_drift(value, cost, f, k)
        up, lo, und = fastlik.outcome_probs([drift], bound, t0, dt=acc.DT)
        up, lo = up[0], lo[0]
        t = np.arange(len(up)) * acc.DT  # bin starts; _draw adds a uniform within the bin
        shift = t[-1] + 1
        domain = np.concatenate([t, t + shift, [-1]])
        probs = np.concatenate([up, lo, und])
        probs = probs / probs.sum()
        # Where an undecided draw ends up, with the lapses taken out (as _upper_after_deadline).
        up0, _, und0 = fastlik.outcome_probs([drift], bound, t0, lapse=0.0, dt=acc.DT)
        p_und = float(und0[0])
        after = 0.5 if p_und < 1e-12 else float(
            np.clip((acc._upper_eventually(drift) - up0.sum()) / p_und, 0.0, 1.0))
        return domain, probs, shift, acc.DT, after

    @contextlib.contextmanager
    def __call__(self, bound):
        saved = acc._sampler, acc.BOUND, persistence.BOUND
        if len(self.cache) > self.CAP:
            self.cache.clear()

        def sampler(key):
            k = (*key, bound)
            if k not in self.cache:
                self.cache[k] = self._table(key, bound)
            return self.cache[k]
        acc._sampler = sampler
        acc.BOUND = persistence.BOUND = bound
        try:
            yield
        finally:
            acc._sampler, acc.BOUND, persistence.BOUND = saved


# Fisher information of one decision about (μ, BOUND, T0), on MU_GRID: the six distinct
# entries of the 3 × 3 matrix, each an array over MU_GRID.
def info_table(bound, t0, h=(0.01, 0.01, 0.005)):
    def probs(mu, b, t):
        up, lo, und = fastlik.outcome_probs(mu, b, max(t, 0.0))
        return np.concatenate([up, lo, und[:, None]], axis=1)
    mid = probs(MU_GRID, bound, t0)
    d = [(probs(MU_GRID + h[0], bound, t0) - probs(MU_GRID - h[0], bound, t0)) / (2 * h[0]),
         (probs(MU_GRID, bound + h[1], t0) - probs(MU_GRID, bound - h[1], t0)) / (2 * h[1]),
         (probs(MU_GRID, bound, t0 + h[2]) - probs(MU_GRID, bound, max(t0 - h[2], 0.0)))
         / (h[2] + min(h[2], t0))]
    w = 1.0 / np.maximum(mid, 1e-300)
    return {(a, b): np.sum(d[a] * d[b] * w, axis=1) for a in range(3) for b in range(a, 3)}


# Information about (f, k, γ, BOUND, T0) in decisions with value v, cost c and ∂v/∂γ vg.
def information(v, c, vg, draw, table):
    f, k = draw[0], draw[1]
    mu = f * v - k * c
    info = {key: np.interp(mu, MU_GRID, arr) for key, arr in table.items()}
    a = np.stack([v, -c, f * vg])
    M = np.zeros((5, 5))
    M[:3, :3] = (a * info[0, 0]) @ a.T
    M[:3, 3] = M[3, :3] = a @ info[0, 1]
    M[:3, 4] = M[4, :3] = a @ info[0, 2]
    M[3, 3], M[3, 4], M[4, 4] = info[1, 1].sum(), info[1, 2].sum(), info[2, 2].sum()
    M[4, 3] = M[3, 4]
    return M


def _hessian(fn, x, steps=STEPS):
    n = len(x)
    H = np.zeros((n, n))
    f0 = fn(x)
    e = np.eye(n) * steps
    for i in range(n):
        H[i, i] = (fn(x + e[i]) - 2 * f0 + fn(x - e[i])) / steps[i] ** 2
        for j in range(i):
            H[i, j] = H[j, i] = (fn(x + e[i] + e[j]) - fn(x + e[i] - e[j]) - fn(x - e[i] + e[j])
                                 + fn(x - e[i] - e[j])) / (4 * steps[i] * steps[j])
    return H


class History:
    """Recorded decisions as arrays, with value at any γ from the shadow learners."""

    def __init__(self, records, grid):
        self.go = np.array([-1 if p["GO"] is None else p["GO"] for p in records])
        self.rt = np.array([np.nan if p["RT"] is None else p["RT"] for p in records])
        self.cost = np.array([p["cost"] for p in records], dtype=float)
        vals = np.array([[sum(p["parts_by_g"][g]) for g in grid] for p in records])
        self.value = CubicSpline(np.array(grid), vals.T, axis=0)

    def nll(self, th):
        mu = th[0] * self.value(th[2]) - th[1] * self.cost
        return fastlik.neg_log_likelihood(self.go, self.rt, mu, th[3], th[4])


class FullInfoController(InfoController):
    """InfoController with γ, BOUND and T0 estimated, drawn and scored (module docstring).
    `plan(state, shadow)`: `state` as measured, its records carrying `parts_by_g`;
    `shadow` the replay learners, γ → {"theta", "slot", "set"} (GridShadow.shadow)."""

    def __init__(self, grid=G_FINE, **kw):
        super().__init__(**kw)
        self.grid = tuple(grid)
        self.map = None
        self.cov = None
        self.bounds = RolloutBound()
        self.tables = {}

    def plan(self, state, shadow):
        self.replay = shadow
        return super().plan(state, UserParams(f=0.8, k=0.8))

    def _snap_g(self, g):
        return min(self.grid, key=lambda x: abs(x - g))

    def _fit(self, hist):
        def loss(th):
            return hist.nll(th) + 0.5 * float(np.sum(((th - PRIOR_MEAN) / PRIOR_SDS) ** 2))
        starts = [PRIOR_MEAN] + ([self.map] if self.map is not None else [])
        best = min((minimize(loss, s, method="L-BFGS-B", bounds=BOUNDS) for s in starts),
                   key=lambda r: r.fun)
        self.map = best.x
        inner = np.clip(best.x, np.array([b[0] for b in BOUNDS]) + 2 * STEPS,
                        np.array([b[1] for b in BOUNDS]) - 2 * STEPS)
        try:
            cov = np.linalg.inv(_hessian(loss, inner))
            ok = np.all(np.linalg.eigvalsh(cov) > 0)
        except np.linalg.LinAlgError:
            ok = False
        self.cov = cov if ok else np.diag(PRIOR_SDS ** 2)

    def _table(self, bound, t0):
        key = (round(bound / BOUND_STEP) * BOUND_STEP, round(t0, 2))
        if key not in self.tables:
            self.tables[key] = info_table(*key)
        return self.tables[key]

    # A rollout person for one draw: value weights and carried errors from the shadow
    # learner nearest the draw's γ, and two more learners at γ ± H_G for ∂v/∂γ.
    def _rollout(self, state, draw, env, patience, seed):
        g = self._snap_g(draw[2])
        lo, hi = self._snap_g(max(g - H_G, self.grid[0])), self._snap_g(min(g + H_G, self.grid[-1]))
        own = self.replay[g]
        st = copy.deepcopy(dataclasses.replace(
            state, initiation_passages=[], first_passage=[], persistence=[],
            theta=own["theta"].copy(), slot_delta=own["slot"], last_set_delta=own["set"]))
        model = UserParams(f=snap_value(draw[0]), k=snap_value(draw[1]), g=g, a=0.05,
                           patience=patience)
        sim = GridShadow(model, (lo, hi), review_mode=env.review_mode,
                         review_rate=env.review_rate, target_p=env.target_p,
                         feedback=env.feedback, skips=True, return_skipped=env.return_skipped)
        sim.state = st
        for x in (lo, hi):
            sim.shadow[x].update(theta=self.replay[x]["theta"].copy(),
                                 slot=self.replay[x]["slot"], set=self.replay[x]["set"])
        random.seed(seed)
        np.random.seed(seed)
        e0 = st.episodes
        with self.bounds(round(draw[3] / BOUND_STEP) * BOUND_STEP):
            for _ in range(self.info_horizon):
                sim.tick()
        recs = st.initiation_passages + st.first_passage
        if not recs:
            return np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), st.episodes - e0
        v = np.array([p["value"] for p in recs])
        c = np.array([p["cost"] for p in recs])
        vg = np.array([(sum(p["parts_by_g"][hi]) - sum(p["parts_by_g"][lo])) / (hi - lo)
                       for p in recs])
        return v, c, vg, st.episodes - e0

    def _plan(self, state, params):
        rows = decided_rows(state)
        data = state.initiation_passages + state.first_passage
        hist = History(data, self.grid) if data else None
        if rows >= MIN_ROWS and (self.map is None or rows - self.rows_at_fit >= self.refit_every):
            self._fit(hist)
            self.rows_at_fit = rows
        mean, cov = ((PRIOR_MEAN, np.diag(PRIOR_SDS ** 2)) if self.map is None
                     else (self.map, self.cov))
        draws = np.clip(self.np_rng.multivariate_normal(mean, cov, size=DRAWS), DRAW_LO, DRAW_HI)
        patience, pnote = estimate_patience(state, mean[0], mean[1])
        self.est = Estimate(float(mean[0]), float(mean[1]), rows=rows,
                            raw=None if self.map is None else tuple(self.map),
                            t0=float(mean[4]), patience=patience,
                            note=("prior only" if self.map is None else
                                  "MAP " + ", ".join(f"{n} {x:.2f}" for n, x in zip(NAMES, mean)))
                            + "; " + pnote)

        prior = np.diag(1.0 / PRIOR_SDS ** 2)
        bases, tables = [], []
        for d in draws:
            table = self._table(d[3], d[4])
            base = prior.copy()
            if hist is not None:
                v = hist.value(d[2])
                vg = hist.value(d[2], 1)
                base += information(v, hist.cost, vg, d, table)
            bases.append(base)
            tables.append(table)

        scores, sets = {}, {}
        for env in self.candidates:
            gains, started = [], []
            for d, base, table in zip(draws, bases, tables):
                v, c, vg, n_sets = self._rollout(state, d, env, patience,
                                                 self.rng.randrange(2**31))
                gain = logdet(base + information(v, c, vg, d, table)) - logdet(base) if len(v) else 0.0
                gains.append(gain)
                started.append(n_sets)
            scores[env], sets[env] = float(np.mean(gains)), float(np.mean(started))
        floor = self.safety * max(sets.values())
        chosen = max((e for e in self.candidates if sets[e] >= floor), key=scores.get)

        previous = self.env
        self.env = chosen
        self.last_plan = len(state.initiation_passages)
        update = Update(len(state.initiation_passages), self.est, previous, chosen,
                        f"information about 5 parameters +{scores[chosen]:.2f} over {DRAWS} draws",
                        scores)
        self.log.append(update)
        return update


# ---------------------------------------------------------------- the experiment

def _records_until(st, t):
    return [p for p in st.initiation_passages + st.first_passage if p["timestep"] <= t]


def _fit_records(records, i):
    node = np.array([p.get("node") == "stage" for p in records], dtype=np.int8)
    go = np.array([-1 if p["GO"] is None else p["GO"] for p in records], dtype=np.int8)
    rt = np.array([np.nan if p["RT"] is None else p["RT"] for p in records])
    parts = np.array([[p["parts_by_g"][g] for g in G_FINE] for p in records])
    h = dict(node=node, go=go, rt=rt, cost=np.array([p["cost"] for p in records]),
             base=parts[:, :, 0], delta=parts[:, :, 1], g_grid=np.array(G_FINE),
             **{f"true_{n}": 0.0 for n in NAMES})
    return Person(i, h=h).fit()


def _one(task):
    arm, i = task
    truth = people()[i]
    seed = SEED + i
    acc.clear_caches()
    random.seed(seed)
    np.random.seed(seed)
    ctl = InfoController(seed=seed) if arm == "current" else FullInfoController(seed=seed)
    start = ctl.env
    ratings = Ratings()
    person = PersonConstants(acc.SIGMA, truth["bound"])
    sim = GridShadow(UserParams(f=truth["f"], k=truth["k"], g=truth["g"], t0=truth["t0"]),
                     G_FINE, reentry_cost=0.6, review_rate=start.review_rate,
                     review_mode=start.review_mode, target_p=start.target_p,
                     feedback=start.feedback, ratings=ratings, skips=True,
                     return_skipped=start.return_skipped)
    st = sim.state
    st.type_diffs = np.array(synthetic_type_diffs(1.0), dtype=float)
    st.ability = 0.0
    st.in_session = True
    blocks = []
    with Quiet():
        with person():
            sim.run_episode()
        t_start = st.t
        mark = dict(reward=st.total_reward, away=st.away, starts=0, ticks=0)
        while st.t < t_start + N_TICKS:
            with person():
                sim.tick()
            person.trim()
            if len(acc._SOLUTION_CACHE) > CACHE_CAP:
                acc.clear_caches()
            if ctl.due(st):
                if arm == "current":
                    ctl.plan(sim.app_view(ratings), UserParams(f=truth["f"], k=truth["k"]))
                else:
                    ctl.plan(measured(st, ratings), sim.shadow)
                apply(sim, ctl.env)
            if st.t - t_start in CHECKPOINTS:
                n = st.t - t_start
                starts = st.initiation_passages[mark["starts"]:]
                blocks.append(dict(ticks=n, t=st.t, reward=st.total_reward - mark["reward"],
                                   away=(st.away - mark["away"]) / (n - mark["ticks"]),
                                   p_go=float(np.mean([p["chose"] == 1 for p in starts]))
                                   if starts else np.nan))
                mark = dict(reward=st.total_reward, away=st.away,
                            starts=len(st.initiation_passages), ticks=n)
        row = dict(arm=arm, person=i, seed=seed, **{f"true_{n}": truth[n] for n in NAMES},
                   reward=sum(b["reward"] for b in blocks), away=st.away / N_TICKS,
                   decisions=len(st.initiation_passages), redos=len(st.first_passage),
                   p_go=float(np.mean([p["chose"] == 1 for p in st.initiation_passages])),
                   feedback_share=float(np.mean([p["feedback"] for p in st.initiation_passages])),
                   redo_settings=float(np.mean([u.chosen.redo_prompts for u in ctl.log]))
                   if ctl.log else np.nan,
                   target=float(np.mean([u.chosen.target_p for u in ctl.log]))
                   if ctl.log else np.nan)
        for b in blocks:
            n = b["ticks"]
            recs = _records_until(st, b["t"])
            row.update({f"reward_{n}": b["reward"], f"away_{n}": b["away"], f"pgo_{n}": b["p_go"],
                        f"rows_{n}": len(recs)})
            est, nll = _fit_records(recs, i) if len(recs) >= 10 else (np.full(5, np.nan), np.nan)
            row.update({f"{name}_{n}": est[j] for j, name in enumerate(NAMES)})
            row[f"nll_{n}"] = nll
    return row


def summarize(df):
    key = ["person"]
    cur = df[df.arm == "current"].set_index(key).sort_index()
    full = df[df.arm == "full"].set_index(key).sort_index()
    both = cur.index.intersection(full.index)
    cur, full = cur.loc[both], full.loc[both]
    print(f"== full vs current controller: {len(both)} paired people, {N_TICKS} ticks ==")
    print(f"{'':>8} " + " ".join(f"{'RMSE ' + str(n) + ' cur / full':>22}" for n in CHECKPOINTS))
    for name in NAMES:
        cells = []
        for n in CHECKPOINTS:
            ec = cur[f"{name}_{n}"] - cur[f"true_{name}"]
            ef = full[f"{name}_{n}"] - full[f"true_{name}"]
            cells.append(f"{np.sqrt(np.nanmean(ec ** 2)):8.3f} / {np.sqrt(np.nanmean(ef ** 2)):.3f}")
        print(f"{name:>8} " + " ".join(f"{c:>22}" for c in cells))
    print(f"\nclaims, paired over people at {N_TICKS} ticks (full − current):")
    for claim, name in (("F1", "k"), ("F2", "f"), ("F3", "g")):
        d = ((full[f"{name}_{N_TICKS}"] - full[f"true_{name}"]) ** 2
             - (cur[f"{name}_{N_TICKS}"] - cur[f"true_{name}"]) ** 2).dropna()
        print(f"  {claim} squared error in {name}: {d.mean():+.4f} ± {se(d):.4f} "
              f"({'holds' if d.mean() < -2 * se(d) else 'does not hold'})")
    d = (full.p_go - cur.p_go).dropna()
    print(f"  F4 P(start): {d.mean():+.3f} ± {se(d):.3f} "
          f"({'holds' if d.mean() > -2 * se(d) else 'does not hold'})")
    print("\nF5 (descriptive), mean over people, current / full:")
    for col in ("reward", "away", "decisions", "redos", "p_go", "feedback_share",
                "redo_settings", "target"):
        d = full[col] - cur[col]
        print(f"  {col:>15}: {cur[col].mean():8.3f} / {full[col].mean():8.3f}   "
              f"difference {d.mean():+.3f} ± {se(d):.3f}")
    for name in ("bound", "t0"):
        d = ((full[f"{name}_{N_TICKS}"] - full[f"true_{name}"]) ** 2
             - (cur[f"{name}_{N_TICKS}"] - cur[f"true_{name}"]) ** 2).dropna()
        print(f"  squared error in {name}, full − current: {d.mean():+.4f} ± {se(d):.4f}")
    print("\nfeedback shown, share of start decisions, by true γ (current / full):")
    bins = pd.cut(cur.true_g, [0.6, 0.7, 0.8, 0.9, 0.97], include_lowest=True)
    for b, idx in cur.groupby(bins, observed=True).groups.items():
        print(f"  γ {b}: {cur.loc[idx, 'feedback_share'].mean():.2f} / "
              f"{full.loc[idx, 'feedback_share'].mean():.2f} ({len(idx)} people)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="merge and summarize the saved results")
    p.add_argument("--people", type=span, default=list(range(N_PEOPLE)), help="for example 0-23")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--tag", default="local", help="names this machine's file, runs_<tag>.csv")
    _design_args(p)
    a = p.parse_args()
    if not a.replot:
        save_conditions(D)
        tasks = [(arm, i) for i in a.people for arm in a.arms.split(",")]
        run_pool(_one, tasks, OUT / f"runs_{a.tag}.csv", fresh=True,
                 label=lambda r: f"{r['arm']} person {r['person']}")
    files = sorted(OUT.glob("runs_*.csv"))
    summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
