"""Planning with every fitted parameter, choosing every setting the model can see.

FullInfoController (experiments.controller_full) fits, draws and scores f, k, γ, BOUND and
T0, but it chooses among InfoController's settings. Redo prompts there need feedback on,
because a hidden miss cannot be offered back, so it showed feedback before 98% of start
decisions and never used the lever that measures γ. AllInfoController, here, removes both
limits:

- **Parameters.** f, k, γ, BOUND, T0 and λ, the share of BOUND kept on a redo prompt
  under a countdown (model/pressure.py, experiments.redo_pressure). All six are fitted by
  MAP with the closed-form likelihood, drawn from the Laplace posterior for rollouts, and
  scored. A countdown decision is read at bound λ · BOUND and its own deadline.
- **Settings.** InfoController's 84 scheduled-review settings, and redo prompts at each of
  the 6 targets in two places, mid-set with feedback shown or on the end-of-set screen with
  feedback hidden (controller.REDO_END), each with no countdown or one of COUNTDOWNS
  seconds: 84 + 6 · 2 · 4 = 132. Feedback, where redos are offered and whether and how
  long they are timed are all chosen by the score, never on a schedule.
- **Score.** The rise in log det of the Fisher information about all six parameters. A
  decision's outcome depends on (μ, b, T0) at its deadline, with μ = f·v(γ) − k·c and
  b = BOUND, or λ · BOUND under a countdown. Its information is Jᵀ I(μ, b, T0) J with J's
  rows (v, −c, f·∂v/∂γ, 0, 0, 0), (0, 0, 0, s, 0, BOUND·[countdown]) with s = λ under a
  countdown and 1 otherwise, and (0, 0, 0, 0, 1, 0). I is from fastlik at that deadline.
- The prior on f, k, γ, BOUND and T0, the 6 draws, the 25-tick rollouts, the safety floor,
  the plan every 25 start decisions and the re-fit every 15 decided rows are
  FullInfoController's. λ's prior is 0.7 ± 0.2.

The experiment. The 128 people and seeds of experiments.controller_full, each given a λ
drawn uniformly from 0.4 to 1.0 by a seeded generator of its own. Each plays the app's task
(question types with Elo ratings, skips on, cold-start cost 0.6) for 300 ticks under
AllInfoController. The other two arms are controller_full's saved runs of the same people
on the same seeds (InfoController as the app runs it, and FullInfoController): neither
offers a countdown, so λ never enters them. The simulated person sees the score at the end
of a hidden set, which the app does not show yet. α 0.05 for person, shadows and
controller; SIGMA 0.7, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0. At 100, 200
and 300 ticks this arm's data are fitted by maximum likelihood, all six parameters (λ is
flat for a person who never met a countdown), as controller_full fits its arms' five.

Claims, fixed before any run (2026-10-28), paired on person, at 300 ticks:
    A1  squared error in γ, all − current, is below zero beyond 2 SE
    A2  squared error in k, all − current, is not above zero beyond 2 SE
    A3  P(start next set), all − current, is not below zero beyond 2 SE
    A4  none: f, BOUND, T0 and λ errors (λ against redo_pressure's, which had 300 start
        decisions); the same against full; reward, time away, decisions, redo decisions;
        the share of start decisions after a hidden set, of redo decisions under each
        countdown, and of redos at the end of the set, by true γ and λ

A second arm, ds (added 2026-10-29, after the all arm's results were read):
DsInfoController is AllInfoController with the score on f, k and γ only (Ds-optimal). It
still fits, draws and rolls out all six parameters, but scores a setting by the rise in
log det of the Schur complement of the information about (f, k, γ), so measuring BOUND, T0
or λ earns nothing for its own sake. Same people, seeds and fits.

Claims, fixed before the ds run (2026-10-29), paired on person, at 300 ticks:
    D1  squared error in k, ds − all, is below zero beyond 2 SE
    D2  squared error in γ, ds − current, is below zero beyond 2 SE
    D3  P(start next set), ds − current, is not below zero beyond 2 SE
    D4  none: the A4 quantities for ds, and ds against all

    python -m experiments.controller_all --people 0-63 --tag server
    python -m experiments.controller_all --arms ds --tag server_ds
    python -m experiments.controller_all --replot
"""
import argparse
import copy
import dataclasses
import random

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from temporal import fastlik
from temporal.controller import (DRAWS, MIN_ROWS, MU_GRID, SCHEDULED, TARGETS, Environment,
                                 Estimate, Update, apply, decided_rows, estimate_patience,
                                 logdet, measured)
from temporal.model import accumulator as acc
from temporal.model.accumulator import snap_value
from temporal.model.params import UserParams
from temporal.model.questions import Ratings, synthetic_type_diffs

from . import controller_full as full
from .common import Quiet, _design_args, design, out_dir, run_pool, save_conditions, se
from .controller_full import (BOUND_STEP, H_G, PRIOR_MEAN, PRIOR_SDS, FullInfoController,
                              History, _hessian, _records_until)
from .gamma_profile import CACHE_CAP, GridShadow
from .misspecification import PersonConstants
from .recovery_lhc import BOUNDS, G_FINE, NAMES, Person, span

D = design("controller_all", people=full.N_PEOPLE, seed=full.SEED, lam_seed=20261033,
           lam_range=(0.4, 1.0), checkpoints=full.CHECKPOINTS, countdowns=(2.0, 3.0, 5.0))
OUT = out_dir(D.out)
N_PEOPLE = D.people
SEED = D.seed
CHECKPOINTS = D.checkpoints
N_TICKS = CHECKPOINTS[-1]
COUNTDOWNS = tuple(D.countdowns)

NAMES6 = (*NAMES, "lam")
BOUNDS6 = [*BOUNDS, (0.2, 1.5)]
PRIOR_MEAN6 = np.array([*PRIOR_MEAN, 0.7])
PRIOR_SDS6 = np.array([*PRIOR_SDS, 0.2])
DRAW_LO6 = np.array([*full.DRAW_LO, 0.2])
DRAW_HI6 = np.array([*full.DRAW_HI, 1.2])
STEPS6 = np.array([*full.STEPS, 0.01])
LAM_STEP = 0.05  # rollout λ is snapped to this, so pressure.py's draw tables are reused

# Redo prompts mid-set (feedback shown) or at the end of the set (hidden), each untimed or
# under one of the countdowns.
REDO_ALL = tuple(Environment(t, 0.0, mid, False, True, not mid, c is not None,
                             c if c is not None else Environment.countdown)
                 for t in TARGETS for mid in (True, False) for c in (None, *COUNTDOWNS))
CANDIDATES_ALL = SCHEDULED + REDO_ALL


def people():
    lam = np.random.default_rng(D.lam_seed).uniform(*D.lam_range, N_PEOPLE)
    return [dict(p, lam=float(x)) for p, x in zip(full.people(), lam)]


# Fisher information of one decision about (μ, b, T0) at deadline `t_dur`, on MU_GRID:
# controller_full.info_table with the deadline free.
def info_table(bound, t0, t_dur, h=(0.01, 0.01, 0.005)):
    def probs(mu, b, t):
        up, lo, und = fastlik.outcome_probs(mu, b, max(t, 0.0), t_dur=t_dur)
        return np.concatenate([up, lo, und[:, None]], axis=1)
    mid = probs(MU_GRID, bound, t0)
    d = [(probs(MU_GRID + h[0], bound, t0) - probs(MU_GRID - h[0], bound, t0)) / (2 * h[0]),
         (probs(MU_GRID, bound + h[1], t0) - probs(MU_GRID, bound - h[1], t0)) / (2 * h[1]),
         (probs(MU_GRID, bound, t0 + h[2]) - probs(MU_GRID, bound, max(t0 - h[2], 0.0)))
         / (h[2] + min(h[2], t0))]
    w = 1.0 / np.maximum(mid, 1e-300)
    return {(a, b): np.sum(d[a] * d[b] * w, axis=1) for a in range(3) for b in range(a, 3)}


class Decisions:
    """Decisions as arrays for the information score: value, cost, ∂v/∂γ, and each one's
    countdown in seconds (NaN for none)."""

    def __init__(self, v, c, vg, deadline):
        self.v, self.c, self.vg = (np.asarray(x, float) for x in (v, c, vg))
        self.deadline = np.asarray(deadline, float)

    def __len__(self):
        return len(self.v)


def _deadlines(records):
    return np.array([p.get("deadline", np.nan) for p in records], dtype=float)


class History6(History):
    """controller_full.History with countdown decisions read at λ · BOUND and their own
    deadline. θ = (f, k, γ, BOUND, T0, λ)."""

    def __init__(self, records, grid):
        super().__init__(records, grid)
        self.deadline = _deadlines(records)
        self.timed = ~np.isnan(self.deadline)
        self.t_dur = np.where(self.timed, self.deadline, acc.T_DUR)

    def nll(self, th):
        mu = th[0] * self.value(th[2]) - th[1] * self.cost
        bound = th[3] * np.where(self.timed, th[5], 1.0)
        return fastlik.neg_log_likelihood(self.go, self.rt, mu, bound, th[4], t_dur=self.t_dur)


class AllInfoController(FullInfoController):
    """FullInfoController over six parameters and CANDIDATES_ALL (module docstring)."""

    def __init__(self, candidates=CANDIDATES_ALL, **kw):
        super().__init__(candidates=candidates, **kw)

    def _fit(self, hist):
        def loss(th):
            return hist.nll(th) + 0.5 * float(np.sum(((th - PRIOR_MEAN6) / PRIOR_SDS6) ** 2))
        starts = [PRIOR_MEAN6] + ([self.map] if self.map is not None else [])
        best = min((minimize(loss, s, method="L-BFGS-B", bounds=BOUNDS6) for s in starts),
                   key=lambda r: r.fun)
        self.map = best.x
        inner = np.clip(best.x, np.array([b[0] for b in BOUNDS6]) + 2 * STEPS6,
                        np.array([b[1] for b in BOUNDS6]) - 2 * STEPS6)
        try:
            cov = np.linalg.inv(_hessian(loss, inner, STEPS6))
            ok = np.all(np.linalg.eigvalsh(cov) > 0)
        except np.linalg.LinAlgError:
            ok = False
        self.cov = cov if ok else np.diag(PRIOR_SDS6 ** 2)

    def _table(self, bound, t0, t_dur=acc.T_DUR):
        key = (round(bound / BOUND_STEP) * BOUND_STEP, round(t0, 2), t_dur)
        if key not in self.tables:
            self.tables[key] = info_table(*key)
        return self.tables[key]

    # Information about (f, k, γ, BOUND, T0, λ) in `dec` for one draw.
    def information(self, dec, draw):
        f, k, bound, t0, lam = draw[0], draw[1], draw[3], draw[4], draw[5]
        M = np.zeros((6, 6))
        if not len(dec):
            return M
        mu = f * dec.v - k * dec.c
        timed = ~np.isnan(dec.deadline)
        n = len(dec)
        J = np.zeros((n, 3, 6))  # d(μ, b, T0) / dθ for each decision
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dec.v, -dec.c, f * dec.vg
        J[:, 1, 3] = np.where(timed, lam, 1.0)
        J[:, 1, 5] = np.where(timed, bound, 0.0)
        J[:, 2, 4] = 1.0
        info = np.zeros((n, 3, 3))  # about (μ, b, T0), per decision
        groups = [(~timed, bound, acc.T_DUR)] + [(dec.deadline == t, lam * bound, float(t))
                                                 for t in np.unique(dec.deadline[timed])]
        for rows, b, t_dur in groups:
            if not rows.any():
                continue
            for (a, c), arr in self._table(b, t0, t_dur).items():
                info[rows, a, c] = info[rows, c, a] = np.interp(mu[rows], MU_GRID, arr)
        return np.einsum("nai,nab,nbj->ij", J, info, J)

    # A rollout person for one draw, in `env`: controller_full's, with its own λ, and redos
    # at the end of the set or under a countdown when `env` has them.
    def _rollout(self, state, draw, env, patience, seed):
        g = self._snap_g(draw[2])
        lo, hi = self._snap_g(max(g - H_G, self.grid[0])), self._snap_g(min(g + H_G, self.grid[-1]))
        own = self.replay[g]
        st = copy.deepcopy(dataclasses.replace(
            state, initiation_passages=[], first_passage=[], persistence=[],
            theta=own["theta"].copy(), slot_delta=own["slot"], last_set_delta=own["set"]))
        model = UserParams(f=snap_value(draw[0]), k=snap_value(draw[1]), g=g, a=0.05,
                           patience=patience, pressure=round(draw[5] / LAM_STEP) * LAM_STEP)
        sim = GridShadow(model, (lo, hi), review_mode=env.review_mode,
                         review_rate=env.review_rate, target_p=env.target_p,
                         feedback=env.feedback, skips=True, return_skipped=env.return_skipped,
                         end_redos=env.end_redos, redo_deadline=env.redo_deadline)
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
        vg = [(sum(p["parts_by_g"][hi]) - sum(p["parts_by_g"][lo])) / (hi - lo) for p in recs]
        dec = Decisions([p["value"] for p in recs], [p["cost"] for p in recs], vg,
                        _deadlines(recs))
        return dec, st.episodes - e0

    def _plan(self, state, params):
        rows = decided_rows(state)
        data = state.initiation_passages + state.first_passage
        hist = History6(data, self.grid) if data else None
        if rows >= MIN_ROWS and (self.map is None or rows - self.rows_at_fit >= self.refit_every):
            self._fit(hist)
            self.rows_at_fit = rows
        mean, cov = ((PRIOR_MEAN6, np.diag(PRIOR_SDS6 ** 2)) if self.map is None
                     else (self.map, self.cov))
        draws = np.clip(self.np_rng.multivariate_normal(mean, cov, size=DRAWS), DRAW_LO6, DRAW_HI6)
        patience, pnote = estimate_patience(state, mean[0], mean[1])
        self.est = Estimate(float(mean[0]), float(mean[1]), rows=rows,
                            raw=None if self.map is None else tuple(self.map),
                            t0=float(mean[4]), patience=patience,
                            note=("prior only" if self.map is None else
                                  "MAP " + ", ".join(f"{n} {x:.2f}" for n, x in zip(NAMES6, mean)))
                            + "; " + pnote)

        prior = np.diag(1.0 / PRIOR_SDS6 ** 2)
        bases = []
        for d in draws:
            base = prior.copy()
            if hist is not None:
                base += self.information(
                    Decisions(hist.value(d[2]), hist.cost, hist.value(d[2], 1), hist.deadline), d)
            bases.append(base)

        scores, sets = {}, {}
        for env in self.candidates:
            gains, started = [], []
            for d, base in zip(draws, bases):
                dec, n_sets = self._rollout(state, d, env, patience, self.rng.randrange(2**31))
                gains.append(self.gain(base, self.information(dec, d)) if len(dec) else 0.0)
                started.append(n_sets)
            scores[env], sets[env] = float(np.mean(gains)), float(np.mean(started))
        floor = self.safety * max(sets.values())
        chosen = max((e for e in self.candidates if sets[e] >= floor), key=scores.get)

        previous = self.env
        self.env = chosen
        self.last_plan = len(state.initiation_passages)
        update = Update(len(state.initiation_passages), self.est, previous, chosen,
                        f"information about {self.scored} +{scores[chosen]:.2f} over {DRAWS} draws",
                        scores)
        self.log.append(update)
        return update

    scored = "6 parameters"

    # What `add` (a setting's rollout) adds to `base` (prior and data so far): the rise in
    # log det of the information about all six parameters.
    def gain(self, base, add):
        return logdet(base + add) - logdet(base)


# Information about the parameters in `interest` once the others are unknown: the Schur
# complement M_ii − M_in M_nn⁻¹ M_ni.
def marginal(M, interest):
    rest = [j for j in range(len(M)) if j not in interest]
    i, n = np.ix_(interest, interest), np.ix_(interest, rest)
    return M[i] - M[n] @ np.linalg.solve(M[np.ix_(rest, rest)], M[n].T)


class DsInfoController(AllInfoController):
    """AllInfoController scored only on f, k and γ (Ds-optimal): BOUND, T0 and λ are still
    fitted, drawn and rolled out, but a setting earns nothing for measuring them."""

    INTEREST = (0, 1, 2)
    scored = "f, k and γ"

    def gain(self, base, add):
        return (logdet(marginal(base + add, self.INTEREST))
                - logdet(marginal(base, self.INTEREST)))


# ---------------------------------------------------------------- the experiment

class Person6(Person):
    """recovery_lhc.Person with countdown decisions read at λ · BOUND and their own
    deadline. θ = (f, k, γ, BOUND, T0, λ)."""

    def __init__(self, i, h, deadline):
        super().__init__(i, h=h)
        self.timed = ~np.isnan(deadline)
        self.t_dur = np.where(self.timed, deadline, acc.T_DUR)

    def nll(self, th):
        bound = th[3] * np.where(self.timed, th[5], 1.0)
        return fastlik.neg_log_likelihood(self.go, self.rt, self.mu(th), bound, th[4],
                                          t_dur=self.t_dur)

    def fit6(self):
        res = differential_evolution(self.nll, BOUNDS6, seed=self.i, tol=1e-8, maxiter=2000,
                                     polish=False)
        res = minimize(self.nll, res.x, method="L-BFGS-B", bounds=BOUNDS6)
        return res.x, float(res.fun)


def _fit_records(records, i):
    node = np.array([p.get("node") == "stage" for p in records], dtype=np.int8)
    go = np.array([-1 if p["GO"] is None else p["GO"] for p in records], dtype=np.int8)
    rt = np.array([np.nan if p["RT"] is None else p["RT"] for p in records])
    parts = np.array([[p["parts_by_g"][g] for g in G_FINE] for p in records])
    h = dict(node=node, go=go, rt=rt, cost=np.array([p["cost"] for p in records]),
             base=parts[:, :, 0], delta=parts[:, :, 1], g_grid=np.array(G_FINE),
             **{f"true_{n}": 0.0 for n in NAMES})
    return Person6(i, h, _deadlines(records)).fit6()


ARMS = {"all": AllInfoController, "ds": DsInfoController}


def _one(task, n_ticks=N_TICKS, checkpoints=CHECKPOINTS):
    arm, i = task
    truth = people()[i]
    seed = SEED + i
    acc.clear_caches()
    random.seed(seed)
    np.random.seed(seed)
    ctl = ARMS[arm](seed=seed)
    start = ctl.env
    ratings = Ratings()
    person = PersonConstants(acc.SIGMA, truth["bound"])
    sim = GridShadow(UserParams(f=truth["f"], k=truth["k"], g=truth["g"], t0=truth["t0"],
                                pressure=truth["lam"]),
                     G_FINE, reentry_cost=0.6, review_rate=start.review_rate,
                     review_mode=start.review_mode, target_p=start.target_p,
                     feedback=start.feedback, ratings=ratings, skips=True,
                     return_skipped=start.return_skipped, end_redos=start.end_redos,
                     redo_deadline=start.redo_deadline)
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
        while st.t < t_start + n_ticks:
            with person():
                sim.tick()
            person.trim()
            if len(acc._SOLUTION_CACHE) > CACHE_CAP:
                acc.clear_caches()
            if ctl.due(st):
                ctl.plan(measured(st, ratings), sim.shadow)
                apply(sim, ctl.env)
            if st.t - t_start in checkpoints:
                n = st.t - t_start
                starts = st.initiation_passages[mark["starts"]:]
                blocks.append(dict(ticks=n, t=st.t, reward=st.total_reward - mark["reward"],
                                   away=(st.away - mark["away"]) / (n - mark["ticks"]),
                                   p_go=float(np.mean([p["chose"] == 1 for p in starts]))
                                   if starts else np.nan))
                mark = dict(reward=st.total_reward, away=st.away,
                            starts=len(st.initiation_passages), ticks=n)
        redos = st.first_passage
        row = dict(arm=arm, person=i, seed=seed, **{f"true_{n}": truth[n] for n in NAMES6},
                   reward=sum(b["reward"] for b in blocks), away=st.away / n_ticks,
                   decisions=len(st.initiation_passages), redos=len(redos),
                   end_redos=sum(bool(p.get("end")) for p in redos),
                   p_go=float(np.mean([p["chose"] == 1 for p in st.initiation_passages])),
                   feedback_share=float(np.mean([p["feedback"] for p in st.initiation_passages])),
                   redo_settings=float(np.mean([u.chosen.redo_prompts for u in ctl.log]))
                   if ctl.log else np.nan,
                   target=float(np.mean([u.chosen.target_p for u in ctl.log]))
                   if ctl.log else np.nan)
        for c in COUNTDOWNS:
            row[f"redos_{c:g}s"] = sum(p.get("deadline") == c for p in redos)
        for b in blocks:
            n = b["ticks"]
            recs = _records_until(st, b["t"])
            row.update({f"reward_{n}": b["reward"], f"away_{n}": b["away"], f"pgo_{n}": b["p_go"],
                        f"rows_{n}": len(recs)})
            est, nll = _fit_records(recs, i) if len(recs) >= 10 else (np.full(6, np.nan), np.nan)
            row.update({f"{name}_{n}": est[j] for j, name in enumerate(NAMES6)})
            row[f"nll_{n}"] = nll
    return row


def _sq(x, name, n=N_TICKS):
    return (x[f"{name}_{n}"] - x[f"true_{name}"]) ** 2


def summarize(df):
    saved = pd.concat([pd.read_csv(f) for f in sorted(full.OUT.glob("runs_*.csv"))],
                      ignore_index=True)
    arms = {"current": saved[saved.arm == "current"], "full": saved[saved.arm == "full"],
            **{a: df[df.arm == a] for a in ARMS if (df.arm == a).any()}}
    arms = {a: g.set_index("person").sort_index() for a, g in arms.items()}
    common = sorted(set.intersection(*(set(g.index) for g in arms.values())))
    arms = {a: g.loc[common] for a, g in arms.items()}
    cur, ful = arms["current"], arms["full"]
    new = [a for a in ARMS if a in arms]
    print(f"== every parameter, every setting: {len(common)} paired people, {N_TICKS} ticks ==")
    print(f"{'RMSE at 100 / 200 / 300':>24} " + "".join(f"{a:>24}" for a in arms))
    for name in NAMES6:
        cells = []
        for x in arms.values():
            cells.append(" / ".join(f"{np.sqrt(np.nanmean(_sq(x, name, n))):.3f}"
                                    if f"{name}_{n}" in x else "-" for n in CHECKPOINTS))
        print(f"{name:>24} " + "".join(f"{c:>24}" for c in cells))

    def claim(label, d, absent=False, below=True):
        m, e = d.mean(), se(d)
        ok = (m < 2 * e if below else m > -2 * e) if absent else m < -2 * e
        print(f"  {label}: {m:+.4f} ± {e:.4f} ({'holds' if ok else 'does not hold'})")
    print("\nclaims, paired at 300 ticks:")
    if "all" in arms:
        al = arms["all"]
        claim("A1 γ sq. error, all − current", (_sq(al, "g") - _sq(cur, "g")).dropna())
        claim("A2 k sq. error, all − current (not above)",
              (_sq(al, "k") - _sq(cur, "k")).dropna(), absent=True)
        claim("A3 P(start), all − current (not below)", (al.p_go - cur.p_go).dropna(),
              absent=True, below=False)
    if "ds" in arms and "all" in arms:
        ds = arms["ds"]
        claim("D1 k sq. error, ds − all", (_sq(ds, "k") - _sq(arms["all"], "k")).dropna())
        claim("D2 γ sq. error, ds − current", (_sq(ds, "g") - _sq(cur, "g")).dropna())
        claim("D3 P(start), ds − current (not below)", (ds.p_go - cur.p_go).dropna(),
              absent=True, below=False)

    pairs = [(arms[a], cur, f"{a} − current") for a in new] + \
            [(arms[a], ful, f"{a} − full") for a in new]
    if "ds" in arms and "all" in arms:
        pairs.append((arms["ds"], arms["all"], "ds − all"))
    print("\ndescriptive, paired differences in squared error at 300 ticks:")
    for x, y, label in pairs:
        cells = [f"{n} {(_sq(x, n) - _sq(y, n)).mean():+.4f} ± {se((_sq(x, n) - _sq(y, n)).dropna()):.4f}"
                 for n in NAMES]
        print(f"  {label:<14} " + "  ".join(cells))
    for a in new:
        x = arms[a]
        fitted = x[x["lam_300"].notna()]
        err = fitted["lam_300"] - fitted.true_lam
        timed = fitted[[f"redos_{c:g}s" for c in COUNTDOWNS]].sum(axis=1) >= 20
        print(f"  {a} λ, {len(fitted)} people fitted: RMSE {np.sqrt(np.mean(err ** 2)):.3f}, "
              f"bias {err.mean():+.3f} ± {se(err):.3f}, "
              f"r {np.corrcoef(fitted['lam_300'], fitted.true_lam)[0, 1]:.3f}; with 20 or more "
              f"timed redos ({timed.sum()}): RMSE {np.sqrt(np.mean(err[timed] ** 2)):.3f}, "
              f"r {np.corrcoef(fitted['lam_300'][timed], fitted.true_lam[timed])[0, 1]:.3f}")
    cols = ("reward", "away", "p_go", "decisions", "redos", "feedback_share", "redo_settings",
            "target")
    print(f"\n{'mean per person':>16} " + "".join(f"{a:>12}" for a in arms))
    for col in cols:
        print(f"{col:>16} " + "".join(f"{x[col].mean():12.3f}" for x in arms.values()))
    for x, y, label in pairs:
        cells = [f"{c} {(x[c] - y[c]).mean():+.3f} ± {se((x[c] - y[c]).dropna()):.3f}"
                 for c in ("reward", "away", "p_go", "decisions")]
        print(f"  {label}: " + ", ".join(cells))
    timed = [f"redos_{c:g}s" for c in COUNTDOWNS]
    for a in new:
        x = arms[a]
        print(f"\n{a} arm: redos per person {x.redos.mean():.0f}, at the end of the set "
              f"{x.end_redos.mean():.0f}, under a countdown of "
              + ", ".join(f"{c:g} s {x[t].mean():.0f}" for c, t in zip(COUNTDOWNS, timed)))
        print(f"by true γ: feedback shown before a start (current / {a}), end-of-set redos")
        for b, idx in x.groupby(pd.cut(x.true_g, [0.6, 0.7, 0.8, 0.9, 0.97],
                                       include_lowest=True), observed=True).groups.items():
            print(f"  γ {b}: {cur.loc[idx, 'feedback_share'].mean():.2f} / "
                  f"{x.loc[idx, 'feedback_share'].mean():.2f}, {x.loc[idx, 'end_redos'].mean():.0f} "
                  f"({len(idx)} people)")
        print("by true λ: redos under a countdown (" + ", ".join(f"{c:g} s" for c in COUNTDOWNS)
              + ")")
        for b, g in x.groupby(pd.cut(x.true_lam, [0.4, 0.55, 0.7, 0.85, 1.0],
                                     include_lowest=True), observed=True):
            print(f"  λ {b}: " + ", ".join(f"{g[t].mean():.0f}" for t in timed)
                  + f" ({len(g)} people)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="merge and summarize the saved results")
    p.add_argument("--people", type=span, default=list(range(N_PEOPLE)), help="for example 0-63")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--tag", default="local", help="names this machine's file, runs_<tag>.csv")
    _design_args(p)
    a = p.parse_args()
    if not a.replot:
        save_conditions(D)
        tasks = [(arm, i) for arm in a.arms.split(",") for i in a.people]
        run_pool(_one, tasks, OUT / f"runs_{a.tag}.csv", fresh=True,
                 label=lambda r: f"{r['arm']} person {r['person']}")
    files = sorted(OUT.glob("runs_*.csv"))
    summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
