"""The controllers: re-fit the person, roll out candidate settings, choose one.

`Controller` plans for reward and switches on a margin; `InfoController` plans to
measure f and k, within a floor on how much the person keeps working.

    python -m temporal.controller --values 0.8,0.8   # one simulated person
"""
import argparse
import copy
import dataclasses
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import fastlik
from .inference import LAM_PRIOR, decision_arrays, fit, fit_closed_form, passages_to_frame
from .model.accumulator import BOUND, DT, snap_value, solution
from .model.params import PATIENCE, UserParams
from .model.persistence import attempts_to_frame, fit_patience
from .model.pressure import REDO_DEADLINE
from .model.questions import Ratings, synthetic_type_diffs
from .model.simulation import Simulation


@dataclass(frozen=True)
class Environment:
    target_p: float = 0.75
    review_rate: float = 0.25
    feedback: bool = True
    return_skipped: bool = False
    redo_prompts: bool = False  # ask whether to redo each seen miss, as the app can
    # With feedback hidden, ask on the end-of-set screen instead of not at all.
    end_redos: bool = False
    pressure: bool = False  # a countdown on every redo prompt (model/pressure.py)
    countdown: float = REDO_DEADLINE  # its length in seconds, when there is one
    stage_amt: int = 4       # questions per set
    show_score: bool = True  # hidden, only right and wrong are seen: the reward is accuracy
    forced: bool = False     # with redo prompts off, every seen miss is redone, undecided
    end_feedback: bool = False  # an end-of-set redo's outcome is shown, and lands

    @property
    def review_mode(self):
        if self.redo_prompts:
            return "choice"
        return "forced" if self.forced else "scheduled"

    def __str__(self):
        return (f"aim {self.target_p:.0%} right, review {self.review_rate:.2f}, "
                f"feedback {'on' if self.feedback else 'off'}, "
                f"skipped {'come back' if self.return_skipped else 'dropped'}"
                + (", redo prompts" if self.redo_prompts else "")
                + (" at the end of the set" if self.end_redos else "")
                + (f" with a {self.countdown:g} s countdown" if self.pressure else "")
                + (", misses repeated" if self.forced else "")
                + (f", {self.stage_amt} per set" if self.stage_amt != 4 else "")
                + ("" if self.show_score else ", score hidden")
                + (", their outcomes shown" if self.end_feedback else ""))

    @property
    def redo_deadline(self):
        return self.countdown if self.pressure else None


TARGETS = (0.9, 0.75, 0.6, 0.45, 0.3, 0.2)
REVIEW_LEVELS = (0.0, 0.25, 0.5, 1.0)
FEEDBACK_LEVELS = (True, False)
SKIPPED_LEVELS = (False, True)
SCHEDULED = tuple(Environment(t, r, fb, rs)
                  for t in TARGETS for r in REVIEW_LEVELS for fb in FEEDBACK_LEVELS
                  for rs in SKIPPED_LEVELS if r > 0 or not rs)
# Misses come back only when the person asks: the redo decision, where cost varies most.
REDO = tuple(Environment(t, 0.0, fb, False, True) for t in TARGETS for fb in FEEDBACK_LEVELS)
CANDIDATES = SCHEDULED + REDO
# Feedback hidden, and redos offered on the end-of-set screen. The app has that screen,
# but these are not among CANDIDATES: only experiments/feedback_blocks.py has tried them,
# with feedback switched in blocks, not as choices for the app's controller.
REDO_END = tuple(Environment(t, 0.0, False, False, True, True) for t in TARGETS)

# Estimation guardrails: the estimate starts at a prior and earns weight with data.
PRIOR = (0.8, 0.8)
MIN_ROWS = 40         # decided rows before the fit is used at all
SHRINK_ROWS = 80      # decided rows at which fit and prior weigh equally
REFIT_EVERY = 15      # new decided rows needed before fitting again
SHRINK_GIVE_UPS = 10  # give-ups at which fitted patience and the default weigh equally
F_BOUNDS = (0.0, 3.0)


@dataclass
class Estimate:
    f: float
    k: float
    rows: int = 0
    raw: tuple = None           # the fit before shrinking; None when not fitted
    note: str = ""
    patience: float = PATIENCE  # seconds per unit of model time while on a question
    t0: float = 0.0             # non-decision time; it shifts RTs, so planning ignores it
    lam: float = LAM_PRIOR[0]   # share of BOUND kept under a redo countdown (AppController)


def decided_rows(state):
    return sum(r["GO"] is not None for r in state.initiation_passages + state.first_passage)


# (f, k) for planning: the fit, shrunk toward the prior by how thin the data is.
def estimate(state, previous=None):
    passages = state.initiation_passages + state.first_passage
    df = passages_to_frame(passages)
    n = len(df)
    if n < MIN_ROWS:
        est = Estimate(*PRIOR, rows=n, note=f"prior only ({n}/{MIN_ROWS} rows)")
    else:
        # T0 is fitted, not assumed away: pinning it at 0 biases f and k down. It only
        # shifts RTs, so the rollouts never use it.
        raw = fit(df, t0=True)
        f_hat, k_hat, t0_hat = raw
        lo, hi = F_BOUNDS
        if min(f_hat, k_hat) <= lo + 0.01 or max(f_hat, k_hat) >= hi - 0.01: # hit the search bound
            base = previous or Estimate(*PRIOR)
            est = Estimate(base.f, base.k, rows=n, raw=raw, t0=t0_hat,
                           note="f/k fit hit a bound, kept previous")
        else:
            w = n / (n + SHRINK_ROWS) # weight on the fit against the prior
            est = Estimate(w * f_hat + (1 - w) * PRIOR[0], w * k_hat + (1 - w) * PRIOR[1],
                           rows=n, raw=raw, t0=t0_hat,
                           note=f"{w:.0%} fit / {1 - w:.0%} prior; T0 {t0_hat:.2f}s")

    est.patience, note = estimate_patience(state, est.f, est.k)
    est.note += "; " + note
    return est


# (patience, note): how quickly this person gives up, with f and k as estimated.
def estimate_patience(state, f, k):
    attempts = attempts_to_frame(state.persistence) # every question worked on
    gave_up = int(attempts["skipped"].sum()) if len(attempts) else 0
    if gave_up == 0:
        return PATIENCE, f"patience {PATIENCE:.0f}s default (nothing given up on yet)"
    raw = fit_patience(f, k, attempts)
    w = gave_up / (gave_up + SHRINK_GIVE_UPS)
    patience = math.exp(w * math.log(raw) + (1 - w) * math.log(PATIENCE)) # shrunk in log terms
    return patience, f"patience fit {raw:.1f}s from {gave_up} give-ups in {len(attempts)} questions"


# The person as the environment knows them: measured ability and type ratings.
def measured(state, ratings):
    return dataclasses.replace(state, ability=ratings.ability,
                               type_diffs=np.array(ratings.difficulty, dtype=float))


# A copy of the person as they are now, logs stripped, placed in `env`.
def fork(state, params, env):
    st = copy.deepcopy(dataclasses.replace(state, initiation_passages=[], first_passage=[],
                                           persistence=[]))
    sim = Simulation(params, review_mode=env.review_mode, review_rate=env.review_rate,
                     target_p=env.target_p, feedback=env.feedback,
                     skips=True, return_skipped=env.return_skipped, end_redos=env.end_redos,
                     redo_deadline=env.redo_deadline, show_score=env.show_score,
                     end_feedback=env.end_feedback)
    sim.state = st
    st.stage_amt = env.stage_amt
    return sim


# Put a running simulation into `env`.
def apply(sim, env):
    sim.target_p, sim.review_rate = env.target_p, env.review_rate
    sim.feedback, sim.return_skipped = env.feedback, env.return_skipped
    sim.review_mode, sim.end_redos = env.review_mode, env.end_redos
    sim.redo_deadline = env.redo_deadline
    sim.show_score, sim.end_feedback = env.show_score, env.end_feedback
    sim.state.stage_amt = env.stage_amt  # from the next set: apply runs between sets


# Total reward earned over `horizon` ticks in `env`. Time away counts against it.
def rollout(state, params, env, horizon, seed):
    # Starts at a decision point, so not Simulation.run: that grants a free first episode.
    sim = fork(state, params, env)
    st = sim.state
    random.seed(seed)
    np.random.seed(seed)
    r0 = st.total_reward
    for _ in range(horizon):
        sim.tick()
    return st.total_reward - r0


@dataclass
class Update:
    decision: int
    estimate: Estimate
    previous: Environment
    chosen: Environment
    reason: str
    scores: dict = field(default_factory=dict)


class Controller:
    def __init__(self, start=Environment(), candidates=CANDIDATES, horizon=80, reps=4,
                 confirm_top=4, confirm_horizon=300, confirm_reps=16,
                 explore=0.15, min_spacing=25, refit_every=REFIT_EVERY, known=None, seed=0):
        self.env = start
        self.candidates = tuple(candidates)
        self.horizon = horizon  # screening: ticks per rollout, cheap
        self.reps = reps        # screening: rollouts per candidate
        self.confirm_top = confirm_top          # front-runners taken through to the re-test
        self.confirm_horizon = confirm_horizon  # the re-test: long rollouts on fresh seeds,
        self.confirm_reps = confirm_reps        # for those and the current setting. Only it decides.
        self.explore = explore  # chance an update picks a random setting instead of the best
        self.min_spacing = min_spacing  # start decisions between plans
        self.refit_every = refit_every  # new decided rows before fitting again
        self.known = known      # (f, k[, patience]), for tests only
        self.rng = random.Random(seed)  # its own generator; never the global one
        self.est = None
        self.rows_at_fit = None
        self.last_plan = None
        self.log = []

    # True once `min_spacing` start decisions have passed since the last plan.
    def due(self, state):
        return (self.last_plan is None
                or len(state.initiation_passages) - self.last_plan >= self.min_spacing)

    # Re-estimate if due, choose a setting, return the Update. `state` must describe the
    # person as measured (see `measured`); `params` supplies only g and a.
    def plan(self, state, params):
        # Saved and restored, so planning cannot perturb the run it is steering.
        saved = random.getstate(), np.random.get_state()
        try:
            return self._plan(state, params)
        finally:
            random.setstate(saved[0])
            np.random.set_state(saved[1])

    def _plan(self, state, params):
        rows = decided_rows(state)
        if self.known is not None:
            f, k = self.known[:2]
            patience = self.known[2] if len(self.known) > 2 else PATIENCE
            self.est = Estimate(f, k, rows=rows, note="known (test)", patience=patience)
        elif self.est is None or rows - self.rows_at_fit >= self.refit_every:
            self.est = estimate(state, self.est)
            self.rows_at_fit = rows

        est = self.est
        model = UserParams(f=snap_value(est.f), k=snap_value(est.k), g=params.g, a=params.a,
                           patience=est.patience)
        previous = self.env
        self.last_plan = len(state.initiation_passages)
        scores = {}

        if self.rng.random() < self.explore:
            chosen = self.rng.choice(self.candidates)
            reason = "exploration probe (random, independent of the person)"
        else:
            # Screening: every candidate, short rollouts, shared seeds.
            seeds = [self.rng.randrange(2**31) for _ in range(self.reps)]
            screen = {env: float(np.mean([rollout(state, model, env, self.horizon, s) for s in seeds]))
                      for env in self.candidates}
            front = sorted(screen, key=screen.get, reverse=True)[:self.confirm_top]

            # Confirmation: the front-runners and the current setting, long rollouts on
            # fresh shared seeds -- short ones are too noisy to trust with a switch.
            finalists = list(dict.fromkeys([*front, previous]))
            seeds = [self.rng.randrange(2**31) for _ in range(self.confirm_reps)]
            per = {env: np.array([rollout(state, model, env, self.confirm_horizon, s) for s in seeds])
                   for env in finalists}
            scores = {env: float(v.mean()) for env, v in per.items()}
            best = max(scores, key=scores.get)

            if best == previous:
                chosen, reason = previous, "current setting is already best"
            else:
                d = per[best] - per[previous]  # paired: same seeds
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                if d.mean() > max(2 * se, 0.0): # switch only on a margin over the noise
                    chosen = best
                    reason = (f"+{d.mean():.1f} reward per {self.confirm_horizon} ticks "
                              f"(±{se:.1f})")
                else:
                    chosen = previous
                    reason = "nothing beats the current setting by more than the noise"

        self.env = chosen
        update = Update(len(state.initiation_passages), est, previous, chosen, reason, scores)
        self.log.append(update)
        return update

    def log_frame(self):
        return pd.DataFrame([{
            "decision": u.decision,
            "rows": u.estimate.rows,
            "f_used": u.estimate.f,
            "k_used": u.estimate.k,
            "patience": u.estimate.patience,
            "t0": u.estimate.t0,
            "estimate": u.estimate.note,
            "to_target": u.chosen.target_p,
            "to_review": u.chosen.review_rate,
            "to_feedback": u.chosen.feedback,
            "to_return_skipped": u.chosen.return_skipped,
            "to_redo_prompts": u.chosen.redo_prompts,
            "switched": u.chosen != u.previous,
            "reason": u.reason,
            "best_score": max(u.scores.values()) if u.scores else None,
        } for u in self.log])


# --- Planning to measure ---------------------------------------------------------------
# Drift is f * value - k * cost, so a decision depends on (f, k) only through its drift mu,
# and its Fisher information about (f, k) is I_mu(mu) * [[v^2, -v c], [-v c, c^2]]. I_mu
# covers choice, RT and undecided past the deadline, from pyddm solutions on a grid of mu.
# A setting is scored by the rise in log det(prior + data so far + its rollouts), which
# favours settings that move value and cost in directions the data has not covered.

PRIOR_SD = 0.5  # prior spread on f and k for the information score
DRAWS = 6       # (f, k) draws the score is averaged over
MU_GRID = np.round(np.arange(-4.0, 4.0001, 0.05), 2)
_I_MU = None


def _outcome_probs(mu):
    sol = solution(float(mu), 0.0, 1.0, 0.0)  # drift = mu * 1 - 0 * 0
    return np.concatenate([sol.pdf("upper") * DT, sol.pdf("lower") * DT,
                           [sol.prob_undecided()]])


# Fisher information of one decision about its drift, on MU_GRID. Computed once.
def i_mu_table(h=0.025):
    global _I_MU
    if _I_MU is None:
        out = []
        for mu in MU_GRID:
            lo, mid, hi = _outcome_probs(mu - h), _outcome_probs(mu), _outcome_probs(mu + h)
            dp = (hi - lo) / (2 * h)
            keep = mid > 1e-12
            out.append(float(np.sum(dp[keep] ** 2 / mid[keep])))
        _I_MU = np.array(out)
    return _I_MU


# Fisher information about (f, k) in a list of decision records.
def fisher(passages, f, k):
    table = i_mu_table()
    info = np.zeros((2, 2))
    for p in passages:
        v, c = float(p["value"]), float(p["cost"])
        i_mu = np.interp(f * v - k * c, MU_GRID, table)
        info += i_mu * np.array([[v * v, -v * c], [-v * c, c * c]])
    return info


def logdet(m):
    return float(np.linalg.slogdet(m)[1])


class InfoController(Controller):
    """Chooses the setting that would most sharpen the (f, k) estimate.

    It plans for every person the data still allows, not one: the score is averaged over
    DRAWS (f, k) from the prior until the fit is used (MIN_ROWS), then from a Laplace
    approximation N(estimate, (prior + data information)^-1). Each draw is rolled out
    with its own parameters. Safety: a setting is eligible only if its rollouts start at
    least `safety` times as many sets as the best setting's.
    """

    def __init__(self, horizon=25, safety=0.5, **kw):
        super().__init__(**kw)
        self.info_horizon = horizon  # ticks per rollout
        self.safety = safety
        self.np_rng = np.random.default_rng(kw.get("seed", 0))

    def _draws(self, est, info):
        if info is None:  # the fit is not used yet: the prior is all there is
            mean, cov = np.array(PRIOR), np.eye(2) * PRIOR_SD ** 2
        else:
            mean = np.array([est.f, est.k])
            cov = np.linalg.inv(np.eye(2) / PRIOR_SD ** 2 + info)
        return np.clip(self.np_rng.multivariate_normal(mean, cov, size=DRAWS), 0.05, 2.5)

    def _plan(self, state, params):
        rows = decided_rows(state)
        if self.known is not None:
            self.est = Estimate(*self.known[:2], rows=rows, note="known (test)")
        elif self.est is None or rows - self.rows_at_fit >= self.refit_every:
            self.est = estimate(state, self.est)
            self.rows_at_fit = rows
        est = self.est
        data = state.initiation_passages + state.first_passage
        if self.known is not None:
            draws = np.array([self.known[:2]] * DRAWS)
        else:
            used = rows >= MIN_ROWS and est.raw is not None
            draws = self._draws(est, fisher(data, est.f, est.k) if used else None)
        bases = [np.eye(2) / PRIOR_SD ** 2 + fisher(data, f, k) for f, k in draws]

        scores, sets = {}, {}
        for env in self.candidates:
            gains, started = [], []
            for (f, k), base in zip(draws, bases):
                model = UserParams(f=snap_value(f), k=snap_value(k), g=params.g, a=params.a,
                                   patience=est.patience)
                sim = fork(state, model, env)
                seed = self.rng.randrange(2**31)
                random.seed(seed)
                np.random.seed(seed)
                e0 = sim.state.episodes
                for _ in range(self.info_horizon):
                    sim.tick()
                ps = sim.state.initiation_passages + sim.state.first_passage
                gains.append(logdet(base + fisher(ps, f, k)) - logdet(base))
                started.append(sim.state.episodes - e0)
            scores[env], sets[env] = float(np.mean(gains)), float(np.mean(started))
        floor = self.safety * max(sets.values())
        chosen = max((e for e in self.candidates if sets[e] >= floor), key=scores.get)

        previous = self.env
        self.env = chosen
        self.last_plan = len(state.initiation_passages)
        update = Update(len(state.initiation_passages), est, previous, chosen,
                        f"information +{scores[chosen]:.2f} over {DRAWS} draws", scores)
        self.log.append(update)
        return update


# --- The app's controller ----------------------------------------------------------------
# InfoController's score over every setting of the app whose effect the model describes,
# with redo prompts under a countdown read at their own bound and deadline. Showing the
# stage count is not among them: the model has no account of it yet.

COUNTDOWNS = (2.0, 3.0, 5.0)  # seconds on a timed redo prompt
STAGE_LEVELS = (2, 4, 6, 8)   # questions per set
LAM_DRAWS = (0.2, 1.2)        # range a drawn λ is clipped to
LAM_STEP = 0.05               # and snapped to, so pressure.py's draw tables are reused


# The settings at one target, set length and score display: InfoController's scheduled
# review ones; redo prompts mid-set with feedback shown or after the score with it hidden
# (and each redo's outcome then shown, as the app does), each untimed or under a
# countdown; and every seen miss redone with no prompt.
def _app_settings(t, n, score):
    common = dict(stage_amt=n, show_score=score)
    scheduled = [Environment(t, r, fb, rs, **common) for r in REVIEW_LEVELS
                 for fb in FEEDBACK_LEVELS for rs in SKIPPED_LEVELS if r > 0 or not rs]
    redo = [Environment(t, 0.0, mid, False, True, not mid, c is not None,
                        REDO_DEADLINE if c is None else c, end_feedback=not mid, **common)
            for mid in (True, False) for c in (None, *COUNTDOWNS)]
    forced = [Environment(t, 0.0, True, False, forced=True, **common)]
    return scheduled + redo + forced


# 6 targets x 4 set lengths x score shown or hidden x 23 settings each = 1104.
APP_CANDIDATES = tuple(e for t in TARGETS for n in STAGE_LEVELS for score in (True, False)
                       for e in _app_settings(t, n, score))


# (f, k, T0, λ) for planning: `estimate`, fitted in closed form so that redo prompts under
# a countdown are read at bound λ·BOUND and their own deadline (inference.fit_closed_form).
def estimate_closed_form(state, previous=None):
    passages = state.initiation_passages + state.first_passage
    go, *_, deadline = decision_arrays(passages)
    n = int((go >= 0).sum())
    timed = int((~np.isnan(deadline)).sum())
    if n < MIN_ROWS:
        est = Estimate(*PRIOR, rows=n, note=f"prior only ({n}/{MIN_ROWS} rows)")
    else:
        raw = fit_closed_form(passages)
        f_hat, k_hat, t0_hat, lam_hat = raw
        lo, hi = F_BOUNDS
        if min(f_hat, k_hat) <= lo + 0.01 or max(f_hat, k_hat) >= hi - 0.01: # hit the search bound
            base = previous or Estimate(*PRIOR)
            est = Estimate(base.f, base.k, rows=n, raw=raw, t0=t0_hat, lam=base.lam,
                           note="f/k fit hit a bound, kept previous")
        else:
            w = n / (n + SHRINK_ROWS) # weight on the fit against the prior
            est = Estimate(w * f_hat + (1 - w) * PRIOR[0], w * k_hat + (1 - w) * PRIOR[1],
                           rows=n, raw=raw, t0=t0_hat, lam=lam_hat,
                           note=f"{w:.0%} fit / {1 - w:.0%} prior; T0 {t0_hat:.2f}s; "
                                f"λ {lam_hat:.2f} from {timed} timed")
    est.patience, note = estimate_patience(state, est.f, est.k)
    est.note += "; " + note
    return est


# Fisher information of one decision under a countdown about its drift μ and its bound b,
# at bound `bound` and deadline `t_dur`, on MU_GRID: (I_μμ, I_μb, I_bb), in closed form.
def countdown_info_table(bound, t_dur, h=0.01):
    def probs(mu, b):
        up, lo, und = fastlik.outcome_probs(mu, b, t_dur=t_dur)
        return np.concatenate([up, lo, und[:, None]], axis=1)
    mid = probs(MU_GRID, bound)
    d_mu = (probs(MU_GRID + h, bound) - probs(MU_GRID - h, bound)) / (2 * h)
    d_b = (probs(MU_GRID, bound + h) - probs(MU_GRID, bound - h)) / (2 * h)
    w = 1.0 / np.maximum(mid, 1e-300)
    return tuple(np.sum(x * y * w, axis=1) for x, y in ((d_mu, d_mu), (d_mu, d_b), (d_b, d_b)))


# log det of the information about (f, k) with λ a nuisance: the Schur complement.
def fk_logdet(m):
    return logdet(m[:2, :2] - np.outer(m[:2, 2], m[:2, 2]) / m[2, 2])


class AppController(InfoController):
    """The controller the app runs: InfoController over APP_CANDIDATES, countdown-aware.

    - The estimate is fitted in closed form, with λ (`estimate_closed_form`).
    - A setting's score is the rise in log det of the information about (f, k) with λ a
      nuisance, which is InfoController's score when no decision is timed. A timed
      decision informs (μ, b) at its own deadline, with b = λ·BOUND.
    - (f, k, λ) are drawn from N(estimate, (prior + data information)^-1), as
      InfoController draws (f, k).
    - Planning screens every setting on `screen_draws` of the draws, then scores the
      `finalists` best on all of them, within the safety floor each time. Every setting
      is rolled out on the same seeds, so settings are compared on the same luck.
    """

    def __init__(self, candidates=APP_CANDIDATES, screen_draws=2, finalists=24, **kw):
        super().__init__(candidates=candidates, **kw)
        self.screen_draws = screen_draws
        self.finalists = finalists
        self.tables = {}

    def _table(self, lam, t_dur):
        key = (round(lam * BOUND, 4), float(t_dur))
        if key not in self.tables:
            self.tables[key] = countdown_info_table(*key)
        return self.tables[key]

    # Information about (f, k, λ) in a list of decision records, for one draw.
    def information(self, passages, f, k, lam):
        m = np.zeros((3, 3))
        if not passages:
            return m
        v = np.array([p["value"] for p in passages], dtype=float)
        c = np.array([p["cost"] for p in passages], dtype=float)
        d = np.array([p.get("deadline") or np.nan for p in passages], dtype=float)
        mu = f * v - k * c
        i_mm = np.interp(mu, MU_GRID, i_mu_table())
        i_mb, i_bb = np.zeros_like(mu), np.zeros_like(mu)
        for t in np.unique(d[~np.isnan(d)]):
            rows = d == t
            mm, mb, bb = self._table(lam, t)
            i_mm[rows] = np.interp(mu[rows], MU_GRID, mm)
            i_mb[rows] = np.interp(mu[rows], MU_GRID, mb)
            i_bb[rows] = np.interp(mu[rows], MU_GRID, bb)
        # μ moves with (f, k, λ) as (v, -c, 0); a timed decision's bound as (0, 0, BOUND).
        m[0, 0] = np.sum(i_mm * v * v)
        m[0, 1] = m[1, 0] = -np.sum(i_mm * v * c)
        m[1, 1] = np.sum(i_mm * c * c)
        m[0, 2] = m[2, 0] = BOUND * np.sum(i_mb * v)
        m[1, 2] = m[2, 1] = -BOUND * np.sum(i_mb * c)
        m[2, 2] = BOUND ** 2 * np.sum(i_bb)
        return m

    def _plan(self, state, params):
        rows = decided_rows(state)
        if self.known is not None:
            lam = self.known[2] if len(self.known) > 2 else LAM_PRIOR[0]
            self.est = Estimate(*self.known[:2], rows=rows, lam=lam, note="known (test)")
        elif self.est is None or rows - self.rows_at_fit >= self.refit_every:
            self.est = estimate_closed_form(state, self.est)
            self.rows_at_fit = rows
        est = self.est
        data = state.initiation_passages + state.first_passage
        prior = np.diag([PRIOR_SD ** -2, PRIOR_SD ** -2, LAM_PRIOR[1] ** -2])
        if self.known is not None:
            draws = np.array([[est.f, est.k, est.lam]] * DRAWS)
        else:
            if rows >= MIN_ROWS and est.raw is not None:
                mean = np.array([est.f, est.k, est.lam])
                cov = np.linalg.inv(prior + self.information(data, *mean))
            else:  # the fit is not used yet: the prior is all there is
                mean, cov = np.array([*PRIOR, LAM_PRIOR[0]]), np.linalg.inv(prior)
            draws = self.np_rng.multivariate_normal(mean, cov, size=DRAWS)
            draws[:, :2] = np.clip(draws[:, :2], 0.05, 2.5)
            draws[:, 2] = np.round(np.clip(draws[:, 2], *LAM_DRAWS) / LAM_STEP) * LAM_STEP
        bases = [prior + self.information(data, *d) for d in draws]
        seeds = [self.rng.randrange(2**31) for _ in draws]  # shared by every setting

        # (score, sets started) of `env`, averaged over the draws numbered in `which`.
        def evaluate(env, which):
            gains, started = [], []
            for j in which:
                f, k, lam = draws[j]
                model = UserParams(f=snap_value(f), k=snap_value(k), g=params.g, a=params.a,
                                   patience=est.patience, pressure=float(lam))
                sim = fork(state, model, env)
                random.seed(seeds[j])
                np.random.seed(seeds[j])
                e0 = sim.state.episodes
                for _ in range(self.info_horizon):
                    sim.tick()
                ps = sim.state.initiation_passages + sim.state.first_passage
                gains.append(fk_logdet(bases[j] + self.information(ps, f, k, lam))
                             - fk_logdet(bases[j]))
                started.append(sim.state.episodes - e0)
            return float(np.mean(gains)), float(np.mean(started))

        # Settings above the safety floor, best first.
        def ranked(results):
            floor = self.safety * max(n for _, n in results.values())
            return sorted((e for e in results if results[e][1] >= floor),
                          key=lambda e: results[e][0], reverse=True)

        screen = {env: evaluate(env, range(min(self.screen_draws, DRAWS)))
                  for env in self.candidates}
        final = {env: evaluate(env, range(DRAWS)) for env in ranked(screen)[:self.finalists]}
        chosen = ranked(final)[0]

        previous = self.env
        self.env = chosen
        self.last_plan = len(state.initiation_passages)
        update = Update(len(state.initiation_passages), est, previous, chosen,
                        f"information +{final[chosen][0]:.2f} over {DRAWS} draws, best of "
                        f"{len(final)} finalists from {len(screen)} settings",
                        {e: s for e, (s, _) in final.items()})
        self.log.append(update)
        return update


# Entry point for a separate process (the app plans off the UI thread).
def plan_in_worker(controller, state, params):
    update = controller.plan(state, params)
    return controller, update


# A simulated person, measured the way the interface measures a human: the controller
# only ever sees the Elo rating built from their answers, never their true difficulty.
# It runs until `n_decisions` start decisions are recorded, or with `n_ticks` for that
# many ticks: a fixed amount of time, in which a person who stops more is away more and
# gives fewer decisions. `each`, if given, is called with the simulation after every tick
# (and the set it started), before the controller plans: for recording a run as it goes.
def run(params, controller=None, env=Environment(), n_decisions=400, seed=0,
        reentry_cost=0.6, type_diffs=None, ability=0.0, spread=1.0, skips=True, each=None,
        n_ticks=None):
    random.seed(seed)
    np.random.seed(seed)
    start = controller.env if controller else env
    ratings = Ratings()
    sim = Simulation(params, reentry_cost=reentry_cost, review_rate=start.review_rate,
                     review_mode=start.review_mode, target_p=start.target_p,
                     feedback=start.feedback, ratings=ratings, skips=skips,
                     return_skipped=start.return_skipped, end_redos=start.end_redos,
                     redo_deadline=start.redo_deadline, show_score=start.show_score,
                     end_feedback=start.end_feedback)
    st = sim.state
    st.stage_amt = start.stage_amt
    truth = type_diffs if type_diffs is not None else synthetic_type_diffs(spread)
    st.type_diffs = np.array(truth, dtype=float) # hidden from the controller
    st.ability = ability

    st.in_session = True
    sim.run_episode()  # already in the task, as Simulation.run does
    end = None if n_ticks is None else st.t + n_ticks
    while (st.t < end) if end is not None else (len(st.initiation_passages) < n_decisions):
        sim.tick()
        if each is not None:
            each(sim)
        if controller is not None and controller.due(st):
            controller.plan(measured(st, ratings), params)
            apply(sim, controller.env)
    return sim


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--values", default="0.8,0.8", help="true f,k of the simulated person")
    p.add_argument("--decisions", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    f, k = (float(x) for x in args.values.split(",")[:2])
    params = UserParams(f=f, k=k)

    fixed = run(params, env=Environment(), n_decisions=args.decisions, seed=args.seed)
    ctrl = Controller(seed=args.seed, refit_every=60)
    steered = run(params, controller=ctrl, n_decisions=args.decisions, seed=args.seed)
    info = InfoController(seed=args.seed)
    measuring = run(params, controller=info, n_decisions=args.decisions, seed=args.seed)

    print(f"person f={f} k={k}, {args.decisions} decisions, seed {args.seed}\n")
    print(f"{'':>12} {'reward':>7} {'sets':>6} {'skips':>6} {'corrections':>12} {'ability':>8}")
    for name, sim in (("fixed", fixed), ("reward", steered), ("measure", measuring)):
        st = sim.state
        print(f"{name:>12} {st.total_reward:7.1f} {st.episodes:6d} {st.skips:6d} "
              f"{st.corrections:12d} {st.ability:8.2f}")
    print(f"(fixed = {Environment()})")
    for name, c in (("reward", ctrl), ("measure", info)):
        if c.log:
            print(f"\n{name} controller updates ({len(c.log)}):")
            with pd.option_context("display.width", 240, "display.max_colwidth", 70):
                print(c.log_frame()[["decision", "rows", "f_used", "k_used", "to_target",
                                     "to_review", "to_feedback", "to_return_skipped",
                                     "to_redo_prompts", "switched", "reason"]]
                      .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
