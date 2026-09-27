# The controller: re-fit the person, roll out candidate settings, switch on a margin.
# One simulated person: python controller.py --values 0.8,0.8

import argparse
import copy
import dataclasses
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import PATIENCE, UserParams
from inference import fit, passages_to_frame
from initiation import snap_value
from persistence import attempts_to_frame, fit_patience
from questions import Ratings, synthetic_type_diffs
from temporal_difference_model import Simulation


@dataclass(frozen=True)
class Environment:
    target_p: float = 0.75
    review_rate: float = 0.25
    feedback: bool = True
    return_skipped: bool = False

    def __str__(self):
        return (f"aim {self.target_p:.0%} right, review {self.review_rate:.2f}, "
                f"feedback {'on' if self.feedback else 'off'}, "
                f"skipped {'come back' if self.return_skipped else 'dropped'}")


TARGETS = (0.9, 0.75, 0.6, 0.45, 0.3, 0.2)
REVIEW_LEVELS = (0.0, 0.25, 0.5, 1.0)
FEEDBACK_LEVELS = (True, False)
SKIPPED_LEVELS = (False, True)
CANDIDATES = tuple(Environment(t, r, fb, rs)
                   for t in TARGETS for r in REVIEW_LEVELS for fb in FEEDBACK_LEVELS
                   for rs in SKIPPED_LEVELS if r > 0 or not rs)

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
    sim = Simulation(params, review_mode="scheduled", review_rate=env.review_rate,
                     target_p=env.target_p, feedback=env.feedback,
                     skips=True, return_skipped=env.return_skipped)
    sim.state = st
    return sim


# Total reward earned over `horizon` start decisions in `env`.
def rollout(state, params, env, horizon, seed):
    # Starts at a decision point, so not Simulation.run: that grants a free first episode.
    sim = fork(state, params, env)
    st = sim.state
    random.seed(seed)
    np.random.seed(seed)
    r0 = st.total_reward
    for _ in range(horizon):
        went = sim.initiate()
        st.t += 1
        if went:
            sim.run_episode()
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
        self.horizon = horizon  # screening: start decisions per rollout, cheap
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
                    reason = (f"+{d.mean():.1f} reward per {self.confirm_horizon} decisions "
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
            "switched": u.chosen != u.previous,
            "reason": u.reason,
            "best_score": max(u.scores.values()) if u.scores else None,
        } for u in self.log])


# Entry point for a separate process (interface.py plans off the UI thread).
def plan_in_worker(controller, state, params):
    update = controller.plan(state, params)
    return controller, update


# A simulated person, measured the way the interface measures a human: the controller
# only ever sees the Elo rating built from their answers, never their true difficulty.
def run(params, controller=None, env=Environment(), n_decisions=400, seed=0,
        reentry_cost=0.6, type_diffs=None, ability=0.0, spread=1.0, skips=True):
    random.seed(seed)
    np.random.seed(seed)
    start = controller.env if controller else env
    ratings = Ratings()
    sim = Simulation(params, reentry_cost=reentry_cost, review_rate=start.review_rate,
                     target_p=start.target_p, feedback=start.feedback, ratings=ratings,
                     skips=skips, return_skipped=start.return_skipped)
    st = sim.state
    truth = type_diffs if type_diffs is not None else synthetic_type_diffs(spread)
    st.type_diffs = np.array(truth, dtype=float) # hidden from the controller
    st.ability = ability

    sim.run_episode()  # already in the task, as Simulation.run does
    st.in_session = True
    while len(st.initiation_passages) < n_decisions:
        went = sim.initiate()
        st.t += 1
        if went:
            sim.run_episode()
        if controller is not None and controller.due(st):
            controller.plan(measured(st, ratings), params)
            sim.target_p = controller.env.target_p
            sim.review_rate = controller.env.review_rate
            sim.feedback = controller.env.feedback
            sim.return_skipped = controller.env.return_skipped
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

    print(f"person f={f} k={k}, {args.decisions} decisions, seed {args.seed}\n")
    print(f"{'':>12} {'reward':>7} {'sets':>6} {'skips':>6} {'corrections':>12} {'ability':>8}")
    for name, sim in (("fixed", fixed), ("controller", steered)):
        st = sim.state
        print(f"{name:>12} {st.total_reward:7.1f} {st.episodes:6d} {st.skips:6d} "
              f"{st.corrections:12d} {st.ability:8.2f}")
    print(f"(fixed = {Environment()})")
    if ctrl.log:
        print(f"\ncontroller updates ({len(ctrl.log)}):")
        with pd.option_context("display.width", 240, "display.max_colwidth", 70):
            print(ctrl.log_frame()[["decision", "rows", "f_used", "k_used", "to_target",
                                    "to_review", "to_feedback", "to_return_skipped",
                                    "switched", "reason"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
