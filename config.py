import random
from dataclasses import dataclass, field

import numpy as np

# Cold-start cost: opening the task at all, from outside a session.
REENTRY_COST_LOW = 0.3
REENTRY_COST_MID = 0.6
REENTRY_COST_HIGH = 1.0

COST_PROFILES = {
    "low": REENTRY_COST_LOW,
    "mid": REENTRY_COST_MID,
    "high": REENTRY_COST_HIGH,
}

WARM_COST = 0.05    # already sitting there, next set one click away
EFFORT_WEIGHT = 0.1 # scales last_episode_effort (1 + review load) into cost units

# Repeat-vs-continue cost: expected attempts to pass the stage (1/p), scaled and capped.
REPEAT_COST_SCALE = 0.15
MAX_REPEAT_ATTEMPTS = 8.0

SESSION_EXIT_STREAK = 3 # consecutive non-GO draws before the learner has left
MAX_EPISODE_STEPS = 40  # cap on stage attempts within one episode (interface.py only)

DIFFICULTY_SCALE = 5.0
CORRECTION_GAIN = 0.06  # ability gained per corrected question; see update_ability
ABILITY_START = 0.5 # = DIFFICULTY_SCALE * default diff, so P(success) starts at 0.5

# Scales a set's (1 - p)-weighted right answers so a perfect set is worth 1.0 per question.
ACHIEVEMENT_SCALE = 4.0

# The prediction error carried out of a slot, or a whole set, into the next decision.
# Pure RPE: the carry weight is 1 by construction, not a fitted parameter. The clip only
# keeps `value` on a bounded grid so draws still hit initiation._SOLUTION_CACHE; measured
# deltas sit well inside it (see README, "Feedback is timing, not reward").
DELTA_CLIP = 1.0

PATIENCE = 10.0 # default seconds per unit of model time while working on a question

# True: a set's last slot bootstraps from V(0), so the task is a cycle. False: from 0.
CONTINUING = True


@dataclass
class UserParams:
    # default_factory, not a bare call: one draw per instance, not one per process.
    f: float = field(default_factory=lambda: random.uniform(0, 1.0)) # Sensitivity to learning progress
    k: float = field(default_factory=lambda: random.uniform(0, 1.0)) # Effort aversion
    g: float = 0.9 # discount factor
    a: float = 0.05 # learning rate
    patience: float = PATIENCE # seconds per unit of model time; small gives up fast
    t0: float = 0.0 # non-decision time added to every RT


@dataclass
class ModelState:
    t: int
    ability: float = ABILITY_START # competence, on the logit scale
    stage_amt: int = 4 # How many stages there are until reward
    diff: float = 0.1 # The difficulty of each stage
    stage_diffs: np.ndarray = None # per-stage difficulty; None means every stage uses `diff`
    # Difficulty of each question type (questions.TYPES) in logits. When set, an item is
    # a type, and `diff` / `stage_diffs` are unused.
    type_diffs: np.ndarray = None
    reentry_cost: float = REENTRY_COST_LOW # cold-start cost of resuming
    theta: np.ndarray = None
    in_session: bool = False # whether the learner is currently "in" the task
    last_episode_passed: float = 0.0 # fraction of questions answered correctly last episode
    last_achievement: float = 0.0 # last episode's right answers weighted by (1 - p), unscaled
    last_episode_effort: float = 0.0 # effort of the last episode; in the simulator, 1 + review load
    last_review_load: float = 0.0 # fraction of last episode's slots spent revisiting
    # What the last set was like. Descriptive only: these are logged and read back in
    # the audit, and no longer enter any drift -- the outcomes they summarise reach the
    # accumulator as prediction error instead (see `slot_reward`).
    last_feedback: int = 1
    last_seen_miss: float = 0.0
    last_unseen_miss: float = 0.0
    # Prediction error, carried. `slot_delta` is the error at the slot just left, for the
    # decisions taken mid-set; `last_set_delta` is the whole last set's, for the decision
    # to start another one.
    slot_delta: float = 0.0
    last_set_delta: float = 0.0
    challenge: float = 0.0 # running total of (1 - p) over every question attempted
    total_reward: float = 0.0 # running total of (1 - p) over right answers; what the controller maximizes
    # Items answered wrong and not yet put right. A list, not a set: the same item can
    # be missed in two episodes, and both are owed.
    missed: list = field(default_factory=list)
    skipped: list = field(default_factory=list) # items given up on; returned only if that setting is on
    skips: int = 0 # questions given up on
    corrections: int = 0 # missed or skipped questions later answered correctly
    stay_streak: int = 0 # consecutive non-GO draws; at SESSION_EXIT_STREAK the session ends
    attempts: int = 0 # non-GO draws since the last GO, for the log; not reset by a session ending
    first_passage: list = field(default_factory=list) # repeat-or-continue decisions (review_mode="choice")
    initiation_passages: list = field(default_factory=list) # one (GO, RT) draw per completed episode
    persistence: list = field(default_factory=list) # one record per question worked on (persistence.py)
    episodes: int = 0 # completed episodes (reward delivered)
    stuck: int = 0 # initiation draws that came back STAY or timed out
    abandoned: int = 0 # episodes that hit MAX_EPISODE_STEPS (interface.py only)
    rpe: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.theta is None:
            self.theta = np.zeros(len(phi(0, self)))
        if not self.rpe:
            self.rpe = {r: 0 for r in range(self.stage_amt)}


def item_diff(state: ModelState, s=None) -> float:
    if s is None or state.stage_diffs is None:
        return state.diff
    return float(state.stage_diffs[s])


# Chance of answering item `s` right. With `type_diffs` set, `s` is a question type.
def success_prob(state: ModelState, s=None) -> float:
    if state.type_diffs is not None and s is not None:
        return float(sigmoid(state.ability - float(state.type_diffs[s])))
    return float(sigmoid(state.ability - DIFFICULTY_SCALE * item_diff(state, s)))


def draw_outcome(state: ModelState, s=None) -> int:
    return 1 if random.random() < success_prob(state, s) else 0


# Ability moves only when a missed question is shown again and answered right.
# `review` says it is that question coming back, not a fresh one of the same type.
def update_ability(state: ModelState, item: int, success: int, review: bool) -> None:
    if review:
        if success:
            state.missed.remove(item) # the debt is paid
            state.ability += CORRECTION_GAIN
            state.corrections += 1
        return # missed again: still owed, and nothing is learned
    if not success:
        state.missed.append(item) # a new question missed is owed from now on


# A question they gave up on came back and was answered.
def settle_skipped(state: ModelState, item: int, success: int) -> None:
    state.skipped.remove(item)
    if success:
        state.ability += CORRECTION_GAIN # a correction, as for a missed question
        state.corrections += 1
    else:
        state.missed.append(item) # from now on it is owed like any miss


def sigmoid(z):
    z = np.clip(z, -60, 60)
    sig: np.ndarray = 1 / (1 + np.exp(-z))
    return sig


# (structural, effort) cost of the next initiation decision.
def cost_components(state: ModelState) -> tuple:
    base = WARM_COST if state.in_session else state.reentry_cost # friction of getting to the task
    return base, EFFORT_WEIGHT * state.last_episode_effort # what the last episode took out of them


def current_cost(state: ModelState) -> float:
    base, effort = cost_components(state)
    return base + effort


# Prediction error as the next decision sees it. Pure RPE: full weight, clipped only to
# keep the condition grid bounded.
def carried_delta(delta: float) -> float:
    return float(np.clip(delta, -DELTA_CLIP, DELTA_CLIP))


# What arrives at a slot, on the achievement scale, given the set's achievement so far
# and how much of it has already been paid. With feedback shown, a question's worth lands
# the moment its outcome does; with it hidden nothing lands until the end-of-set score,
# which pays the set in one lump. A set totals the same either way -- feedback moves when
# the signal arrives, not how much, and that timing is the whole of its effect.
# Returns (reward, achievement now credited).
def slot_reward(state: ModelState, achievement: float, credited: float,
                feedback: bool, last: bool) -> tuple:
    due = achievement if (feedback or last) else credited
    return ACHIEVEMENT_SCALE * (due - credited) / state.stage_amt, due


# (value, cost) for the repeat-or-continue decision after a miss: `s` is the slot,
# for discounting, and `item` the question, for difficulty.
def stage_conditions(state: ModelState, p: UserParams, s: int, item=None) -> tuple:
    remaining = state.stage_amt - 1 - s
    # This slot's discounted share of the reward, plus the error just taken at it: the
    # node fires on a miss the person was shown, so that error is the miss landing.
    value = (1.0 / state.stage_amt) * (p.g ** remaining) + carried_delta(state.slot_delta)
    p_item = success_prob(state, s if item is None else item)
    expected_attempts = min(1.0 / max(p_item, 1e-3), MAX_REPEAT_ATTEMPTS) # cost of insisting on it
    return value, REPEAT_COST_SCALE * expected_attempts


def phi(s: int, state: ModelState):
    # Slot position only. `diff` was a second feature, but it never varies during
    # learning, so theta[1] collapsed to theta[0]*d -- collinear with the intercept.
    s_norm = s / (state.stage_amt - 1)
    return np.array([1.0, s_norm])


def V(state: ModelState, s):
    v = float(state.theta @ phi(s, state))
    return v


# What the next set is worth at the moment of deciding: the learned value of starting
# one, plus the surprise of the set just played. A set that went worse than predicted
# carries its negative error into the decision to start another.
def value_at_choice_point(state: ModelState) -> float:
    return V(state, 0) + carried_delta(state.last_set_delta)


def value_of_stage(state: ModelState, p: UserParams, s, reward=0.0):
    V_s = V(state, s)
    if s < state.stage_amt - 1:
        V_next = V(state, s + 1)
    else:
        V_next = V(state, 0) if CONTINUING else 0.0 # bootstrap the last slot

    delta = reward + p.g * V_next - V_s # TD error
    state.theta = state.theta + p.a * delta * phi(s, state)
    return delta
