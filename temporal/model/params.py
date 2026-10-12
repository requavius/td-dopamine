"""The person (`UserParams`) and everything the environment knows about them (`ModelState`)."""
import random
from dataclasses import dataclass, field

import numpy as np

# Cold-start cost: opening the task at all, from outside a session.
REENTRY_COST_LOW = 0.3
REENTRY_COST_MID = 0.6
REENTRY_COST_HIGH = 1.0
COST_PROFILES = {"low": REENTRY_COST_LOW, "mid": REENTRY_COST_MID, "high": REENTRY_COST_HIGH}

ABILITY_START = 0.5  # = DIFFICULTY_SCALE * default diff, so P(success) starts at 0.5
PATIENCE = 10.0      # default seconds per unit of model time while working on a question
N_FEATURES = 2       # value.phi: intercept and slot position


@dataclass
class UserParams:
    # default_factory, not a bare call: one draw per instance, not one per process.
    f: float = field(default_factory=lambda: random.uniform(0, 1.0))  # reward sensitivity
    k: float = field(default_factory=lambda: random.uniform(0, 1.0))  # effort aversion
    g: float = 0.9   # discount factor
    a: float = 0.05  # learning rate
    patience: float = PATIENCE  # seconds per unit of model time; small gives up fast
    t0: float = 0.0  # non-decision time added to every RT


@dataclass
class ModelState:
    t: int
    ability: float = ABILITY_START  # competence, on the logit scale
    stage_amt: int = 4              # questions (slots) per set
    diff: float = 0.1               # difficulty of every slot on the single-diff task
    stage_diffs: np.ndarray = None  # per-slot difficulty; None means every slot uses `diff`
    # Difficulty of each question type (questions.TYPES) in logits. When set, an item is
    # a type, and `diff` / `stage_diffs` are unused.
    type_diffs: np.ndarray = None
    reentry_cost: float = REENTRY_COST_LOW  # cold-start cost of resuming
    theta: np.ndarray = None                # TD value weights over value.phi
    in_session: bool = False                # whether the learner is currently "in" the task

    # What the last set was like, for the next start decision.
    last_episode_passed: float = 0.0  # fraction of questions answered correctly
    last_achievement: float = 0.0     # right answers weighted by (1 - p), unscaled
    last_episode_effort: float = 0.0  # in the simulator, 1 + review load
    last_review_load: float = 0.0     # fraction of slots spent revisiting
    # Descriptive only: logged and read back in the audit, but no longer in any drift --
    # the outcomes they summarise reach the accumulator as prediction error instead.
    last_feedback: int = 1
    last_seen_miss: float = 0.0
    last_unseen_miss: float = 0.0
    # Prediction error, carried. `slot_delta` is the error at the slot just left, for the
    # decisions taken mid-set; `last_set_delta` is the whole last set's, for the decision
    # to start another one.
    slot_delta: float = 0.0
    last_set_delta: float = 0.0

    challenge: float = 0.0     # running total of (1 - p) over every question attempted
    total_reward: float = 0.0  # running total of (1 - p) over right answers
    # Items answered wrong and not yet put right. A list, not a set: the same item can
    # be missed in two episodes, and both are owed.
    missed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)  # given up on; returned only if that setting is on
    skips: int = 0        # questions given up on
    corrections: int = 0  # missed or skipped questions later answered correctly
    stay_streak: int = 0  # consecutive non-GO draws; at SESSION_EXIT_STREAK the session ends
    attempts: int = 0     # non-GO draws since the last GO; not reset by a session ending

    # The records the fit reads.
    first_passage: list = field(default_factory=list)        # redo decisions
    initiation_passages: list = field(default_factory=list)  # start decisions
    persistence: list = field(default_factory=list)          # one per question worked on

    episodes: int = 0   # completed sets (reward delivered)
    stuck: int = 0      # start decisions that came back STAY or timed out
    abandoned: int = 0  # sets walked out of
    rpe: dict = field(default_factory=dict)  # last TD error at each slot

    def __post_init__(self):
        if self.theta is None:
            self.theta = np.zeros(N_FEATURES)
        if not self.rpe:
            self.rpe = {r: 0 for r in range(self.stage_amt)}
