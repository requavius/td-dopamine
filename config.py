import random
from dataclasses import dataclass, field

import numpy as np

REENTRY_COST_LOW = 0.3
REENTRY_COST_HIGH = 1.0

COST_PROFILES = {"low": REENTRY_COST_LOW, "high": REENTRY_COST_HIGH}

DIFFICULTY_SCALE = 5.0
GAMMA_SUCCESS = 0.04
RHO_FAILURE = 0.12
ABILITY_START = 0.5 # = DIFFICULTY_SCALE * default diff, so P(success) starts at 0.5


@dataclass
class UserParams:
    f: float = random.uniform(0, 1.0) # Sensitivity to learning progress
    k: float = random.uniform(0, 1.0) # Effort aversion
    b: float = random.uniform(0, 1.0) # Boredom rate
    g: float = 0.9 # discount factor
    a: float = 0.05 # learning rate


@dataclass
class ModelState:
    t: int
    ability: float = ABILITY_START # competence, on the logit scale
    stage_amt: int = 4 # How many stages there are until reward
    diff: float = 0.1 # The difficulty of each stage
    reentry_cost: float = REENTRY_COST_LOW # structural cost of resuming
    theta: np.ndarray = None
    # Reserved for the within episode continue/leave node
    first_passage: list = field(default_factory=list)
    # Between-episode re-entry decisions: one (GO, RT) draw per completed episode.
    initiation_passages: list = field(default_factory=list)
    episodes: int = 0 # completed episodes (reward delivered)
    stuck: int = 0 # initiation draws that came back STAY or timed out
    rpe: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.theta is None:
            self.theta = np.zeros(len(phi(0, self)))
        if not self.rpe:
            self.rpe = {r: 0 for r in range(self.stage_amt)}


def success_prob(state: ModelState) -> float:
    return float(sigmoid(state.ability - DIFFICULTY_SCALE * state.diff))


def draw_outcome(state: ModelState) -> int:
    return 1 if random.random() < success_prob(state) else 0


def update_ability(state: ModelState, success: int) -> None:
    state.ability += GAMMA_SUCCESS if success else RHO_FAILURE


def sigmoid(z):
    z = np.clip(z, -60, 60)
    sig: np.ndarray = 1 / (1 + np.exp(-z))
    return sig


def phi(s: int, state: ModelState):
    d = state.diff
    s_norm = s / (state.stage_amt - 1)
    return np.array([1.0, d, s_norm])


def V(state: ModelState, s):
    v = float(state.theta @ phi(s, state))
    return v


def value_at_choice_point(state: ModelState) -> float:
    return V(state, 0)


def value_of_stage(state: ModelState, p: UserParams, s, reward=0.0):
    V_s = V(state, s)
    V_next = 0.0 if s == state.stage_amt - 1 else V(state, s + 1)

    delta = reward + p.g * V_next - V_s
    state.theta = state.theta + p.a * delta * phi(s, state)
    return delta
