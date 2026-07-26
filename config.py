import math
import random
from dataclasses import dataclass, field

import numpy as np


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
    skill: float # initial preformance that will scale
    stage_amt: int = 4 # How many stages there are until reward
    diff: float = 0.1 # The difficulty of each stage
    theta: np.ndarray = None
    first_passage: list = field(default_factory=list)
    rpe: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.theta is None:
            self.theta = np.zeros(len(phi(0, self)))
        if not self.rpe:
            self.rpe = {r: 0 for r in range(self.stage_amt)}
    

def get_sigma(state: ModelState, base_sigma=0.02, scaling_factor=0.3):
    raw_sigma = base_sigma + state.diff * scaling_factor + 0.1
    sigma = raw_sigma / math.sqrt(state.skill) if state.skill != 0 else raw_sigma
    return sigma


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



def drift_rate(delta, f_val, b_val, timestep):
    reward_signal = np.log1p(np.maximum(10 * (f_val * delta), -1 + 1e-9))
    boredom = np.log1p(b_val * timestep) 
    # boredom pushes toward disengagement; learning progress (reward) pushes back
    return  boredom - reward_signal

def value_of_stage(state: ModelState, p: UserParams, s = None):
    s = state.stage_amt - 1 if None else s
    V_s = V(state, s)
    if s == state.stage_amt - 1:
        sigma = get_sigma(state)
        r = sigmoid(5 * (state.skill - state.diff)) + random.gauss(0, sigma)
        r = max(0, min(1, r))
        V_next = 0.0
    else:
        r, V_next = 0.0, V(state, s + 1)

    delta = r + p.g * V_next - V_s
    state.theta = state.theta + p.a * delta * phi(s, state)
    return delta

def choose_difficulty(sim, max_disengage_prob=0.3):
    grid=np.linspace(0.05, 0.5, 20)
    state = sim.state
    best = grid[0]
    for diff in grid:
        state.difficulty = diff
        _delta, _, _, sol = sim.useddm(state.stage_amt - 1)
        if sol.prob('correct') <= max_disengage_prob:
            best = diff
    return best