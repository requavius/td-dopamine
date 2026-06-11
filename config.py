from dataclasses import dataclass, field
import numpy as np
import math


@dataclass(frozen=True)
class Config:
    g: float = 0.9  # discount factor
    a: float = 0.05  # learning rate
    # task parameters (model tweaks):
    stage_amt: int = 4  # How many stages there are until reward
    diff: float = 0.1  # The difficulty of each stage; will not be changed yet until ready for inference

    @property
    def param_values(self) -> dict:
        return {
            "bias": 1,
            "d": self.diff,
        }


config = Config()


@dataclass
class UserParams:
    f: float  # Sensitivity to learning progress
    k: float  # Effort aversion
    b: float  # Boredom rate


@dataclass
class ModelState:
    theta: np.ndarray
    t: int
    weights: np.ndarray
    skill: float  # initial preformance that will scale
    particle_matrix: np.ndarray
    first_passage: list = field(default_factory=list)
    rpe: dict = field(default_factory=lambda: {r: 0 for r in range(config.stage_amt)})


def get_sigma(state: ModelState, base_sigma=0.02, scaling_factor=0.3):
    raw_sigma = base_sigma + config.diff * scaling_factor + 0.1
    sigma = raw_sigma / math.sqrt(state.skill) if state.skill != 0 else raw_sigma
    return sigma


def sigmoid(z):
    z = np.clip(z, -60, 60)
    sig: np.ndarray = 1 / (1 + np.exp(-z))
    return sig


def phi(s: int):
    d = config.diff
    s_norm = s / (config.stage_amt - 1)
    return np.array([1.0, d, s_norm])


def V(theta, s):
    v = float((theta @ phi(s)))
    return v


def drift_rate(delta, f_val, t):
    fatigue = np.log1p(0.3 * t)
    reward_signal = np.log1p(np.maximum(10 * (f_val * delta), -1 + 1e-9))
    return fatigue - reward_signal
