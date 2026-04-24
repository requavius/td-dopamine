from dataclasses import dataclass, field
import numpy as np
import math

g = .9 
a =.05

# task parameters (model tweaks):
stage_amt = 4 # How many stages there are until reward
diff = .1 # The difficulty of each stage; will not be changed yet until ready for infererence

param_values = {
    'bias' : 1,   
    'd' : diff,
} 

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
    stage_log: list = field(default_factory=list)
    episode_log: list = field(default_factory=list)
    rpe: dict = field(default_factory=lambda: {r: 0 for r in range(stage_amt)})

def get_sigma(state: ModelState, base_sigma=.02, scaling_factor=.3):
    raw_sigma = base_sigma + diff * scaling_factor + 0.1
    sigma = raw_sigma / math.sqrt(state.skill) if state.skill != 0 else raw_sigma
    return sigma

def sigmoid(z):
    z = np.clip(z, -60, 60)
    sig: np.ndarray = 1 / (1 + np.exp(-z))
    return sig

def phi(s: int):
    d = diff
    s_norm = s / (stage_amt - 1)
    return np.array([1.0, d, s_norm])

def V(theta, s):

    v = float((theta @ phi(s)))
    return v

def smoothstep(x):
    return (3*x**2) - (2*x**3)