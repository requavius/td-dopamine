import random
import math
import numpy as np
from config import *

def formula(f_val, k_val, b_val, v, delta, state: ModelState):
    delta_scale = 10 * (sigmoid(abs(delta)))
    
    signal = (f_val) * delta_scale + 1
    denom = max(0, abs(v) + state.skill) + 1
    effort_cost = ((k_val) * stage_amt + 1) *((state.t_since_eng + 1) + diff * stage_amt) / denom
    boredom_cost = ((b_val)) * (state.t * 0.01)
    
    return signal - effort_cost - boredom_cost

def engagement_score(delta, v, f_val, k_val, b_val, state: ModelState, part = False):
    
    score = formula(f_val, k_val, b_val, v, delta, state)
    
    if score >= state.highest_eng and not part:
        state.t_since_eng = 0
        state.highest_eng = score
    

    return score

def engage(state: ModelState, score):
    prob = sigmoid(score)
    decision = 1 if random.random() < prob else 0
    if not decision: state.highest_eng = -math.inf
    return decision

def bayesian_particle_update(engaged, delta, v, state: ModelState):

    scores = formula(state.particle_matrix[0], state.particle_matrix[1], state.particle_matrix[2], v, delta, state)

    probs: np.ndarray = sigmoid(scores)
    likelihoods = probs if engaged else (1 - probs)

    new_weights = state.weights * np.maximum(likelihoods, 1e-8)
    total = new_weights.sum()
    if total == 0:
        new_weights = np.ones_like(new_weights) / len(new_weights)
    else:
        new_weights /= total

    return new_weights

def resample_if_needed(state: ModelState, threshold=0.5):
    n = len(state.weights)
    ess = 1.0 / np.sum(state.weights ** 2)
    if ess < threshold * n:
        indices = np.random.choice(n, size=n, p=state.weights)
        for s in range(state.particle_matrix.shape[0]):
            p_range = state.particle_matrix[s].max() - state.particle_matrix[s].min()
            state.particle_matrix[s] = np.clip(state.particle_matrix[s][indices] + np.random.normal(0, 0.02, n), 0.05, 1.0) 
            noise = random.gauss(0, p_range * .2 * (np.power(state.particle_matrix.shape[1], (-1)/state.particle_matrix.shape[0])))
            state.particle_matrix[s] += noise 
        state.weights = np.ones(n) / n 


