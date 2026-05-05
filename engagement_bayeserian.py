import random
import math
import numpy as np
from config import *
from scipy.integrate import nsum    

def ddm(f_val, k_val, b_val, delta, s, state: ModelState, prob = False):
    boundary = (1-b_val)*4
    startpos = k_val * 2
    
    dt = 1/stage_amt
    decay_rate = 0.3 
    if state.t > 1:
        delta = state.stage_log[(state.t-1)*stage_amt-1]['delta'] * math.exp(-decay_rate * s)
    
    dW = random.gauss() * math.sqrt(dt) # noise * tiny change in time
    sigma = .01
    
    fatigue = .3 * math.log1p(state.t)
    reward_signal = 10*(f_val * delta)
    
    
    drift_rate = fatigue - reward_signal
    time = state.t + dt*s
    if s > 0 and len(state.stage_log) > 0:
        position = state.stage_log[-1]['position'] + drift_rate*dt + sigma*dW
    else:
        position = startpos + drift_rate*dt + sigma*dW
    if prob:
        return boundary, sigma, drift_rate, startpos
    elif position < boundary:
        return time, False, position
    elif position >= boundary:
        # print(f"delta={delta: 3f} noise={.1*dW: 3f}")
        return time, True, position
    
def formula(f_val, k_val, b_val, s, delta, state: ModelState):
    time = state.t + (s / stage_amt)
    boundary, sigma, drift, startpos = ddm(f_val, k_val, b_val, delta, s, state, prob=True)

    mask = time < (0.5 * boundary**2) / (np.pi**2)

    exponent = -(drift * startpos * boundary) - (drift**2 * time) / 2
    K = 20
    k = np.arange(-K, K + 1).reshape(-1, 1)
    small_terms = (2 * boundary * k + startpos) * np.exp(-((2 * boundary * k + startpos)**2) / (2 * time))
    small_summ = small_terms.sum(axis=0)
    small_result = (boundary / np.sqrt(2 * np.pi * time**3)) * np.exp(exponent) * small_summ

    k = np.arange(1, K + 1).reshape(-1, 1)
    large_terms = k * np.sin(k * np.pi * startpos) * np.exp(-(k**2 * np.pi**2 * time) / (2 * boundary**2))
    large_summ = large_terms.sum(axis=0)
    large_result = (np.pi / boundary**2) * np.exp(exponent) * large_summ

    return np.where(mask, small_result, large_result)
        

def bayesian_particle_update(disengaged, delta, s, state: ModelState):

    probs = formula(state.particle_matrix[0], state.particle_matrix[1], state.particle_matrix[2], s, delta, state)

    likelihoods = probs if not disengaged else (1 - probs)

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
            state.particle_matrix[s] = np.clip(state.particle_matrix[s][indices] + np.random.normal(0, 0.02, n), 0.05, 1.0) 
        state.weights = np.ones(n) / n


