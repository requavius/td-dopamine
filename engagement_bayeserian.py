import random
import math
import numpy as np
from config import *

def ddm(f_val, k_val, b_val, delta, s, state: ModelState, prob = False):
    boundary = 1/k_val
    dt = 1/stage_amt
    decay_rate = 0.3 
    if state.t > 1:
        delta = state.stage_log[(state.t-1)*stage_amt-1]['delta'] * math.exp(-decay_rate * s)
    
    dW = random.gauss() * math.sqrt(dt) # noise * tiny change in time
    sigma = .1
    
    boredom = b_val * math.log1p(state.t)
    reward_signal = f_val * delta 
    
    drift_rate = boredom - reward_signal
    time = state.t + dt*s
    try: position = state.stage_log[state.t+s -1]['position'] + drift_rate*dt + sigma*dW
    except: position = 0 + drift_rate*dt + sigma*dW
    if prob:
        return boundary, sigma, drift_rate
    elif position < boundary:
        return time, False, position
    elif position >= boundary:
        # print(f"delta={delta: 3f} noise={.1*dW: 3f}")
        return time, True, position
    
def formula(f_val, k_val, b_val, s, delta, state: ModelState):
    startpos = 0
    boundary, sigma, drift_rate = ddm(f_val, k_val, b_val, delta, s, state, prob = True)
    prob = (boundary - startpos)/(sigma*(math.sqrt(2*math.pi*state.t**3)))*math.exp(-(boundary-startpos-drift_rate*state.t)**2/2*(sigma**2)*state.t)

    

def bayesian_particle_update(engaged, delta, v, state: ModelState):

    scores = formula(state.particle_matrix[0], state.particle_matrix[1], state.particle_matrix[2], delta, state)

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
            state.particle_matrix[s] = np.clip(state.particle_matrix[s][indices] + np.random.normal(0, 0.02, n), 0.05, 1.0) 
        state.weights = np.ones(n) / n


