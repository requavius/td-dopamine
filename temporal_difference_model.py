import numpy as np
import random
import math
import matplotlib.pyplot as plt
from engagement_bayeserian import *
from config import *


def value_of_stage(state: ModelState, s, pers_param):
    V_s = V(state.theta, s)
    if s == stage_amt - 1:
        sigma = get_sigma(state)
        reward_divergence = random.gauss(0, sigma)
        r = sigmoid(5 * (state.skill - diff)) + reward_divergence
        r = max(0, min(1, r))
        #r = 1
        V_next = 0.0
    else:
        r, V_next = 0.0, V(state.theta, s+1)

    delta = r + g * V_next - V_s
    state.theta = state.theta + a * delta * phi(s)
    # engagment_prob = sigmoid(stage_engagement)
    dt, engaged_observation, position = ddm( pers_param.f, pers_param.k, pers_param.b, delta, s, state) 
    stage_engagement = formula( pers_param.f, pers_param.k, pers_param.b, delta, s, state) 
    # state.weights = bayesian_particle_update(engaged_observation, delta, V_s, state)
    resample_if_needed(state)

    state.stage_log.append({
        'trial': state.t,
        'stage': s,
        'dt': dt,
        # 'engagement_score': stage_engagement,
        # 'engagement_prob': engagment_prob,
        'engaged_obs': engaged_observation,
        'delta': delta,
        'V': V_s,
        'position' : position
    })
    
    learning_gain = max(0,delta) * (1.0 - state.skill)  # skill grows with practice but saturates. This might be changed based on what makes sense for skill improvment
    state.skill += (min(learning_gain / stage_amt, 1))
    return delta, engaged_observation, stage_engagement

def simulate(state: ModelState, pers_param):
    stages_completed = 0
    for s in range(stage_amt):
        state.rpe[s], disengaged, tot_stage_engagement = value_of_stage(state, s, pers_param)
        stages_completed += 1
        if disengaged: break
    state.episode_log.append({
        'trial': state.t,
        'Stages completed': stages_completed, # how many stages were completed before disengagement
        'max_abs_rpe': max(abs(x) for x in state.rpe.values()),
        'est_f': np.dot(state.weights, state.particle_matrix[0]),
        'est_k': np.dot(state.weights, state.particle_matrix[1]),
        'est_b': np.dot(state.weights, state.particle_matrix[2]),
    })
    return True if not disengaged else False

def train(state: ModelState, pers_param, debug):
    low_rpe_streak = 0
    while True:
        engaged = simulate(state, pers_param)

        max_rpe = max(abs(x) for x in state.rpe.values())

        if max_rpe < 0.05:
            low_rpe_streak += 1
        else:
            low_rpe_streak = 0

        average_v = sum(V(state.theta, s) for s in range(stage_amt))/stage_amt
        
        if low_rpe_streak >= 10 and average_v > 0.1 or not engaged or state.t >= 2000:
            if debug:
                pass
            return 

    
        state.t += 1
        
        
def test_train(true_f = None, true_k = None, true_b = None, debug = False, extra = False):
    if true_f is None: true_f = random.uniform(0.05, 1.0)
    if true_k is None: true_k = random.uniform(0.05, 1.0)
    if true_b is None: true_b = random.uniform(0.05, 1.0)
    
    fixed = UserParams(f=true_f, k=true_k, b=true_b)
    n_particles = 100
    weights = np.ones(n_particles) / n_particles
    theta = np.zeros(len(param_values) + 1)

    state = ModelState(
        theta = theta,
        t = 1,
        weights = weights,
        particle_matrix = np.random.uniform(0.05, 1.0, size=(3, n_particles)),
        skill = .1,
    )

    train(state, fixed, debug)
    
    avg_stages = sum(ep['Stages completed'] for ep in state.episode_log) / len(state.episode_log) if len(state.episode_log) != 0 else 0
    
    if debug == True:
        print(f"stopped after {state.t} trials")
        print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
        print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
        print("Estimated f:", np.dot(state.weights, state.particle_matrix[0]))
        print("Estimated k:", np.dot(state.weights, state.particle_matrix[1]))
        print("Estimated b:", np.dot(state.weights, state.particle_matrix[2]))
        print(f"Average stages completed per episode: {avg_stages:.2f}")
        print("True params:", fixed)
    
    est_f = np.dot(state.weights, state.particle_matrix[0])
    est_k = np.dot(state.weights, state.particle_matrix[1])
    est_b = np.dot(state.weights, state.particle_matrix[2])
    

    
    
    if not extra:
        return {'true_f': true_f, 'true_k': true_k, 'true_b': true_b, 'avg_stages': avg_stages,
                'est_f': est_f, 'est_k': est_k, 'est_b': est_b, 'trials': state.t}
    else:
        log = state.stage_log
        return fixed, log, state.t





