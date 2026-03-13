import copy
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

g = .9 

a =.01

# task parameters (model tweaks):
stage_amt = 4 # How many stages there are until reward
diff = .1 # The difficulty of each stage; will not be changed yet until ready for infererence
pacing = 0.1 # Speed of difficulty escalation 

param_values = {
    'bias' : 1,   
    'd' : diff,
} 

@dataclass
class UserParams:
    f: float  # Sensitivity to learning progress  
    k: float  # Effort aversion  
    b: float  # Boredom rate 


# user parameters (model does not see only estimates)
fixed_params = UserParams(
    f = random.uniform(0.05, 1.0),  # Sensitivity to learning progress  
    k = 0, # Effort aversion  
    b = random.uniform(0.05, 1.0),# Boredom rate 
)

stage = {s: {"V" : 0.0,  ** copy.deepcopy(param_values)} for s in range(stage_amt)} 

@dataclass
class ModelState:
    theta: np.ndarray
    t: int
    weights: np.ndarray
    skill: float  # initial preformance that will scale
    particles: list
    highest_eng: int
    t_since_eng: int
    stage_log: list = field(default_factory=list)
    episode_log: list = field(default_factory=list)
    rpe: dict = field(default_factory=lambda: {r: 0 for r in range(stage_amt)})


def get_sigma(state: ModelState, base_sigma=.02, scaling_factor=.3):
    raw_sigma = base_sigma + diff * scaling_factor + 0.1
    sigma = raw_sigma / math.sqrt(state.skill) if state.skill != 0 else raw_sigma
    return sigma

def engagement_score(delta, v, f_val, k_val, b_val, state: ModelState, part = False):
    signal = f_val * abs(delta)
    denom = max(0.05, abs(v) + state.skill)
    effort_cost = k_val * ((state.t_since_eng) + diff * stage_amt) / denom 
    boredom_cost = b_val * max(0, state.t_since_eng - signal * 10) * 0.01
    
    score = signal - effort_cost - boredom_cost
    if score < 0 and not part:
        print(f"trial: {state.t} signal {signal} - effort {effort_cost} - boredom {boredom_cost}")
    if score >= state.highest_eng and not part:
        state.t_since_eng = 0
        state.highest_eng = score
    return score

def engage(state: ModelState, formula):
    cont = formula
    prob = sigmoid(cont)
    decision = 1 if random.random() < prob else 0
    state.highest_eng = 0 if not decision else state.highest_eng
    return decision

def makeparaguess(paramlist, other = None):
    paramvalues = {}
    for param in paramlist:
        paramvalues[param] = random.uniform(0.05, 1.0)
    if other:
        paramvalues[other] = random.uniform(0.05, 1.0)
    return paramvalues

def phi(s: int):
    d = stage[s]['d']
    s_norm = s / (stage_amt - 1)
    return np.array([1.0, d, s_norm])

def sigmoid(z):
    z = max(-60, min(60, z))
    return 1 / (1 + math.exp(-z))

def bayesian_particle_update(engaged, delta, v, state: ModelState):
    new_weights = state.weights.copy()

    for i, p in enumerate(state.particles):
        f_g, k_g, b_g= p['f'], p['k'], p['b']

        cont = engagement_score(delta, v, f_g, k_g, b_g, state, part = True)
        prob = sigmoid(cont)
        likeh = prob if engaged else (1 - prob)
        new_weights[i] *= max(likeh, 1e-8)

    total = new_weights.sum()
    if total == 0:
        new_weights = np.ones_like(new_weights) / len(new_weights)
    else:
        new_weights /= total

    return new_weights

def V(theta, s): 

    v = float((theta @ phi(s)))
    return v

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
    stage_engagement = engagement_score(delta, V_s, pers_param.f, pers_param.k, pers_param.b, state) 
    engaged_observation = engage(state, stage_engagement)
    engagment_prob = sigmoid(stage_engagement)
    state.weights = bayesian_particle_update(engaged_observation, delta, V_s, state)
    
    state.stage_log.append({
        'trial': state.t,
        'stage': s,
        'engagement_score': stage_engagement,
        'engagement_prob': engagment_prob,
        'engaged_obs': engaged_observation,
        'delta': delta,
        'V': V_s,
    })
    
    learning_gain = 0.02 * (1.0 - state.skill) # skill grows with practice but saturates. This might be changed based on what makes sense for skill improvment
    state.skill += learning_gain / stage_amt
    
    return delta, stage_engagement, engaged_observation

def simulate(state, pers_param):
    tot_epi_engagement = 0
    stages_completed = 0
    for s in range(stage_amt):
        state.rpe[s], tot_stage_engagement, engaged = value_of_stage(state, s, pers_param)
        tot_epi_engagement += tot_stage_engagement
        stages_completed += 1
        if not engaged and state.t > 1:
            print(f"quit after {stages_completed} stages with rpe of {state.rpe[s]}")
            break
        else:
            print(f'kept engagement for {s} stages with rpe of {state.rpe[s]}')
    state.episode_log.append({
        'trial': state.t,
        'total_engagement': tot_epi_engagement, # will be re engagment factor
        'Stages completed': stages_completed, # how many stages were completed before disengagement
        'max_abs_rpe': max(abs(x) for x in state.rpe.values()),
        'est_f': sum(w * p['f'] for w, p in zip(state.weights, state.particles)),
        'est_k': sum(w * p['k'] for w, p in zip(state.weights, state.particles)),
        'est_b': sum(w * p['b'] for w, p in zip(state.weights, state.particles)),
    })
    
    return engaged

def train(state: ModelState, pers_param):
    low_rpe_streak = 0
    while True:
        engaged = simulate(state, pers_param)

        max_rpe = max(abs(x) for x in state.rpe.values())

        if max_rpe < 0.05:
            low_rpe_streak += 1
        else:
            low_rpe_streak = 0

        if low_rpe_streak >= 10:
            print(f"trained after {state.t} trials")
            print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
            print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
            return state.theta, state.t, state.weights

        if state.t >= stage_amt +1 and not engaged:
            print(f"stopped after {state.t} trials")
            print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
            print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
            return state.theta, state.t, state.weights

        state.t += 1
        state.t_since_eng += 1
        
def test_train():
    t = 1  
    N = 100
    particles = [makeparaguess(vars(fixed_params).keys()) for _ in range(N)]
    weights = np.ones(N) / N
    theta = np.zeros(len(param_values) + 1)

    state = ModelState(
        theta = theta,
        t = t,
        weights = weights,
        particles = particles,
        skill = .1,
        highest_eng = 0,
        t_since_eng = 0
    )

    theta, t, weights = train(state, fixed_params)
    print("Estimated f:", sum(w * p['f'] for w, p in zip(state.weights, state.particles)))
    print("Estimated k:", sum(w * p['k'] for w, p in zip(state.weights, state.particles)))
    print("Estimated b:", sum(w * p['b'] for w, p in zip(state.weights, state.particles)))
    print("True params:", fixed_params)
    return t

test_train()
