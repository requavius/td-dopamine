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
    skill: float  # initial preformance that will scale; will not be changed yet until ready for infererence

# user parameters (model does not see only estimates)
fixed_params = UserParams(
    f=.5,  # Sensitivity to learning progress  
    k=.5, #random.gauss(.5, .5/3) # Effort aversion  
    b=.5, #random.gauss(.5, .5/3) # Boredom rate 
    skill=.1 # initial preformance that will scale; will not be changed yet until ready for infererence
)

stage = {s: {"V" : 0.0,  ** copy.deepcopy(param_values)} for s in range(stage_amt)} 

@dataclass
class ModelState:
    theta: np.ndarray
    t: int
    weights: np.ndarray
    particles: list
    stage_log: list = field(default_factory=list)
    episode_log: list = field(default_factory=list)
    rpe: dict = field(default_factory=lambda: {r: 0 for r in range(stage_amt)})

def get_sigma(pers_param, base_sigma=.02, scaling_factor=.3):
    raw_sigma = base_sigma + diff * scaling_factor + 0.1
    sigma = raw_sigma / math.sqrt(pers_param.skill) if pers_param.skill != 0 else raw_sigma
    return sigma

def engagement_score(delta, v, t, f_val, k_val, b_val):
    signal = f_val * abs(delta)
    denom = max(0.05, abs(v))
    effort_cost = k_val * ((t * pacing) + diff * stage_amt) / denom
    boredom_cost = b_val * max(0, t - signal * 10) * 0.01
    return signal - effort_cost - boredom_cost

def engage(delta, v, t, pers_param):
    cont = engagement_score(delta, v, t, pers_param.f, pers_param.k, pers_param.b)
    prob = sigmoid(cont)
    return 1 if random.random() < prob else 0

def makeparaguess(paramlist):
    paramvalues = {}
    for param in paramlist:
        paramvalues[param] = random.uniform(0.05, 1.0)
    return paramvalues

def phi(s: int):
    d = stage[s]['d']
    s_norm = s / (stage_amt - 1)
    return np.array([1.0, d, s_norm])

def sigmoid(z):
    z = max(-60, min(60, z))
    return 1 / (1 + math.exp(-z))

def bayesian_particle_update(engaged, particles, weights, delta, v, t):
    new_weights = weights.copy()

    for i, p in enumerate(particles):
        f_g, k_g, b_g, skill_g = p['f'], p['k'], p['b'], p['skill']

        cont = engagement_score(delta, v, t, f_g, k_g, b_g)
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

def value_of_stage(state, s, pers_param):
    V_s = V(state.theta, s)
    if s == stage_amt - 1:
        sigma = get_sigma(pers_param)
        reward_divergence = random.gauss(0, sigma)
        r = sigmoid(5 * (pers_param.skill - diff)) + reward_divergence
        r = max(0, min(1, r))
        #r = 1
        V_next = 0.0
    else:
        r, V_next = 0.0, V(state.theta, s+1)

    delta = r + g * V_next - V_s
    state.theta = state.theta + a * delta * phi(s)
    state.weights = bayesian_particle_update(engage(delta, V_s, state.t, pers_param), state.particles, state.weights, delta, V_s, state.t)
    stage_engagement = engagement_score(delta, V_s, state.t, pers_param.f, pers_param.k, pers_param.b) 
    
    state.stage_log.append({
        'trial': state.t,
        'stage': s,
        'engagement': stage_engagement,
        'delta': delta,
        'V': V_s,
    })
    
    return delta, stage_engagement

def simulate(state, pers_param):
    tot_epi_engagement = 0
    for s in range(stage_amt):
        state.rpe[s], tot_stage_engagement = value_of_stage(state, s, pers_param)
        tot_epi_engagement += tot_stage_engagement
    state.episode_log.append({
        'trial': state.t,
        'total_engagement': tot_epi_engagement,
        'max_abs_rpe': max(abs(x) for x in state.rpe.values()),
        'est_f': sum(w * p['f'] for w, p in zip(state.weights, state.particles)),
        'est_k': sum(w * p['k'] for w, p in zip(state.weights, state.particles)),
        'est_b': sum(w * p['b'] for w, p in zip(state.weights, state.particles)),
        'est_skill': sum(w * p['skill'] for w, p in zip(state.weights, state.particles)),
    })
    if pers_param.skill < 1:
        pers_param.skill += .01
    return tot_epi_engagement

def train(state, pers_param):
    low_rpe_streak = 0
    while True:
        
        simulate(state, pers_param)

        max_rpe = max(abs(x) for x in state.rpe.values())

        if max_rpe < 0.05:
            low_rpe_streak += 1
            #print(f"streak at {low_rpe_streak}")
        else:
            low_rpe_streak = 0
            #print("streak reset")

        if low_rpe_streak >= 10:
            print(f"trained after {state.t} trials")
            print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
            print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
            return state.theta, state.t, state.weights

        if state.t == 2000:
            print(f"stopped after {state.t} trials")
            print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
            print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
            return state.theta, state.t, state.weights

        state.t += 1
        
def test_train():
    t = 1  
    N = 100
    particles = [makeparaguess(vars(fixed_params).keys()) for _ in range(N)]
    weights = np.ones(N) / N
    theta = np.zeros(len(param_values) + 1)

    state = ModelState(
        theta=theta,
        t=t,
        weights=weights,
        particles=particles
    )

    theta, t, weights = train(state, fixed_params)
    print("Estimated f:", sum(w * p['f'] for w, p in zip(weights, state.particles)))
    print("Estimated k:", sum(w * p['k'] for w, p in zip(weights, state.particles)))
    print("Estimated b:", sum(w * p['b'] for w, p in zip(weights, state.particles)))
    print("Estimated skill:", sum(w * p['skill'] for w, p in zip(weights, state.particles)))
    print("True params:", fixed_params)

def run_model_with_params(f_val, k_val, b_val, skill_val, max_trials=100):

    pers_param = UserParams(
        f=f_val,
        k=k_val,
        b=b_val,
        skill=skill_val
    )

    t = 1
    N = 100
    particles = [makeparaguess(vars(pers_param).keys()) for _ in range(N)]
    weights = np.ones(N) / N
    theta = np.zeros(len(param_values) + 1)

    state = ModelState(
        theta=theta,
        t=t,
        weights=weights,
        particles=particles
    )

    while state.t <= max_trials:
        simulate(state, pers_param)
        state.t += 1

    return [x['total_engagement'] for x in state.episode_log]

def plot_param_sweep(param_name, values, fixed_params, max_trials=100):
    plt.figure()

    for val in values:
        params = copy.copy(fixed_params)
        setattr(params, param_name, val)

        engagement_curve = run_model_with_params(
            f_val=params.f,
            k_val=params.k,
            b_val=params.b,
            skill_val=params.skill,
            max_trials=max_trials
        )

        plt.plot(range(1, max_trials + 1), engagement_curve, label=f"{param_name}={val}")

    plt.title(f"Effect of {param_name} on Engagement")
    plt.xlabel("Trial")
    plt.ylabel("Total Engagement")
    plt.legend()
    plt.show()

#plot_param_sweep('e', [0.1, 0.3, 0.5, 0.7, 0.9], fixed_params, 100)
test_train()