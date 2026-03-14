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

@dataclass
class ModelState:
    theta: np.ndarray
    t: int
    weights: np.ndarray
    skill: float  # initial preformance that will scale
    particles: list
    highest_eng: float
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
    denom = max(f_val, abs(v) + state.skill) + 1
    effort_cost = k_val * ((state.t_since_eng) + diff * stage_amt) / denom 
    boredom_cost = b_val * state.t_since_eng
    
    score = signal - effort_cost - boredom_cost
    if score < 0 and not part:
        pass
        #print(f"trial: {state.t} signal {signal} - effort {effort_cost} - boredom {boredom_cost}")
    if score >= state.highest_eng and not part:
        state.t_since_eng = 0
        state.highest_eng = score
    

    return score

def engage(state: ModelState, formula):
    cont = formula
    prob = sigmoid(cont)
    decision = 1 if random.random() < prob else 0
    if not decision: state.highest_eng = 0
    return decision

def makeparaguess(paramlist, other = None):
    paramvalues = {}
    for param in paramlist:
        paramvalues[param] = random.uniform(0.05, 1.0)
    if other:
        paramvalues[other] = random.uniform(0.05, 1.0)
    return paramvalues

def phi(s: int):
    d = diff
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
    
    learning_gain = abs(delta) * (1.0 - state.skill) + (0.001 * stage_amt) # skill grows with practice but saturates. This might be changed based on what makes sense for skill improvment
    state.skill += learning_gain / stage_amt
    return delta, stage_engagement, engaged_observation

def simulate(state: ModelState, pers_param):
    tot_epi_engagement = 0
    stages_completed = 0
    for s in range(stage_amt):
        state.rpe[s], tot_stage_engagement, engaged = value_of_stage(state, s, pers_param)
        tot_epi_engagement += tot_stage_engagement
        stages_completed += 1
        if not engaged and state.t > 1:
            #print(f"quit after {stages_completed} stages with rpe of {state.rpe[s]}")
            state.t_since_eng = 0
            break
        else:
            pass
            #print(f'kept engagement for {s} stages with rpe of {state.rpe[s]}')
    state.episode_log.append({
        'trial': state.t,
        'total_engagement': tot_epi_engagement, # will be re engagment factor
        'Stages completed': stages_completed, # how many stages were completed before disengagement
        'max_abs_rpe': max(abs(x) for x in state.rpe.values()),
        'est_f': sum(w * p['f'] for w, p in zip(state.weights, state.particles)),
        'est_k': sum(w * p['k'] for w, p in zip(state.weights, state.particles)),
        'est_b': sum(w * p['b'] for w, p in zip(state.weights, state.particles)),
    })
    
    state.t_since_eng += 1 
    return engaged

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
        
        if low_rpe_streak >= 10 and average_v > 0.1:
            if debug:
                print(f"trained after {state.t} trials")
                print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
                print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
            return 

        if state.t >= 2000:
            if debug:
                print(f"stopped after {state.t} trials")
                print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
                print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
            return 
    
        state.t += 1
        
        
def test_train(true_f = random.uniform(0.05, 1.0), true_k = random.uniform(0.05, 1.0), true_b = random.uniform(0.05, 1.0), debug = False):
    
    fixed = UserParams(f=true_f, k=true_k, b=true_b)
    particles = [makeparaguess(vars(fixed).keys()) for _ in range(100)]
    weights = np.ones(100) / 100
    theta = np.zeros(len(param_values) + 1)

    state = ModelState(
        theta = theta,
        t = 1,
        weights = weights,
        particles = particles,
        skill = .1,
        highest_eng = 0,
        t_since_eng = 0,
    )

    train(state, fixed, debug)
    
    avg_stages = sum(ep['Stages completed'] for ep in state.episode_log) / len(state.episode_log)
    if debug == True:
        
        print("Estimated f:", sum(w * p['f'] for w, p in zip(state.weights, state.particles)))
        print("Estimated k:", sum(w * p['k'] for w, p in zip(state.weights, state.particles)))
        print("Estimated b:", sum(w * p['b'] for w, p in zip(state.weights, state.particles)))
        print(f"Average stages completed per episode: {avg_stages:.2f}")
        print("True params:", fixed)
    
    return {'true_f': true_f, 'true_k': true_k, 'true_b': true_b, 'avg_stages': avg_stages}

def collect_results(n=60, repeats=5):
    results = []
    sweep = np.linspace(0.05, 0.95, n)
    fixed = 0.1

    for i, val in enumerate(sweep):
        f_stages = np.mean([test_train(val, fixed, fixed)['avg_stages'] for _ in range(repeats)])
        k_stages = np.mean([test_train(fixed, val, fixed)['avg_stages'] for _ in range(repeats)])
        b_stages = np.mean([test_train(fixed, fixed, val)['avg_stages'] for _ in range(repeats)])
        results.append({'true_f': val, 'true_k': fixed, 'true_b': fixed, 'avg_stages': f_stages})
        results.append({'true_f': fixed, 'true_k': val, 'true_b': fixed, 'avg_stages': k_stages})
        results.append({'true_f': fixed, 'true_k': fixed, 'true_b': val, 'avg_stages': b_stages})
        print(f"completed {i+1}/{n}")

    return results

def plot_results(results):
    f_vals     = [r['true_f']     for r in results]
    k_vals     = [r['true_k']     for r in results]
    b_vals     = [r['true_b']     for r in results]
    avg_stages = [r['avg_stages'] for r in results]

    _, ax = plt.subplots(figsize=(8, 6))

    k_sorted = sorted(zip(k_vals, avg_stages))
    f_sorted = sorted(zip(f_vals, avg_stages))
    b_sorted = sorted(zip(b_vals, avg_stages))

    ax.plot(*zip(*k_sorted), color='#FF5722', label='k (effort aversion)')
    ax.plot(*zip(*f_sorted), color='#2196F3', label='f (progress sensitivity)')
    ax.plot(*zip(*b_sorted), color='#4CAF50', label='b (boredom rate)')

    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Average stages completed')
    ax.set_title('Parameter vs Engagement')
    ax.legend()

    plt.tight_layout()

    plt.savefig('engagement_by_params.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_results(collect_results(60,1))
#test_train(true_f=0.9, true_k=.1, true_b=.1, debug=True)
