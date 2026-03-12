import copy
import numpy as np
import random
import math
import matplotlib.pyplot as plt

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

def randomize_user_params(): #debug
    f = random.gauss(.5, .5/3)
    k = random.gauss(.5, .5/3)
    e = random.gauss(.5, .5/3)
    b = random.gauss(.5, .5/3)
    return f, k, e, b

# user parameters (model does not see only estimates)
fixed_params = {
'f' : .5,  # Sensitivity to learning progress  
'k' : .5, #random.gauss(.5, .5/3) # Effort aversion  
"e" : .5, #random.gauss(.5, .5/3) # Variability in performance
"b" : .5, #random.gauss(.5, .5/3) # Boredom rate 
"skill" : .1 # initial preformance that will scale; will not be changed yet until ready for infererence
}

stage = {s: {"V" : 0.0,  ** copy.deepcopy(param_values)} for s in range(stage_amt)} 
rpe = {r : 0 for r in range(stage_amt)} 

def get_sigma(pers_param, base_sigma = .02, scaling_factor = .3):
    sigma = (base_sigma + diff * scaling_factor) / math.sqrt(pers_param['skill']) if pers_param['skill'] != 0 else (base_sigma + diff * scaling_factor)
    return sigma

def engagement_score(delta, v, t, f_val, k_val, e_val, b_val):
    signal = f_val * abs(delta)
    denom = max(0.05, abs(v))
    effort_cost = k_val * ((t * pacing) + diff * stage_amt) / denom
    boredom_cost = b_val * t * 0.01
    return signal - effort_cost - boredom_cost

def engage(delta, v, t, pers_param):
    cont = engagement_score(delta, v, t, pers_param['f'], pers_param['k'], pers_param['e'], pers_param['b'])
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
    return 1 / (1 + math.exp(-z))

def bayesian_particle_update(engaged, particles, weights, delta, v, t):
    new_weights = weights.copy()

    for i, p in enumerate(particles):
        f_g, k_g, b_g, e_g, skill_g = p['f'], p['k'], p['b'], p['e'], p['skill']

        cont = engagement_score(delta, v, t, f_g, k_g, e_g, b_g)
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

def value_of_stage(theta, s, t, weights, particles, pers_param, stage_log):
    V_s = V(theta, s)
    if s == stage_amt - 1:
        sigma = get_sigma(pers_param)
        reward_divergence = random.gauss(0, sigma)
        r = sigmoid(5 * (pers_param['skill'] - diff)) + reward_divergence
        r = max(0, min(1, r))
        #r = 1
        V_next = 0.0
    else:
        r, V_next = 0.0, V(theta, s+1)

    delta = r + g * V_next - V_s
    theta = theta + a * delta * phi(s)
    weights = bayesian_particle_update(engage(delta, V_s, t, pers_param), particles, weights, delta, V_s, t)
    stage_engagement = engagement_score(delta, V_s, t, pers_param['f'], pers_param['k'], pers_param['e'], pers_param['b']) 
    
    stage_log.append({
        'trial': t,
        'stage': s,
        'engagement': stage_engagement,
        'delta': delta,
        'V': V_s,
    })
    
    return theta, delta, weights, stage_engagement

def simulate(theta, t, weights, particles, pers_param, stage_log, episode_log):
    tot_epi_engagement = 0
    for s in range(stage_amt):
        theta, rpe[s], weights, tot_stage_engagement = value_of_stage(theta, s, t, weights, particles, pers_param, stage_log)
        tot_epi_engagement += tot_stage_engagement
    episode_log.append({
        'trial': t,
        'total_engagement': tot_epi_engagement,
        'max_abs_rpe': max(abs(x) for x in rpe.values()),
        'est_f': sum(w * p['f'] for w, p in zip(weights, particles)),
        'est_k': sum(w * p['k'] for w, p in zip(weights, particles)),
        'est_e': sum(w * p['e'] for w, p in zip(weights, particles)),
        'est_b': sum(w * p['b'] for w, p in zip(weights, particles)),
        'est_skill': sum(w * p['skill'] for w, p in zip(weights, particles)),
    })
    return theta, weights, tot_epi_engagement

def train(theta, t, weights, particles, pers_param, stage_log, episode_log):
    engagement = 0
    while True:
        
        theta, weights, new_engagement = simulate(theta, t, weights, particles, pers_param, stage_log, episode_log)
        engagement += new_engagement

        if max(abs(x) for x in rpe.values()) < 0.05:
            print(f"trained after {t} trials")
            print("V:", [round(V(theta, s), 3) for s in range(stage_amt)])
            print("RPE:", [round(rpe[s], 3) for s in range(stage_amt)])
            return theta, t, weights

        if t == 200:
            print(f"stopped after {t} trials")
            print("V:", [round(V(theta, s), 3) for s in range(stage_amt)])
            print("RPE:", [round(rpe[s], 3) for s in range(stage_amt)])
            return theta, t, weights

        t += 1

def run_model_with_params(f_val, k_val, e_val, b_val, skill_val, max_trials=100):

    pers_param = {
        'f': f_val,
        'k': k_val,
        'e': e_val,
        'b': b_val,
        'skill': skill_val
    }

    episode_log = []
    stage_log = []

    t = 1
    N = 100
    particles = [makeparaguess(pers_param.keys()) for _ in range(N)]
    weights = np.ones(N) / N
    theta = np.zeros(len(param_values) + 1)

    while t <= max_trials:
        theta, weights, _ = simulate(theta, t, weights, particles, pers_param, stage_log, episode_log)
        t += 1

    return [x['total_engagement'] for x in episode_log]

def plot_param_sweep(param_name, values, fixed_params, max_trials=10):
    plt.figure()

    for val in values:
        params = fixed_params.copy()
        params[param_name] = val

        engagement_curve = run_model_with_params(
            f_val=params['f'],
            k_val=params['k'],
            e_val=params['e'],
            b_val=params['b'],
            skill_val=params['skill'],
            max_trials=max_trials
        )

        plt.plot(range(1, max_trials + 1), engagement_curve, label=f"{param_name}={val}")

    plt.title(f"Effect of {param_name} on Engagement")
    plt.xlabel("Trial")
    plt.ylabel("Total Engagement")
    plt.legend()
    plt.show()

def test_train():
    episode_log = []
    stage_log = []
    
    t = 1  
    N = 100
    particles = [makeparaguess(fixed_params) for _ in range(N)]
    weights = np.ones(N) / N
    theta = np.zeros(len(param_values) + 1)
    theta, t, weights = train(theta, t, weights, particles, fixed_params, stage_log, episode_log)

#plot_param_sweep('f', [0.1, 0.3, 0.5, 0.7, 0.9], fixed_params)
test_train()