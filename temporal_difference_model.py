import copy
import numpy as np
import random
import math

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
f = .5  # Sensitivity to learning progress  
k = .5 #random.gauss(.5, .5/3) # Effort aversion  
e = .5 #random.gauss(.5, .5/3) # Variability in performance
b = .5 #random.gauss(.5, .5/3) # Boredom rate lower is less boredom
skill = .1 # initial preformance that will scale; will not be changed yet until ready for infererence
pers_param = ['f', 'k', 'e', 'b', 'skill']



stage = {s: {"V" : 0.0,  ** copy.deepcopy(param_values)} for s in range(stage_amt)} 
rpe = {r : 0 for r in range(stage_amt)} 

def get_sigma(base_sigma = .02, scaling_factor = .3):
    sigma = (base_sigma + diff * scaling_factor) / math.sqrt(skill) if skill != 0 else (base_sigma + diff * scaling_factor)
    return sigma

def engagement_score(delta, v, t, f_val, k_val, e_val, b_val):
    signal = f_val * abs(delta)
    denom = max(0.05, abs(v) + e_val)
    effort_cost = k_val * ((t * pacing) + diff * stage_amt) / denom
    boredom_term = 0.01 / max(0.05, b_val)
    return signal - effort_cost + boredom_term

def engage(delta, v, t):
    cont = engagement_score(delta, v, t, f, k, get_sigma(), b) 
    if cont < 0: 
        if random.uniform(-1, g - abs(delta)) < 0:
            pass #will quit when ready for inferance
    return 0 if cont < 0 else 1

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

def value_of_stage(theta, s, t, weights, frus):
    V_s = V(theta, s)
    if s == stage_amt - 1:
        sigma = get_sigma()
        reward_divergence = random.gauss(0, sigma)
        r = sigmoid(5 * (skill - diff)) + reward_divergence
        r = max(0, min(1, r))
        #r = 1
        V_next = 0.0
    else:
        r, V_next = 0.0, V(theta, s+1)

    delta = r + g * V_next - V_s
    theta = theta + a * delta * phi(s)
    weights = bayesian_particle_update(engage(delta, V_s, t), particles, weights, delta, V_s, t)
    stage_engagement = engagement_score(delta, V_s, t, f, k, e, b) 
    return theta, delta, V_s, weights, stage_engagement

def simulate(theta, t, weights, frus):
    tot_epi_engagement = 0
    for s in range(stage_amt):
        theta, rpe[s], stage[s]["V"], weights, tot_stage_engagement = value_of_stage(theta, s, t, weights, frus)
        tot_epi_engagement += tot_stage_engagement
    return theta, weights, tot_epi_engagement

def train(theta, t, weights):
    engagement = 0
    while True:
        theta, weights, new_engagement = simulate(theta, t, weights, engagement)
        engagement += new_engagement
        
        #if random.gauss()
        if max(abs(x) for x in rpe.values()) < 0.05:
            print(f"trained after {t} trials")
            print("V:", [round(stage[s]["V"], 3) for s in range(stage_amt)])
            print("RPE:", [round(rpe[s], 3) for s in range(stage_amt)])
            return theta, t, weights

        if t == 200:
            print(f"stopped after {t} trials")
            print("V:", [round(stage[s]["V"], 3) for s in range(stage_amt)])
            print("RPE:", [round(rpe[s], 3) for s in range(stage_amt)])
            return theta, t, weights

        t += 1


        
t = 1  
N = 100
particles = [makeparaguess(pers_param) for _ in range(N)]
weights = np.ones(N) / N
theta = np.zeros(len(param_values) + 1)
theta, t, weights = train(theta, t, weights)
print(f"actual params: f: {f}, k: {k}, e: {e}, b: {b}, skill: {skill}")