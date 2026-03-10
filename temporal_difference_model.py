import copy
import numpy as np
import random
import math
from logitical_regression_model import train_pparams


# goals: create a model that adaptively selects task difficulty for a specific individual by modeling and regulating reward prediction error (RPE)
# 1. the learner practices tasks
# 2. it estimates how much RPE (dopamine released basically) a person generates
# 3. it adjusts difficulty parameters to maintain engagement
# 4. it personalizes the learning curve
# secondary observations (Not important now but i think is interesting): discount factor (gamma) higher values equates to future rewards mattering more. 
# Model individual differences in temporal discount and reward sensitivity commonly assosiated with neurodivergents.
# Possible integration with model parameters to simulate different neurological conditions

# Discount Factor: High values means future rewards matter more.
g = .9 

# Learning rate: How quickly the model minimizes loss.
a =.01

# Reward: binary 0 or 1
r = 1

# parameter features (model tweaks):
stage_amt = 4 # How many stages there are until reward
diff = .2 # The difficulty of each stage
pacing = 0.1 # Speed of difficulty escalation 


param_values = {
    'bias' : 1,   
    'd' : diff,
} 

# user parameters (model does not see only estimates)
f = .5 #random.gauss(.5, .5/3) # Sensitivity learning progress  
k = .5 #random.gauss(.5, .5/3) # Effort aversion  
e = .5 #random.gauss(.5, .5/3) # Variability in performance
b = .5 #random.gauss(.5, .5/3) # Boredom rate higher is less boredom
skill = .1 # initial preformance at an activity for a controlled difficulty
pers_param = ['f', 'k', 'e', 'b', 'skill']

base_sigma = .02 
scaling_factor = .3



stage = {s: {"V" : 0.0,  ** copy.deepcopy(param_values)} for s in range(stage_amt)} 
rpe = {r : 0 for r in range(stage_amt)} # RPE for each stage. Pos = outcome better than expected, Neg = outcome worse, approx 0: fully predicted no learning

# Engagement factor:
def engage(delta, v, t):
    e = random.gauss(0, .05)
    if t == 0:
        cont = (f) - (((t * 0.01) + (stage_amt * -0.05)/(v + e)) * k) + 0.01/b
    else:
        cont = (f * delta) - (((t * 0.01)*(stage_amt * -0.05)/(v + e)) * k) + 0.01/b
    
    if t >= 0:
        pass
        # print(f"cont factor: {cont} with values delta: {delta} and value: {v}")
        # print(f"equation is ({round(f,3)} * {round(delta,3)}) - ((({round(t,3)}/100)*({stage_amt}/100)/{round(v,3)}) * {round(k,3)})")
    if cont < 0:
        donechance = random.uniform(-1, g-delta)
        if donechance < 0:
            # print(f"bored after {t} trials")
            # print("V:", [round(stage[s]["V"],3) for s in range(stage_amt)])
            # print("RPE:", [round(rpe[s],3) for s in range(stage_amt)])
            # print(trained_params[-1])
            # print(f"actual params: f: {f}, k: {k}, e: {e}, b: {b}, skill: {skill}")
            # quit()
            pass
    
    return 0 if cont < 0 else 1 

def makeparaguess(paramlist):
    
    paramvalues = {}
    paramvalues["Bias"] = random.gauss(.5, .5/3)
    for param in paramlist: 
        paramvalues[param] = random.gauss(.5, .5/3)
    return paramvalues

def phi(s: int):
    d = stage[s]['d']
    s_norm = s / (stage_amt - 1)
    return np.array([1.0, d, s_norm])

def V(theta, s): 

    v = float((theta @ phi(s)))
    return v

def value_of_stage(theta, s, t, r = r, g = g, a = a):
    V_s = V(theta, s)
    if s == stage_amt - 1:
        sigma = (base_sigma + diff * scaling_factor) / math.sqrt(skill) if skill != 0 else (base_sigma + diff * scaling_factor)
        e = random.gauss(0, sigma)
        r = 1 / (1 + math.e**-(5*(diff - skill))) + e
        r = max(0, min(1, r))
        #r = 1
        V_next = 0.0
    else:
        r, V_next = 0.0, V(theta, s+1)

    delta = r + g * V_next - V_s
    theta = theta + a * delta * phi(s)
    engage(delta, s, t)
    return theta, delta, V_s

def simulate(theta, t):
    for s in range(stage_amt):
        theta, rpe[s], stage[s]["V"] = value_of_stage(theta, s, t)
    return theta

def train(theta, t):
    trained = False
    while not trained:
        e = random.gauss(0, .05)
        theta = simulate(theta, t)
        m_reward = sum(round(rpe[s],3) for s in range(stage_amt)) / stage_amt
        if 0 < m_reward < e:
            print(f"trained after {t} trials")
            print("V:", [round(stage[s]["V"],3) for s in range(stage_amt)])
            print("RPE:", [round(rpe[s],3) for s in range(stage_amt)])
            trained = True
        if t == 3:
            print(f"trained after {t} trials")
            print("V:", [round(stage[s]["V"],3) for s in range(stage_amt)])
            print("RPE:", [round(rpe[s],3) for s in range(stage_amt)])
            return t
        print(t)
        trained_params[t] = (train_pparams(True, t, trained_params))
        print(trained_params[t])
        break
        t += 1

        
t = 1  
trained_params = {}
trained_params[0] = makeparaguess(pers_param) 
print(f"original: {trained_params[0]}")
theta = np.zeros(len(param_values) + 1)
t = train(theta, t)
#print(trained_params[t])
print(f"actual params: f: {f}, k: {k}, e: {e}, b: {b}, skill: {skill}")



