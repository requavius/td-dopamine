import copy
import numpy as np
import random
import math

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
diff = 0.1 # The difficulty of each stage
pacing = 0.1 # Speed of difficulty escalation 

param_values = {
    'bias' : 1,   
    'd' : diff,
} 

# user parameters (model does not see only estimates)
f = random.gauss(.5, .5/3) # Sensitivity learning progress  
k = random.gauss(.5, .5/3) # Effort aversion  
e = random.gauss(.5, .5/3) # Variability in performance
b = random.gauss(.5, .5/3) # Boredom rate higher is less boredom
skill = .1 # initial preformance at an activity for a controlled difficulty
pers_param = ['f', 'k', 'e', 'b', 'skill']

base_sigma = .02 
scaling_factor = .3


stage = {s: {"V" : 0.0,  **copy.deepcopy(param_values)} for s in range(stage_amt)} 
rpe = {r : 0 for r in range(stage_amt)} # RPE for each stage. Pos = outcome better than expected, Neg = outcome worse, approx 0: fully predicted no learning

# Engagement factor:
def engage(delta, t, v, s):
    if t == 0:
        cont = (f * abs(delta)) - (((t/10)*(stage_amt/100)/v) * k)
    else:
        cont = f - (((t/10)*(stage_amt/100)/v) * k)
    
    if t >= 0:
        pass
        # print(f"cont factor: {cont} with values delta: {delta} and value: {v}")
        # print(f"equation is ({round(f,3)} * {round(delta,3)}) - ((({round(t,3)}/100)*({stage_amt}/100)/{round(v,3)}) * {round(k,3)})")
    if cont < 0:
        # print(f"bored after {t} trials")
        # print("V:", [round(stage[s]["V"],3) for s in range(stage_amt)])
        # print("RPE:", [round(rpe[s],3) for s in range(stage_amt)])
        # print(trained_params[-1])
        # print(f"actual params: f: {f}, k: {k}, e: {e}, b: {b}, skill: {skill}")
        pass
        #quit()
    
    return 0 if cont < 0 else 1 

def makeparaguess(paramlist):
    
    paramvalues = {}
    paramvalues["Bias"] = random.gauss(.5, .5/3)
    for param in paramlist: 
        paramvalues[param] = random.gauss(.5, .5/3)
    return paramvalues

def calculateconfidence(randomparams, t):
    newparams = randomparams.copy()
    for p in newparams:
        if p == 'Bias': continue
        if trained_params != []:
            newparams[p] *= trained_params[t][p]
    z = sum(newparams.values()) 
    
    guess = 1/(1+math.e**(-z)) 
    return guess

def calculateloss(randomparams, cont):
        firstlossvalue = []

        for t in range(trained_params):
            guess = calculateconfidence(randomparams, t)
            engagedornot = (cont)
            firstlossvalue.append(-((engagedornot * math.log(guess)) + (1-engagedornot) * math.log(1-guess)))

        lossvalues = ((sum(firstlossvalue)) / t)
        return lossvalues

def freezeparameter(randomparams, cont, t):
    
    gradients = {}
    for param in randomparams:
        z = 0
        if param != 'Bias':
            for i in range(len(trained_params)):
                engagedornot = (cont)
                z += (calculateconfidence(randomparams, i) - engagedornot)*trained_params[i][param]
        else:
            for i in range(len(trained_params)):
                engagedornot = (cont)
                z += (calculateconfidence(randomparams, i) - engagedornot)
        gradients[param] = z / t
        
    return gradients

def train_pparams(params, cont, t, learningrate = a):
    derivs = freezeparameter(params, cont, t)
    for k in params:
        params[k] -= learningrate * derivs[k]
    return params

def phi(s: int):
    d = stage[s]['d']
    s_norm = s / (stage_amt - 1)
    return np.array([1.0, d, s_norm])

def V(theta, s): 
    sigma = (base_sigma + diff * scaling_factor) / math.sqrt(skill) if skill != 0 else (base_sigma + diff * scaling_factor)
    e = random.gauss(0, sigma)
    v = float((theta @ phi(s)))
    return v + e

def value_of_stage(theta, s, t, r = r, g = g, a = a):
    V_s = V(theta, s)
    if s == stage_amt - 1:
        r, V_next = r, 0.0
    else:
        r, V_next = 0.0, V(theta, s+1)

    delta = r + g * V_next - V_s
    theta = theta + a * delta * phi(s)
    if t != 0:
        trained_params.append(train_pparams(makeparaguess(pers_param), engage(delta, t, V_s, s), t))
    return theta, delta, V_s

def simulate(theta, t):
    for s in range(stage_amt):
        theta, rpe[s], stage[s]["V"] = value_of_stage(theta, s, t)
    return theta

def train(theta):
    trained = False
    t = 0
    while not trained:
        theta = simulate(theta, t)
        if round(stage[stage_amt-1]["V"], 3) == 1:
            print(f"trained after {t} trials")
            print("V:", [round(stage[s]["V"],3) for s in range(stage_amt)])
            print("RPE:", [round(rpe[s],3) for s in range(stage_amt)])
            trained = True
        if t == 500:
            print(f"trained after {t} trials")
            print("V:", [round(stage[s]["V"],3) for s in range(stage_amt)])
            print("RPE:", [round(rpe[s],3) for s in range(stage_amt)])
            trained = True
        t += 1
trained_params = []   
theta = np.zeros(len(param_values) + 1)
train(theta)
print(trained_params[-1])
print(f"actual params: f: {f}, k: {k}, e: {e}, b: {b}, skill: {skill}")



