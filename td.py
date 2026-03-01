import random
import copy
import numpy as np

# goals: create a model that adaptively selects task difficulty for a specific individual by modeling and regulating reward prediction error (RPE)
# 1. the learner practices tasks
# 2. it estimates how much RPE (dopamine released basically) a person generates
# 3. it adjusts difficulty parameters to maintain engagement
# 4. it personalizes the learning curve
# secondary observations (Not important now but i think is interesting): discount factor (gamma) higher values equates to future rewards mattering more. 
# Model individual differences in temporal discount and reward sensitivity commonly assosiated with neurodivergents.
# Possible integration with model parameters to simulate different neurological conditions



# user parameters (model does not see only estimates)
pers_param = {
# Discount Factor: High values means future rewards matter more. High value is standard but possible tweaks
'g' : .9, 

# Learning rate: How quickly the model minimizes loss. 
# Higher values are quicker but overbiases parameters, 
# lower values take longer but are more accurate
'a' : .05,

# Reward sensitivity: How much reward based on difficulty
'r' : 1 

}


# parameter features (model tweaks):
stage_amt = 4 # How many stages there are until reward
diff = 0.1 # The difficulty of each stage


param_values = {
    'bias' : 1,
    'd' : diff
    } 

stage = {s: {"V" : 0.0,  **copy.deepcopy(param_values)} for s in range(stage_amt)} 
rpe = {r : 0 for r in range(stage_amt)} # RPE for each stage. Pos = outcome better than expected, Neg = outcome worse, approx 0: fully predicted no learning
#param_values['stage_index'] = 0 




# Engagement factor:

def phi(s: int):
    d = stage[s]['d']
    s_norm = s / (stage_amt - 1)
    return np.array([1.0, d, s_norm])

def V(theta, s): 
    return float(theta @ phi(s))

def value_of_stage(theta, s):
    V_s = V(theta, s)
    if s == stage_amt - 1:
        r, V_next = pers_param['r'], 0.0
    else:
        r, V_next = 0.0, V(theta, s+1)

    delta = r + pers_param['g'] * V_next - V_s
    theta = theta + pers_param['a'] * delta * phi(s)
    return theta, delta, V_s

def train(epoch = 1):
    theta = np.zeros(len(param_values) + 1)

    for _ in range(epoch):
        for s in range(stage_amt):
            theta, rpe[s], stage[s]["V"] = value_of_stage(theta, s)
train(50)
print("V:", [round(stage[s]["V"],3) for s in range(stage_amt)])
print("RPE:", [round(rpe[s],3) for s in range(stage_amt)])
    