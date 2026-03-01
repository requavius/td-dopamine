import random
import copy

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
'a' : .9,

# Reward sensitivity: How much reward based on difficulty
'r' : 1 

}


# parameter features (model tweaks):
stage_amt = 4 # How many stages there are until reward
diff = 0.1 # The difficulty of each stage

param_values = {
    'bias' :  {'x' : 1, 't' : 0.0},
    'd' : {'x' : diff, 't' : 0.0}
    } 

stage = {s: {"V" : 0.0, "stage_index" : {'x' : s, 't' : 0.0} , **copy.deepcopy(param_values)} for s in range(stage_amt)} 
rpe = {r : 0 for r in range(stage_amt)} # RPE for each stage. Pos = outcome better than expected, Neg = outcome worse, approx 0: fully predicted no learning
param_values['stage_index'] = 0

# Engagement factor:


def value_of_stage():
    oldV = copy.deepcopy(stage)
    newV = {}
    for s in range(stage_amt): # for each stage calculate value and update

        if s == stage_amt - 1:
            target = pers_param["r"]
        else:
            target = pers_param["g"] * oldV[s + 1]["V"]
        
        rpe[s] = target - oldV[s]["V"] # calculate rpe
        newV[s] = sum(oldV[s][p]['x'] * oldV[s][p]["t"] for p in param_values) # update value
        stage[s]["V"] = newV[s]

        # update parameter

        for p in param_values: stage[s][p]["t"] = stage[s][p]['t'] + pers_param["a"] * rpe[s] * stage[s][p]['x']

print(f"Original stages: {stage}, \nOriginal rpe{rpe}")
value_of_stage()
print(f"New stages: {stage}, \nNew rpe{rpe}")
    