import random
import math

#goals: create a model that adaptively selects task difficulty for a specific individual by modeling and regulating reward prediction error (RPE)
#1. the learner practices tasks
#2. it estimates how much RPE (dopamine basically) a person generates
#3. it adjusts difficulty parameters to maintain engagement
#4. it personalizes the learning curve
# secondary observations (Not important now but i think is interesting): discount factor (gamma) higher values equates to future rewards mattering more. 
# Model individual differences in temporal discount and reward sensitivity commonly assosiated with neurodivergents.
# Possible integration with model parameters to simulate different neurological conditions

stage_amt = 4
#Discount Factor: High values means future rewards matters more. High value is standard but possible tweaks
gamma = .9

#Learning rate: How quickly the model minimizes loss. Higher values are quicker but overbiases parameters, lower values take longer but are more accurate
a = .1
#Reward: can be binary but will probably be based on difficulty
r=1

def value_of_stage():

    stage = {s: {"V": 0.0} for s in range(stage_amt)}
    rpe = {r: 0 for r in range(stage_amt)} #RPE for each stage. Pos = outcome better than expected, Neg = outcome worse, approx 0: fully predicted no learning


    oldV = {s: stage[s]["V"] for s in stage} 

    for s in range(stage_amt): #for each stage calculate value and update

        if s == stage_amt - 1:
            target = r
        else:
            target = gamma * oldV[s + 1]

        rpe[s] = target - oldV[s] #calculate rpe
        stage[s]["V"] = oldV[s] + a * (target - oldV[s]) #update value
            

    return stage

print(value_of_stage())
    