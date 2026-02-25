import random
import math


#if skill is too much higher than difficulty no novelty and if skill too much lower no reward
#skill increases through iterations
stage_amt = 4
difficulty = float(1)
skill = float(2)
#if large reward initially that drops off
disc_fac = .9
a = .9
#reward how total enjoyment after iteration
r = 1
noise_level = 0.5

def value_of_stage(difficulty, skill, epochs = 1):
    stage = {s: {"V": 0.0, "D": 0.0} for s in range(stage_amt)}
    for _ in range(epochs):
        for s in range(stage_amt): 
            oldV = {s: stage[s]["V"]} 
            epsilon = random.uniform(-noise_level, noise_level)
            
            obs = 1 / math.e**-(skill - difficulty) + epsilon
            obs = max(0, min(1, obs))

            stage[s]["V"] = oldV[s] + a * (obs - oldV[s])
            stage[s]["D"] = round((obs - oldV[s]), 4)
            stage[s]["V"] = round(stage[s]["V"], 4)
            
    return stage

def prediction_error(stage_vals):
    surprise = [
    stage_vals[s]["R"] + disc_fac * stage_vals[s+1]["V"] - stage_vals[s]["V"]
    if s < len(stage_vals) - 1
    else stage_vals[s]["R"] - stage_vals[s]["V"]
    for s in range(len(stage_vals))
]
    return surprise
print(value_of_stage(4, 1, 3))

    