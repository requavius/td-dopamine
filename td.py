import random
import math
import itertools

#if skill is too much higher than difficulty no novelty and if skill too much lower no reward
#skill increases through iterations
stage_amt = 4
#if large reward initially that drops off
disc_fac = .9
a = .9
#reward how total enjoyment after iteration
r = 1


stage_dict = {s: {"V": 0.0, "D": 0.0} for s in range(stage_amt)} 
stage_dict["Trained Amount"] = 0

def value_of_stage(difficulty, skill, stage, epochs = 1, base_sigma = .02, scaling_factor = .3):

    for _ in range(epochs):
        for s in range(stage_amt): 
            oldV = {s: stage[s]["V"]} 
            sigma = (base_sigma + difficulty * scaling_factor) / math.sqrt(skill) if skill != 0 else (base_sigma + difficulty * scaling_factor)
            epsilon = random.gauss(0, sigma)
            
            obs = 1 / (1 + math.e**-(skill - difficulty) ) + epsilon
            obs = max(0, min(1, obs))

            stage[s]["V"] = oldV[s] + a * (obs - oldV[s])
            stage[s]["D"] = obs - oldV[s]
            stage[s]["V"] = stage[s]["V"]
        stage["Expected Overall"] = sum(stage[s]["V"] for s in range(stage_amt))/stage_amt
        stage["Delta Overall"] = sum(stage[s]["D"] for s in range(stage_amt))/stage_amt
    
        stage["Trained Amount"] += 1
    # for s in range(stage_amt):
    #     stage[s]["V"] = round(stage[s]["V"], 2)
    #     stage[s]["D"] = round(stage[s]["D"], 2)
    # stage["Expected Overall"], stage["Delta Overall"] = round(stage["Expected Overall"], 2), round(stage["Delta Overall"], 2)
    return stage

def bored(stage_vals):
    stage_copy = stage_dict
    diff = 4 / 9 - 1/3
    i = 0
    count = 1
    while True:
        scores = {i: {"Overall Exp": 0.0, "Overall Delta": 0.0}}
        skill = 2 / (1 + math.e**-(i) ) - 1
        result  = stage_vals(diff, skill, stage_copy)
        scores[i]["Overall Exp"] = result["Expected Overall"]
        scores[i]["Overall Delta"] = result["Delta Overall"]
        if scores[i]["Overall Exp"] >= .9:
            print(stage_copy["Trained Amount"])
            break
        i += 1 + scores[i]["Overall Delta"]
        count += 1
    return scores
    
        

print(bored(value_of_stage))
#print(value_of_stage(0, 2, stage_dict, 100))
    