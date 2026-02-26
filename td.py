import random
import math

#if skill is too much higher than difficulty no novelty and if skill too much lower no reward
#skill increases through iterations
stage_amt = 4
disc_fac = .9
a = .9


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

        stage["Expected Overall"] = sum(stage[s]["V"] for s in range(stage_amt))/stage_amt
        stage["Delta Overall"] = sum(stage[s]["D"] for s in range(stage_amt))/stage_amt
    
        stage["Trained Amount"] += 1

    return stage

def bored(stage_vals):
    stored_drive = 0
    stage_copy = stage_dict
    diff = 0
    skill_learning = 0
    count = 1
    i = 0
    while count < 5:
        scores = {i: {"Expected Overall": 0.0, "Delta Overall": 0.0}}
        result  = stage_vals(diff, i, stage_copy)
        scores[i]["Expected Overall"] = result["Expected Overall"]
        scores[i]["Delta Overall"] = result["Delta Overall"]

        if scores[i]["Expected Overall"] >= .9:
            print(f"Trained {stage_copy["Trained Amount"]} time(s) skill level of {skill_learning}")
            final_test_enjoyment = stage_vals(diff, i, stage_copy)["Delta Overall"]
            if -0.01 < final_test_enjoyment:
                diff += 1 + final_test_enjoyment
                print(f"I'm pretty confident let me try something harder. Diff is now {diff}")
                if final_test_enjoyment > 0.01:
                    stored_drive += final_test_enjoyment
            elif final_test_enjoyment < -0.01:
                print(f"Guess im not good enough. ")
            
            count += 1
        skill_learning += 1 + scores[i]["Delta Overall"]
        i += 1
        
    return scores
    
        

bored(value_of_stage)
#print(value_of_stage(0, 2, stage_dict, 100))
    