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

a = .9
#Reward: can be binary but will probably be based on difficulty
r=1

def value_of_stage():

    stage = {s: {"V": 0.0} for s in range(stage_amt)}
    rpe = {r: 0 for r in range(stage_amt)} #RPE for each stage. Pos = outcome better than expected, Neg = outcome worse, approx 0: fully predicted no learning

    for _ in range(stage_amt): #1st run last stage has rpe of r and subsequent subsequent interations fill previous stages until fully trained at stage_amt
        oldV = {s: stage[s]["V"] for s in stage} 

        for s in range(stage_amt): #for each stage calculate value and update

            if s == stage_amt - 1:
                target = r
            else:
                target = gamma * oldV[s + 1]

            rpe[s] = target - oldV[s]
            stage[s]["V"] = oldV[s] + a * (target - oldV[s])
            

    return stage

# def bored(stage_vals):
#     stored_drive = 0
#     stage_copy = stage_dict.copy()
#     diff = 0
#     skill_learning = 0
#     count = 1
#     i = 0
#     while count < 5:
#         scores = {i: {"Expected Overall": 0.0, "Delta Overall": 0.0}}
#         result  = stage_vals(diff, skill_learning, stage_copy)
#         scores[i]["Expected Overall"] = result["Expected Overall"]
#         scores[i]["Delta Overall"] = result["Delta Overall"]

#         if scores[i]["Expected Overall"] >= .9:
#             print(f"Trained {stage_copy["Trained Amount"]} time(s) skill level of {skill_learning}")
#             final_test_enjoyment = stage_vals(diff, skill_learning, stage_copy)["Delta Overall"]
#             if -0.01 < final_test_enjoyment:
#                 diff += 1 + final_test_enjoyment
#                 print(f"I'm pretty confident let me try something harder. Diff is now {diff}")
#                 if final_test_enjoyment > 0.01:
#                     stored_drive += final_test_enjoyment
#             elif final_test_enjoyment < -0.01:
#                 print(f"Guess im not good enough.")
            
#             count += 1
#         skill_learning += 1 + scores[i]["Delta Overall"]
#         i += 1
        
#     return scores
    
        

# bored(value_of_stage)
print(value_of_stage())
    