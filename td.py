


#if skill is too much higher than difficulty no novelty and if skill too much lower no reward
#skill increases through iterations
stage_amt = 4
difficulty = 2
skill = 2
#if large reward initially that drops off
disc_fac = .9
a = 1
#reward how total enjoyment after iteration
r = 1

def value_of_stage():
    stage = {s: {"V": 0.0, "R": 0.0} for s in range(stage_amt)}

    for _ in range(stage_amt): 
        oldV = {s: stage[s]["V"] for s in stage} 

        for s in range(stage_amt):

            if s == stage_amt - 1:
                target = r
            else:
                target = disc_fac * oldV[s + 1]

            stage[s]["V"] = oldV[s] + a * (target - oldV[s])

    return stage

print(value_of_stage())

    