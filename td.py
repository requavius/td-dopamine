


#if large reward initially that drops off
stage_amt = 5
difficulty = 2
skill = 2
disc_fac = .9
learning_rate = .5
#if skill is too much higher than difficulty no novelty and if skill too much lower no reward
#skill increases through iterations
def loop():
    stage_values = []
    for i in range(stage_amt):
        stage_values.append(0)
    print(stage_values)
    
loop()

    