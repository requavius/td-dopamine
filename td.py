


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

def loop():
    stage_values = []
    for i in range(stage_amt):
        stage_values.append(0)
    for i in range(stage_amt):
        if i == 1:
            for pos, i in enumerate(stage_values):
                if pos <= stage_amt-2:
                    stage_values[pos] = i + a*(0 + (disc_fac*stage_values[pos+1]) - i)
                else:
                    stage_values[pos] = i + a*(1 - stage_values[pos])
        else:
            for pos, i in enumerate(stage_values):
                if pos <= stage_amt-2:
                    stage_values[pos] = i + a*(0 + (disc_fac*stage_values[pos+1]) - i)
                else:
                    stage_values[pos] = i + a*(1 - stage_values[pos])
    
    print(stage_values)
    
loop()

    