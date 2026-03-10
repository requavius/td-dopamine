import random
import math

a = 0.01


def calculateconfidence(t, trained_params):
    newparams = trained_params.copy()
    for p in newparams[t-1]:
        if p == 'Bias': continue
        else: 
            newparams[t-1][p] *= trained_params[t - 1][p]
            print(f"feature: {p} is going to be multiplied ")
    z = sum(newparams[t-1].values()) 
    
    quit()
    
    guess = 1/(1+math.e**(-z)) 
    return guess

def calculateloss(cont, trained_params):
        firstlossvalue = []

        for t in range(trained_params):
            guess = calculateconfidence(t, trained_params)
            engagedornot = (cont)
            firstlossvalue.append(-((engagedornot * math.log(guess)) + (1-engagedornot) * math.log(1-guess)))

        lossvalues = ((sum(firstlossvalue)) / t)
        return lossvalues

def freezeparameter(cont, t, trained_params):
    
    gradients = {}
    for param in trained_params[t-1]:
        z = 0
        if param != 'Bias':
            for i in range(t):
                engagedornot = (cont)
                z += (calculateconfidence(t, trained_params) - engagedornot)*trained_params[t-1][param]
        else:
            for i in range(t):
                engagedornot = (cont)
                z += (calculateconfidence(t, trained_params) - engagedornot)
        gradients[param] = z / t
        
    return gradients

def train_pparams(cont, t, trained_params, learningrate = a):
    derivs = freezeparameter(cont, t, trained_params)
    newparams = trained_params[t-1].copy()
    for k in newparams:
        newparams[k] -= learningrate * derivs[k]
    return newparams

