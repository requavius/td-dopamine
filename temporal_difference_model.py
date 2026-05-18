from engagement_bayeserian import *
from config import *
from ddm import *
import random

def value_of_stage(state: ModelState, s, pers_param):
    V_s = V(state.theta, s)
    if s == stage_amt - 1:
        sigma = get_sigma(state)
        reward_divergence = random.gauss(0, sigma)
        r = sigmoid(5 * (state.skill - diff)) + reward_divergence
        r = max(0, min(1, r))
        V_next = 0.0
    else:
        r, V_next = 0.0, V(state.theta, s+1)

    delta = r + g * V_next - V_s
    state.theta = state.theta + a * delta * phi(s)
    model, dt = ddm(delta, s, state, pers_param.f, pers_param.k, pers_param.b) 
    chance = random.random()
    disengaged = False if model < chance else True
    
    state.stage_log.append({
        'trial': state.t,
        'stage': s,
        'dt': dt,
        'delta': delta,
        'V': V_s,
    })
    
    learning_gain = max(0,delta) * (1.0 - state.skill)  # skill grows with practice but saturates. This might be changed based on what makes sense for skill improvment
    state.skill += (min(learning_gain / stage_amt, 1))
    return delta, disengaged

def simulate(state: ModelState, pers_param):
    for s in range(stage_amt):
        state.rpe[s], disengaged = value_of_stage(state, s, pers_param)
        if disengaged: break
    state.first_passage[len(state.first_passage)] = [disengaged, state.t, [state.rpe[s] for s in range(stage_amt)]]
    return True if not disengaged else False

def train(state: ModelState, pers_param, debug):
    low_rpe_streak = 0
    while True:
        engaged = simulate(state, pers_param)

        max_rpe = max(abs(x) for x in state.rpe.values())

        if max_rpe < 0.05:
            low_rpe_streak += 1
        else:
            low_rpe_streak = 0

        average_v = sum(V(state.theta, s) for s in range(stage_amt))/stage_amt
        
        if low_rpe_streak >= 10 and average_v > 0.1 or not engaged or state.t >= 2000:
            if debug:
                pass
            return 

    
        state.t += 1
        
        
def test_train(true_f = None, true_k = None, true_b = None, debug = False, extra = 0, repeat = 1):
    repeats = {}
    if true_f is None: true_f = random.uniform(0.05, 1.0)
    if true_k is None: true_k = random.uniform(0.05, 1.0)
    if true_b is None: true_b = random.uniform(0.05, 1.0)
    
    fixed = UserParams(f=true_f, k=true_k, b=true_b)
    n_particles = 100
    cols = ['Engagment', 'Engaged T', 'Delta']
    state = ModelState(
        theta = np.zeros(len(param_values) + 1),
        t = 1,
        weights = np.ones(n_particles) / n_particles,
        particle_matrix = np.random.uniform(0.05, 1.0, size = (3, n_particles)),
        skill = .1,
        first_passage = pd.DataFrame(columns = cols)
    )

    for i in range(repeat):
        train(state, fixed, debug)
        repeats[i] = {'f' : np.dot(state.weights, state.particle_matrix[0]), 'k' : np.dot(state.weights, state.particle_matrix[1]), 'b' : np.dot(state.weights, state.particle_matrix[2])}
    if debug == True:
        print(f"stopped after {state.t} trials")
        print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
        print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
        print("Estimated f:", np.dot(state.weights, state.particle_matrix[0]))
        print("Estimated k:", np.dot(state.weights, state.particle_matrix[1]))
        print("Estimated b:", np.dot(state.weights, state.particle_matrix[2]))
        print("True params:", fixed)
    
    est_f = np.dot(state.weights, state.particle_matrix[0])
    est_k = np.dot(state.weights, state.particle_matrix[1])
    est_b = np.dot(state.weights, state.particle_matrix[2])
    

    
    
    if not extra:
        return {'true_f': true_f, 'true_k': true_k, 'true_b': true_b,
                'est_f': est_f, 'est_k': est_k, 'est_b': est_b, 'trials': state.t}
    elif extra == 1:
        log = state.stage_log
        return fixed, log, state.t
    elif extra == 2:
        return repeats




