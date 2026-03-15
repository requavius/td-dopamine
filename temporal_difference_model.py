import numpy as np
import random
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

g = .9 

a =.05

# task parameters (model tweaks):
stage_amt = 4 # How many stages there are until reward
diff = .1 # The difficulty of each stage; will not be changed yet until ready for infererence

param_values = {
    'bias' : 1,   
    'd' : diff,
} 

@dataclass
class UserParams:
    f: float  # Sensitivity to learning progress  
    k: float  # Effort aversion  
    b: float  # Boredom rate 

@dataclass
class ModelState:
    theta: np.ndarray
    t: int
    weights: np.ndarray
    skill: float  # initial preformance that will scale
    f_arr: np.ndarray
    k_arr: np.ndarray
    b_arr: np.ndarray
    highest_eng: float
    t_since_eng: int
    stage_log: list = field(default_factory=list)
    episode_log: list = field(default_factory=list)
    rpe: dict = field(default_factory=lambda: {r: 0 for r in range(stage_amt)})



def get_sigma(state: ModelState, base_sigma=.02, scaling_factor=.3):
    raw_sigma = base_sigma + diff * scaling_factor + 0.1
    sigma = raw_sigma / math.sqrt(state.skill) if state.skill != 0 else raw_sigma
    return sigma

def engagement_score(delta, v, f_val, k_val, b_val, state: ModelState, part = False):
    delta_scale = 10 * (sigmoid(abs(delta)))
    
    signal = f_val * delta_scale + 1
    denom = max(0, abs(v) + state.skill) + 1
    effort_cost = (k_val) * (((state.t * .01) * stage_amt) / (denom - diff))
    boredom_cost = (b_val) * (state.t_since_eng)
    
    score = signal - effort_cost - boredom_cost


    if delta >= state.highest_eng and not part:
        state.t_since_eng = 0
        state.highest_eng = delta
    

    return score

def engage(state: ModelState, formula):
    cont = formula
    prob = sigmoid(cont)
    decision = 1 if random.random() < prob else 0
    if not decision: state.highest_eng = -math.inf
    return decision


def phi(s: int):
    d = diff
    s_norm = s / (stage_amt - 1)
    return np.array([1.0, d, s_norm])

def sigmoid(z):
    z = max(-60, min(60, z))
    return 1 / (1 + math.exp(-z))

def sigmoid_vec(z):
    z = np.clip(z, -60, 60)
    return 1 / (1 + np.exp(-z))

def bayesian_particle_update(engaged, delta, v, state: ModelState):
    delta_scale = 10 * sigmoid(abs(delta))
    signal = state.f_arr * delta_scale + 1
    denom = max(0, abs(v) + state.skill) + 1
    effort_cost = (state.k_arr) * (((state.t * .01) * stage_amt) / (denom - diff))
    boredom_cost = (state.b_arr) * (state.t_since_eng)
    scores = signal - effort_cost - boredom_cost

    probs = sigmoid_vec(scores)
    likelihoods = probs if engaged else (1 - probs)

    new_weights = state.weights * np.maximum(likelihoods, 1e-8)
    total = new_weights.sum()
    if total == 0:
        new_weights = np.ones_like(new_weights) / len(new_weights)
    else:
        new_weights /= total

    return new_weights

def resample_if_needed(state: ModelState, threshold=0.5):
    n = len(state.weights)
    ess = 1.0 / np.sum(state.weights ** 2)
    if ess < threshold * n:
        indices = np.random.choice(n, size=n, p=state.weights)
        state.f_arr = np.clip(state.f_arr[indices] + np.random.normal(0, 0.02, n), 0.05, 1.0)
        state.k_arr = np.clip(state.k_arr[indices] + np.random.normal(0, 0.02, n), 0.05, 1.0)
        state.b_arr = np.clip(state.b_arr[indices] + np.random.normal(0, 0.02, n), 0.05, 1.0)
        state.weights = np.ones(n) / n

def V(theta, s):

    v = float((theta @ phi(s)))
    return v

def value_of_stage(state: ModelState, s, pers_param):
    V_s = V(state.theta, s)
    if s == stage_amt - 1:
        sigma = get_sigma(state)
        reward_divergence = random.gauss(0, sigma)
        r = sigmoid(5 * (state.skill - diff)) + reward_divergence
        r = max(0, min(1, r))
        #r = 1
        V_next = 0.0
    else:
        r, V_next = 0.0, V(state.theta, s+1)

    delta = r + g * V_next - V_s
    state.theta = state.theta + a * delta * phi(s)
    stage_engagement = engagement_score(delta, V_s, pers_param.f, pers_param.k, pers_param.b, state) 
    engaged_observation = engage(state, stage_engagement)
    engagment_prob = sigmoid(stage_engagement)
    state.weights = bayesian_particle_update(engaged_observation, delta, V_s, state)
    resample_if_needed(state)

    state.stage_log.append({
        'trial': state.t,
        'stage': s,
        'engagement_score': stage_engagement,
        'engagement_prob': engagment_prob,
        'engaged_obs': engaged_observation,
        'delta': delta,
        'V': V_s,
    })
    
    learning_gain = max(0,delta) * (1.0 - state.skill)  # skill grows with practice but saturates. This might be changed based on what makes sense for skill improvment
    state.skill += (min(learning_gain / stage_amt, 1))
    return delta, stage_engagement, engaged_observation

def simulate(state: ModelState, pers_param):
    tot_epi_engagement = 0
    stages_completed = 0
    for s in range(stage_amt):
        state.rpe[s], tot_stage_engagement, engaged = value_of_stage(state, s, pers_param)
        tot_epi_engagement += tot_stage_engagement
        stages_completed += 1
        if not engaged and state.t > 1:
            state.t_since_eng = 0
            break
    state.episode_log.append({
        'trial': state.t,
        'total_engagement': tot_epi_engagement, # will be re engagment factor
        'Stages completed': stages_completed, # how many stages were completed before disengagement
        'max_abs_rpe': max(abs(x) for x in state.rpe.values()),
        'est_f': np.dot(state.weights, state.f_arr),
        'est_k': np.dot(state.weights, state.k_arr),
        'est_b': np.dot(state.weights, state.b_arr),
    })
    state.t_since_eng += 1 
    

def train(state: ModelState, pers_param, debug):
    low_rpe_streak = 0
    while True:
        simulate(state, pers_param)

        max_rpe = max(abs(x) for x in state.rpe.values())

        if max_rpe < 0.05:
            low_rpe_streak += 1
        else:
            low_rpe_streak = 0

        average_v = sum(V(state.theta, s) for s in range(stage_amt))/stage_amt
        
        if low_rpe_streak >= 10 and average_v > 0.1:
            if debug:
                print(f"trained after {state.t} trials")
                print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
                print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
            return 

        if state.t >= 2000:
            if debug:
                print(f"stopped after {state.t} trials")
                print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
                print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
            return 
    
        state.t += 1
        
        
def test_train(true_f = None, true_k = None, true_b = None, debug = False):
    if true_f is None: true_f = random.uniform(0.05, 1.0)
    if true_k is None: true_k = random.uniform(0.05, 1.0)
    if true_b is None: true_b = random.uniform(0.05, 1.0)
    
    fixed = UserParams(f=true_f, k=true_k, b=true_b)
    n_particles = 100
    weights = np.ones(n_particles) / n_particles
    theta = np.zeros(len(param_values) + 1)

    state = ModelState(
        theta = theta,
        t = 1,
        weights = weights,
        f_arr = np.random.uniform(0.05, 1.0, n_particles),
        k_arr = np.random.uniform(0.05, 1.0, n_particles),
        b_arr = np.random.uniform(0.05, 1.0, n_particles),
        skill = .1,
        highest_eng = -math.inf,
        t_since_eng = 0,
    )

    train(state, fixed, debug)
    
    avg_stages = sum(ep['Stages completed'] for ep in state.episode_log) / len(state.episode_log)
    if debug == True:
        
        print("Estimated f:", np.dot(state.weights, state.f_arr))
        print("Estimated k:", np.dot(state.weights, state.k_arr))
        print("Estimated b:", np.dot(state.weights, state.b_arr))
        print(f"Average stages completed per episode: {avg_stages:.2f}")
        print("True params:", fixed)
    
    est_f = np.dot(state.weights, state.f_arr)
    est_k = np.dot(state.weights, state.k_arr)
    est_b = np.dot(state.weights, state.b_arr)
    return {'true_f': true_f, 'true_k': true_k, 'true_b': true_b, 'avg_stages': avg_stages,
            'est_f': est_f, 'est_k': est_k, 'est_b': est_b}

def collect_results(n=60, repeats=5):
    results = []
    sweep = np.linspace(0.05, 0.95, n)
    fixed = 0.5

    for i, val in enumerate(sweep):
        f_runs = [test_train(val, fixed, fixed) for _ in range(repeats)]
        k_runs = [test_train(fixed, val, fixed) for _ in range(repeats)]
        b_runs = [test_train(fixed, fixed, val) for _ in range(repeats)]
        results.append({'param': 'f', 'true_f': val, 'true_k': fixed, 'true_b': fixed,
                        'avg_stages': np.mean([r['avg_stages'] for r in f_runs]),
                        'est_f': np.mean([r['est_f'] for r in f_runs]),
                        'est_k': np.mean([r['est_k'] for r in f_runs]),
                        'est_b': np.mean([r['est_b'] for r in f_runs])})
        results.append({'param': 'k', 'true_f': fixed, 'true_k': val, 'true_b': fixed,
                        'avg_stages': np.mean([r['avg_stages'] for r in k_runs]),
                        'est_f': np.mean([r['est_f'] for r in k_runs]),
                        'est_k': np.mean([r['est_k'] for r in k_runs]),
                        'est_b': np.mean([r['est_b'] for r in k_runs])})
        results.append({'param': 'b', 'true_f': fixed, 'true_k': fixed, 'true_b': val,
                        'avg_stages': np.mean([r['avg_stages'] for r in b_runs]),
                        'est_f': np.mean([r['est_f'] for r in b_runs]),
                        'est_k': np.mean([r['est_k'] for r in b_runs]),
                        'est_b': np.mean([r['est_b'] for r in b_runs])})
        print(f"completed {i+1}/{n}")

    return results

def plot_results(results):
    f_sweep = [(r['true_f'], r['avg_stages']) for r in results if r['param'] == 'f']
    k_sweep = [(r['true_k'], r['avg_stages']) for r in results if r['param'] == 'k']
    b_sweep = [(r['true_b'], r['avg_stages']) for r in results if r['param'] == 'b']

    f_est = [(r['true_f'], r['est_f']) for r in results if r['param'] == 'f']
    k_est = [(r['true_k'], r['est_k']) for r in results if r['param'] == 'k']
    b_est = [(r['true_b'], r['est_b']) for r in results if r['param'] == 'b']

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(*zip(*sorted(k_sweep)), color='#FF5722', label='k (effort aversion)')
    ax1.plot(*zip(*sorted(f_sweep)), color='#2196F3', label='f (progress sensitivity)')
    ax1.plot(*zip(*sorted(b_sweep)), color='#4CAF50', label='b (boredom rate)')
    ax1.set_xlabel('Parameter value')
    ax1.set_ylabel('Average stages completed')
    ax1.set_title('Parameter vs Engagement')
    ax1.legend()

    lims = [0.05, 0.95]
    ax2.plot(lims, lims, 'k--', alpha=0.4, label='ideal recovery')
    ax2.scatter(*zip(*sorted(f_est)), color='#2196F3', s=15, alpha=0.7, label='est f')
    ax2.scatter(*zip(*sorted(k_est)), color='#FF5722', s=15, alpha=0.7, label='est k')
    ax2.scatter(*zip(*sorted(b_est)), color='#4CAF50', s=15, alpha=0.7, label='est b')
    ax2.set_xlabel('True parameter value')
    ax2.set_ylabel('Estimated parameter value')
    ax2.set_title('Parameter Recovery')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.legend()

    plt.tight_layout()

    plt.savefig('engagement_by_params.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_results(collect_results(60,10))
#test_train(0.1, debug=True)
