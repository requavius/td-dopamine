from config import (
    ModelState, UserParams, V, phi, sigmoid, get_sigma,
    stage_amt, diff, a, g, param_values, np, pd,
)
from ddm import ddm, effective_delta
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
        r, V_next = 0.0, V(state.theta, s + 1)

    delta = r + g * V_next - V_s
    state.theta = state.theta + a * delta * phi(s)
    model_solved, dt, model = ddm.modelsolve(
        delta, s, state, pers_param.f, pers_param.k, pers_param.b
    )
    disengaged = False if model_solved.pdf("correct")[-1] < random.random() else True

    state.stage_log.append(
        {
            "trial": state.t,
            "stage": s,
            "dt": dt,
            "delta": delta,
            "V": V_s,
        }
    )

    learning_gain = (
        max(0, delta) * (1.0 - state.skill)
    )  # skill grows with practice but saturates. This might be changed based on what makes sense for skill improvment
    state.skill += min(learning_gain / stage_amt, 1)
    if not disengaged:
        return delta, disengaged, dt
    else:
        return delta, disengaged, dt, model


def simulate(state: ModelState, pers_param):
    for s in range(stage_amt):
        state.rpe[s], disengaged, dt, *model = value_of_stage(state, s, pers_param)
        state.first_passage.loc[
            len(state.first_passage), ["Disengagment", "Engaged T", "delta"]
        ] = [disengaged, dt, round(effective_delta(state.rpe[s], state, s), 2)]
        if disengaged:
            break
    return False if not disengaged else True, model


def train(state: ModelState, pers_param):
    low_rpe_streak = 0
    while True:
        disengaged, *model = simulate(state, pers_param)

        max_rpe = max(abs(x) for x in state.rpe.values())

        if max_rpe < 0.05:
            low_rpe_streak += 1
        else:
            low_rpe_streak = 0

        average_v = sum(V(state.theta, s) for s in range(stage_amt)) / stage_amt

        state.t += 1

        if (low_rpe_streak >= 10 and average_v > 0.1) or state.t >= 100:
            return model


def test_train(true_f=None, true_k=None, true_b=None, debug=False, extra=0, repeat=1):
    repeats = {}
    if true_f is None:
        true_f = random.uniform(0, 1.0)
    if true_k is None:
        true_k = random.uniform(0, 1.0)
    if true_b is None:
        true_b = random.uniform(0, 1.0)

    fixed = UserParams(f=true_f, k=true_k, b=true_b)
    n_particles = 100
    cols = ["Disengagment", "Engaged T", "delta"]
    state = ModelState(
        theta=np.zeros(len(param_values) + 1),
        t=1,
        weights=np.ones(n_particles) / n_particles,
        particle_matrix=np.random.uniform(0.05, 1.0, size=(3, n_particles)),
        skill=0.1,
        first_passage=pd.DataFrame(columns=cols),
    )

    for i in range(repeat):
        train(state, fixed)
        _, fitted = ddm().loglikelihood(state)
        repeats[i] = {
            "f": float(fitted["drift"]["f_val"]),
            "k": float(fitted["IC"]["k_val"]),
            "b": float(fitted["bound"]["b_val"]),
            "t": state.t,
        }
    est_f = repeats[repeat - 1]["f"]
    est_k = repeats[repeat - 1]["k"]
    est_b = repeats[repeat - 1]["b"]
    if debug:
        print(f"stopped after {state.t} trials")
        print("V:", [round(V(state.theta, s), 3) for s in range(stage_amt)])
        print("RPE:", [round(state.rpe[s], 3) for s in range(stage_amt)])
        print("Estimated f:", est_f)
        print("Estimated k:", est_k)
        print("Estimated b:", est_b)
        print("True params:", fixed)

    if not extra:
        return {
            "true_f": true_f,
            "true_k": true_k,
            "true_b": true_b,
            "est_f": est_f,
            "est_k": est_k,
            "est_b": est_b,
            "trials": state.t,
        }
    elif extra == 1:
        log = state.stage_log
        return fixed, log, state.t
    elif extra == 2:
        return repeats
