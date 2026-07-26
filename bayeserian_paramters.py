from config import ModelState, np


def bayesian_particle_update(disengaged, delta, s, state: ModelState):

    probs, _ = None

    likelihoods = probs if not disengaged else (1 - probs)

    new_weights = state.weights * np.maximum(likelihoods, 1e-8)
    total = new_weights.sum()
    if total == 0:
        new_weights = np.ones_like(new_weights) / len(new_weights)
    else:
        new_weights /= total

    return new_weights


def resample_if_needed(state: ModelState, threshold=0.5):
    n = len(state.weights)
    ess = 1.0 / np.sum(state.weights**2)
    if ess < threshold * n:
        indices = np.random.choice(n, size=n, p=state.weights)
        for s in range(state.particle_matrix.shape[0]):
            state.particle_matrix[s] = np.clip(
                state.particle_matrix[s][indices] + np.random.normal(0, 0.02, n),
                0.05,
                1.0,
            )
        state.weights = np.ones(n) / n
