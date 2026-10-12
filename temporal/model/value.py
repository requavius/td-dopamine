"""TD value of the task: what each slot of a set is worth, and the error when it lands."""
import numpy as np

from .params import ModelState, UserParams

# Scales a set's (1 - p)-weighted right answers so a perfect set is worth 1.0 per question.
ACHIEVEMENT_SCALE = 4.0
# True: a set's last slot bootstraps from V(0), so the task is a cycle. False: from 0.
CONTINUING = True


def phi(s: int, state: ModelState):
    # Slot position only. `diff` was a second feature, but it never varies during
    # learning, so theta[1] collapsed to theta[0]*d -- collinear with the intercept.
    s_norm = s / (state.stage_amt - 1)
    return np.array([1.0, s_norm])


def V(state: ModelState, s):
    return float(state.theta @ phi(s, state))


# TD(0) update at slot `s` with `reward` landing there. Returns the error.
def value_of_stage(state: ModelState, p: UserParams, s, reward=0.0):
    V_s = V(state, s)
    if s < state.stage_amt - 1:
        V_next = V(state, s + 1)
    else:
        V_next = V(state, 0) if CONTINUING else 0.0  # bootstrap the last slot

    delta = reward + p.g * V_next - V_s
    state.theta = state.theta + p.a * delta * phi(s, state)
    return delta


# What arrives at a slot, on the achievement scale, given the set's achievement so far
# and how much of it has already been paid. With feedback shown, a question's worth lands
# the moment its outcome does; with it hidden nothing lands until the end-of-set score,
# which pays the set in one lump. A set totals the same either way -- feedback moves when
# the signal arrives, not how much, and that timing is the whole of its effect.
# Returns (reward, achievement now credited).
def slot_reward(state: ModelState, achievement: float, credited: float,
                feedback: bool, last: bool) -> tuple:
    due = achievement if (feedback or last) else credited
    return ACHIEVEMENT_SCALE * (due - credited) / state.stage_amt, due
