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


# What arrives at a slot, on the achievement scale. `worth` is what this question earned;
# `guess` is what they credit it with if its outcome is hidden (feedback.believed times
# what a right answer is worth); `achievement` is the set's total so far and `credited`
# how much of it has been paid. A shown outcome lands its worth; a hidden one lands the
# guess, and the end-of-set score pays whatever is left, so a set totals the same either
# way -- feedback moves when the signal arrives, and how wrong the guess was.
# Returns (reward, achievement now credited).
def slot_reward(state: ModelState, worth: float, shown: bool, guess: float,
                achievement: float, credited: float, last: bool) -> tuple:
    due = achievement if last else credited + (worth if shown else guess)
    return ACHIEVEMENT_SCALE * (due - credited) / state.stage_amt, due
