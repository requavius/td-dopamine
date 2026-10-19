"""TD value of the task: what each slot of a set is worth, and the error when it lands."""
import numpy as np

from .params import ModelState, UserParams

# Scales a set's (1 - p)-weighted right answers so a perfect set is worth 1.0 per question.
ACHIEVEMENT_SCALE = 4.0
# True: a set's last slot bootstraps from V(0), so the task is a cycle. False: from 0.
CONTINUING = True


# Features of slot `s` with `pending` the chances of the outcomes this set has hidden and
# not yet told. Slot position, plus two for what is pending: its worth if every pending
# one were right, sum (1 - p), and its worth if each is right at its chance, sum p(1 - p),
# both on the reward's scale. Their weights are what a hidden outcome means to this
# person, learned by the same TD error as everything else: the score that tells pending
# outcomes lands at the set's last slot, against the value those features predicted.
# With nothing ever hidden both are 0, and V is what it was with position alone.
def phi(s: int, state: ModelState, pending=None):
    # Slot position only, for the shown task. `diff` was a feature once, but it never
    # varies during learning, so its weight collapsed to theta[0]*d -- collinear with the
    # intercept.
    s_norm = s / (state.stage_amt - 1)
    pending = state.pending if pending is None else pending
    scale = ACHIEVEMENT_SCALE / state.stage_amt
    if_right = scale * sum(1.0 - q for q in pending)
    at_chance = scale * sum(q * (1.0 - q) for q in pending)
    return np.array([1.0, s_norm, if_right, at_chance])


def V(state: ModelState, s, pending=None):
    return float(state.theta @ phi(s, state, pending))


# TD(0) update at slot `s` with `reward` landing there. Returns the error. `before` is what
# was pending when the slot began, `after` once its outcome was shown or kept back; both
# default to what is pending now.
def value_of_stage(state: ModelState, p: UserParams, s, reward=0.0, before=None, after=None):
    before = state.pending if before is None else before
    after = state.pending if after is None else after
    x = phi(s, state, before)
    if s < state.stage_amt - 1:
        V_next = V(state, s + 1, after)
    else:
        # The score has told everything: the next set starts with nothing pending.
        V_next = V(state, 0, ()) if CONTINUING else 0.0  # bootstrap the last slot

    delta = reward + p.g * V_next - float(state.theta @ x)
    state.theta = state.theta + p.a * delta * x
    return delta


# What arrives at a slot, on the achievement scale. `worth` is what this question earned,
# `achievement` the set's total so far and `credited` how much of it has been paid. A shown
# outcome lands its worth; a hidden one lands nothing, and the end-of-set score pays the
# rest. A set totals the same either way. What a hidden outcome feels worth meanwhile is
# not paid here: it is the value of having it pending (phi), which the person learns.
# Returns (reward, achievement now credited).
def slot_reward(state: ModelState, worth: float, shown: bool,
                achievement: float, credited: float, last: bool) -> tuple:
    due = achievement if last else credited + (worth if shown else 0.0)
    return ACHIEVEMENT_SCALE * (due - credited) / state.stage_amt, due
