"""What each decision node feeds the accumulator: its (value, cost).

The start node and the redo node share one accumulator (accumulator.py), so their
conditions are on one scale. The give-up node's are in persistence.conditions.
"""
import numpy as np

from .competence import success_prob
from .params import ModelState, UserParams
from .value import V

WARM_COST = 0.05     # already sitting there, next set one click away
EFFORT_WEIGHT = 0.1  # scales last_episode_effort (1 + review load) into cost units

# Redo cost: expected attempts to get the question right (1/p), scaled and capped.
REPEAT_COST_SCALE = 0.15
MAX_REPEAT_ATTEMPTS = 8.0

# The prediction error carried out of a slot, or a whole set, into the next decision.
# Pure RPE: the carry weight is 1 by construction, not a fitted parameter. The clip only
# keeps `value` on a bounded grid so draws still hit accumulator._SOLUTION_CACHE; measured
# deltas sit well inside it (see README, "Feedback is timing, not reward").
DELTA_CLIP = 1.0


def carried_delta(delta: float) -> float:
    return float(np.clip(delta, -DELTA_CLIP, DELTA_CLIP))


# Start node, value: the learned value of starting a set, plus the surprise of the set
# just played. A set that went worse than predicted carries its negative error into the
# decision to start another.
def value_at_choice_point(state: ModelState) -> float:
    return V(state, 0) + carried_delta(state.last_set_delta)


# Start node, cost: (structural, effort).
def cost_components(state: ModelState) -> tuple:
    base = WARM_COST if state.in_session else state.reentry_cost  # friction of getting to the task
    return base, EFFORT_WEIGHT * state.last_episode_effort  # what the last set took out of them


# Redo node: (value, cost) after a miss. `s` is the slot, for discounting, and `item`
# the question, for difficulty.
def stage_conditions(state: ModelState, p: UserParams, s: int, item=None) -> tuple:
    remaining = state.stage_amt - 1 - s
    # This slot's discounted share of the reward, plus the error just taken at it: the
    # node fires on a miss the person was shown, so that error is the miss landing.
    value = (1.0 / state.stage_amt) * (p.g ** remaining) + carried_delta(state.slot_delta)
    p_item = success_prob(state, s if item is None else item)
    expected_attempts = min(1.0 / max(p_item, 1e-3), MAX_REPEAT_ATTEMPTS)  # cost of insisting on it
    return value, REPEAT_COST_SCALE * expected_attempts
