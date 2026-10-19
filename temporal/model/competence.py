"""Competence: the chance of getting a question right, and how it grows."""
import math
import random

from .params import ModelState

DIFFICULTY_SCALE = 5.0  # single-diff task: logits per unit of `diff`
CORRECTION_GAIN = 0.06  # ability gained per corrected question, far below the ceiling
# A simulated person's ceiling sits where the bank's hardest type is still answered right
# only this often, so every target the controller can choose stays servable (see
# ability_ceiling). CEILING_WIDTH, in logits, is how gradually gains shrink on the way.
HARDEST_AT_CEILING = 0.2
CEILING_WIDTH = 1.0


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(z)))))


def item_diff(state: ModelState, s=None) -> float:
    if s is None or state.stage_diffs is None:
        return state.diff
    return float(state.stage_diffs[s])


# Chance of answering item `s` right. With `type_diffs` set, `s` is a question type.
def success_prob(state: ModelState, s=None) -> float:
    if state.type_diffs is not None and s is not None:
        return float(sigmoid(state.ability - float(state.type_diffs[s])))
    return float(sigmoid(state.ability - DIFFICULTY_SCALE * item_diff(state, s)))


# The ability at which the hardest of `type_diffs` has a HARDEST_AT_CEILING chance.
def ability_ceiling(type_diffs) -> float:
    return float(max(type_diffs)) + math.log(HARDEST_AT_CEILING / (1.0 - HARDEST_AT_CEILING))


# Ability gained by one correction: CORRECTION_GAIN far from the ceiling, shrinking
# toward 0 as ability nears it, so practice pays less the better they already are. With
# no ceiling (inf, the default) it is CORRECTION_GAIN exactly.
def correction_gain(state: ModelState) -> float:
    gap = max(0.0, state.ability_ceiling - state.ability)
    return CORRECTION_GAIN * (1.0 - math.exp(-gap / CEILING_WIDTH))


def draw_outcome(state: ModelState, s=None) -> int:
    return 1 if random.random() < success_prob(state, s) else 0


# Ability moves only when a missed question is shown again and answered right.
# `review` says it is that question coming back, not a fresh one of the same type.
def update_ability(state: ModelState, item: int, success: int, review: bool) -> None:
    if review:
        if success:
            state.missed.remove(item)  # the debt is paid
            state.ability += correction_gain(state)
            state.corrections += 1
        return  # missed again: still owed, and nothing is learned
    if not success:
        state.missed.append(item)  # a new question missed is owed from now on


# A question they gave up on came back and was answered.
def settle_skipped(state: ModelState, item: int, success: int) -> None:
    state.skipped.remove(item)
    if success:
        state.ability += correction_gain(state)  # a correction, as for a missed question
        state.corrections += 1
    else:
        state.missed.append(item)  # from now on it is owed like any miss
