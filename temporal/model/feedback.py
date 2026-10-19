"""Feedback, question by question: whether an outcome is shown now, and what a hidden one is worth.

A policy decides, for each question before a set's last, whether its outcome is shown the
moment it is answered or kept until the end-of-set score:

    on           every outcome shown
    off          every outcome kept until the score
    guard        rights shown; a miss is kept back when showing it would raise the chance
                 they leave on the next question by GUARD_MARGIN or more
    guard+cover  as guard, and rights are kept back at the rate misses have been, so that
                 a hidden outcome says nothing about how it went

A hidden outcome is not worth nothing. The person credits it with what they believe they
earned: their chance of having got it right, given that it was hidden, times its worth.
That belief is learned. Every hidden outcome is told at the score, and the offset moves by
the surprise, as confidence does (confidence.py):

    believed = sigmoid(logit(p) + offset)        offset += HIDDEN_RATE * (outcome - believed)

Hide only misses and every reveal is a 0: the offset falls, a hidden outcome comes to
read as a miss, and hiding stops changing anything. Hide rights as often as misses and
the reveals average out to the chance they had: the offset stays near 0 and a hidden
miss is credited at that chance until the score. The guard's own rule sees this. It
compares showing with hiding under the offset as it stands, so once hiding reads as a
miss the two come out equal and it stops hiding.
"""
import math

from .conditions import carried_delta, stay_cost
from .persistence import SOLVE_BASE, SOLVE_EXPONENT, log_survival, quit_drift
from .value import ACHIEVEMENT_SCALE, V

POLICIES = ("on", "off", "guard", "guard+cover")

HIDDEN_RATE = 0.5    # how far one revealed hidden outcome moves the offset; assumed, not fitted
OFFSET_CLIP = 3.0    # |offset| bound, in logits
GUARD_MARGIN = 0.01  # a miss is kept back if showing it adds this much chance of leaving
MATCH_RATE = 0.1     # how fast guard+cover's record of the share of misses kept back moves
P_FLOOR = 1e-3


def policy(feedback):
    # True and False are the per-set switch this replaced.
    if feedback is True:
        return "on"
    if feedback is False:
        return "off"
    if feedback not in POLICIES:
        raise ValueError(f"feedback must be one of {POLICIES}, got {feedback!r}")
    return feedback


def _logit(p):
    p = min(max(float(p), P_FLOOR), 1 - P_FLOOR)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


# Their chance of having got a question right, given only that its outcome was hidden.
def believed(p, offset):
    return sigmoid(_logit(p) + offset)


# The score told them how a hidden question went.
def learn(offset, outcome, believed_then):
    offset += HIDDEN_RATE * (outcome - believed_then)
    return max(-OFFSET_CLIP, min(OFFSET_CLIP, offset))


# Chance of leaving while working on the next question, after an error `delta` at slot
# `s`. The next question is taken at the target chance, for how long it takes; `guard` is
# (f, k, patience), the person as the environment estimates them.
def leave_chance(state, s, delta, target_p, guard):
    f, k, patience = guard
    p = min(max(float(target_p), P_FLOOR), 1 - P_FLOOR)
    mu = quit_drift(f, k, V(state, s + 1) + carried_delta(delta), stay_cost(state))
    t = SOLVE_BASE / p ** SOLVE_EXPONENT / patience  # their usual time on it, in model time
    return 1.0 - math.exp(float(log_survival(t, mu)))


# Whether showing this miss, rather than hiding it, adds GUARD_MARGIN to the chance they
# leave on the next question. `guess` is what hiding it would credit, on the achievement scale.
def guard_hides(state, params, s, guess, target_p, guard):
    rest = params.g * V(state, s + 1) - V(state, s)  # the error with nothing landing
    shown = leave_chance(state, s, rest, target_p, guard)
    hidden = leave_chance(state, s, rest + ACHIEVEMENT_SCALE * guess / state.stage_amt,
                          target_p, guard)
    return shown - hidden >= GUARD_MARGIN


# Whether the outcome of the question at slot `s` is shown now. `guess` is what hiding a
# miss would credit. With feedback off even the last is left to the score; otherwise the
# last is shown, since the score straight after it would tell it anyway.
def shows(state, params, pol, s, success, guess, target_p, guard):
    if pol == "off":
        return False
    if s == state.stage_amt - 1 or pol == "on":
        return True
    if success:
        if pol == "guard":
            return True
        # Cover: rights kept back at the rate misses have been, so hiding is no tell. Paced
        # rather than drawn, so the rate is met exactly and no random draw is taken.
        state.cover_debt += state.miss_hide_rate
        if state.cover_debt >= 1.0:
            state.cover_debt -= 1.0
            return False
        return True
    hide = guard_hides(state, params, s, guess, target_p, guard)
    if pol == "guard+cover":
        state.miss_hide_rate += MATCH_RATE * (hide - state.miss_hide_rate)
    return not hide
