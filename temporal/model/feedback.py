"""Feedback, question by question: whether an outcome is shown now or kept for the score.

A policy decides, for each question, whether its outcome is shown the moment it is
answered or kept until the end-of-set score:

    on           every outcome shown
    off          every outcome kept until the score, the last one too
    guard        rights shown; a miss is kept back when showing it would raise the chance
                 they leave on the next question by GUARD_MARGIN or more
    guard+cover  as guard, and rights are kept back at the rate misses have been, so that
                 a hidden outcome says nothing about how it went

What a hidden outcome means to the person is not a rule of its own. It is TD value: an
outcome kept back is pending until the score, and pending outcomes are features of the
state (value.phi), so the value of having one pending is learned by the same error as
everything else, at the same rate, when the score lands. Hide only misses and the scores
after hiding come in low: pending comes to be worth nothing, hiding a miss feels like
seeing it, and hiding stops changing anything. Hide rights as often and pending settles
at what an outcome is worth on average. The guard compares showing with hiding under the
value as it stands, so once hiding has been learned to mean a miss it stops hiding.
"""
import math

from .conditions import carried_delta, stay_cost
from .persistence import SOLVE_BASE, SOLVE_EXPONENT, log_survival, quit_drift
from .value import V

POLICIES = ("on", "off", "guard", "guard+cover")

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


# Chance of leaving while working on the next question, at slot `s + 1` with `pending`
# outcomes kept back and an error `delta` carried in. The next question is taken at the
# target chance, for how long it takes; `guard` is (f, k, patience), the person as the
# environment estimates them.
def leave_chance(state, s, pending, delta, target_p, guard):
    f, k, patience = guard
    p = min(max(float(target_p), P_FLOOR), 1 - P_FLOOR)
    mu = quit_drift(f, k, V(state, s + 1, pending) + carried_delta(delta), stay_cost(state))
    t = SOLVE_BASE / p ** SOLVE_EXPONENT / patience  # their usual time on it, in model time
    return 1.0 - math.exp(float(log_survival(t, mu)))


# Whether showing the miss at slot `s` (chance `p`), rather than keeping it back, adds
# GUARD_MARGIN to the chance they leave on the next question. Shown, it lands nothing;
# kept back, it joins what is pending, which is worth what the person has learned it is.
def guard_hides(state, params, s, p, target_p, guard):
    now = V(state, s)
    held = [*state.pending, p]
    shown = leave_chance(state, s, state.pending,
                         params.g * V(state, s + 1, state.pending) - now, target_p, guard)
    hidden = leave_chance(state, s, held, params.g * V(state, s + 1, held) - now, target_p, guard)
    return shown - hidden >= GUARD_MARGIN


# Whether the outcome of the question at slot `s`, with chance `p`, is shown now. With
# feedback off even the last is left to the score; otherwise the last is shown, since the
# score straight after it would tell it anyway.
def shows(state, params, pol, s, success, p, target_p, guard):
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
    hide = guard_hides(state, params, s, p, target_p, guard)
    if pol == "guard+cover":
        state.miss_hide_rate += MATCH_RATE * (hide - state.miss_hide_rate)
    return not hide
