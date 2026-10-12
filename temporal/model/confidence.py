"""Confidence: what a person believes their chance on a question is, and how outcomes move it.

Confidence `c` is a logit offset on the chance the environment would predict: the person
works on a question as if their chance were `perceived(p, c) = sigmoid(logit(p) + c)`.
It enters every judgement they make about a question -- what solving it would add and
what it costs, when deciding whether to give up (persistence.conditions) and whether to
redo a miss (conditions.stage_conditions). Rewards and the TD value are untouched: what a
set earned is what the environment measured.

How outcomes move `c` is the person's, set by `eta` and a rule:

    elo     c += eta * (outcome - p_hat)                  surprise-weighted: an expected easy
                                                          win barely registers
    count   c = (c + eta * (outcome - ref)) * (1 - REVERT)   wins and losses counted against a
                                                          personal reference rate
    asym    as elo, but a loss moves c twice as far: not among HYPOTHESES, so it tests a
            fit facing a rule it does not have

A skip is an outcome of 0. `eta = 0` is no confidence dynamics at all; `c` then stays
exactly 0 and every quantity is bit-for-bit what it was before confidence existed.
"""
import math

RULES = ("elo", "count", "asym")
REVERT = 0.05  # count rule: drift back toward calibrated per question
C_CLIP = 3.0   # |c| bound, in logits
P_FLOOR = 1e-3

# The hypotheses a fit weighs: (rule, eta, ref). The first is "none".
NONE = ("elo", 0.0, 0.75)
HYPOTHESES = (NONE, ("elo", 0.5, 0.75), ("elo", 1.0, 0.75),
              ("count", 0.5, 0.65), ("count", 1.0, 0.65),
              ("count", 0.5, 0.85), ("count", 1.0, 0.85))
PRIOR_NONE = 0.5  # prior weight on "none"; the rest is spread evenly


def perceived(p, c):
    if c == 0.0:
        return p
    p = min(max(float(p), P_FLOOR), 1 - P_FLOOR)
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) + c)))


def update(c, outcome, p_hat, eta, rule="elo", ref=0.75):
    if eta == 0.0:
        return c
    if rule == "elo":
        c = c + eta * (outcome - p_hat)
    elif rule == "count":
        c = (c + eta * (outcome - ref)) * (1 - REVERT)
    elif rule == "asym":  # losses count twice; for testing a rule no fit assumes
        c = c + eta * (outcome - p_hat) * (1.0 if outcome else 2.0)
    else:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    return max(-C_CLIP, min(C_CLIP, c))


# c before each question, from the questions' (p, outcome) in the order they were worked.
def trajectory(attempts, hypothesis):
    rule, eta, ref = hypothesis
    c, out = 0.0, []
    for p, outcome in attempts:
        out.append(c)
        c = update(c, outcome, perceived(p, c), eta, rule, ref)
    return out, c


def prior(hypotheses=HYPOTHESES):
    rest = (1 - PRIOR_NONE) / (len(hypotheses) - 1)
    return [PRIOR_NONE if h == NONE else rest for h in hypotheses]
