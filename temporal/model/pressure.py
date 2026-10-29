"""Time pressure on the redo prompt: a countdown, and a lower bound under it.

Under time pressure people lower their response threshold, and the striatum and pre-SMA
carry that adjustment (Forstmann et al., 2008): they wait for less evidence, so they
decide sooner and less consistently. A pressured redo prompt shows a countdown of
REDO_DEADLINE seconds. Undecided by then, the prompt closes and the question is not
redone. The person's bound on it is `UserParams.pressure` × BOUND (1 keeps the bound);
their drift, f·value − k·cost, is unchanged. Time pressure changes how much evidence they
wait for, not what the evidence says, so the decision still observes the same (f, k).

Draws come from the closed-form likelihood (temporal.fastlik), which agrees with pyddm's
solution within 0.2% of the density's peak (tests/test_fastlik.py), so a new bound or
deadline costs no Fokker-Planck solve. They use numpy's global generator as
accumulator._draw does.
"""
import numpy as np

from .. import fastlik
from . import accumulator as acc

REDO_DEADLINE = 3.0  # seconds on the countdown
CACHE_CAP = 20000
_TABLES = {}


def _table(f_val, k_val, value, cost, t0, bound, deadline):
    key = (f_val, k_val, value, cost, t0, bound, deadline)
    if key not in _TABLES:
        if len(_TABLES) > CACHE_CAP:
            _TABLES.clear()
        drift = acc.initiation_drift(value, cost, f_val, k_val)
        up, lo, und = fastlik.outcome_probs([drift], bound, t0, t_dur=deadline, dt=acc.DT)
        t = np.arange(up.shape[1]) * acc.DT  # bin starts; a uniform within the bin is added
        shift = t[-1] + 1  # lower-bound times sit past the upper ones
        probs = np.concatenate([up[0], lo[0], und])
        _TABLES[key] = (np.concatenate([t, t + shift, [-1]]), probs / probs.sum(), shift)
    return _TABLES[key]


# The redo prompt under a countdown. GO 1 takes the question again, 0 moves on, None is
# undecided at the deadline, when the prompt closes and they move on (chose 0).
def pressured_decision(value, cost, f_val, k_val, t0=0.0, scale=1.0, deadline=REDO_DEADLINE):
    value = acc.snap_value(value)
    cost = acc.snap_cost(cost)
    bound = round(acc.BOUND * scale, 6)  # acc.BOUND as it is now: a person's own, inside theirs
    domain, probs, shift = _table(round(f_val, 6), round(k_val, 6), value, float(cost),
                                  round(t0, 6), bound, deadline)
    samp = np.random.choice(domain, p=probs, replace=True, size=1)
    samp = samp[samp != -1]
    samp += np.random.uniform(0, acc.DT, len(samp))
    if len(samp) == 0:
        go, rt = None, None
    elif samp[0] < shift:
        go, rt = 1, float(samp[0])
    else:
        go, rt = 0, float(samp[0] - shift)
    return {"value": value, "cost": float(cost), "GO": go, "RT": rt,
            "chose": int(go == 1), "node": "stage", "deadline": deadline}
