"""The accumulator: one drift-diffusion process behind every start and redo decision.

SIGMA, BOUND and T_DUR are pinned on purpose: freed, they absorb variance that belongs
to f and k, and identifiability goes with it.
"""
import numpy as np
import pyddm

SIGMA = 0.7
BOUND = 1.5
T_DUR = 7.0  # deadline in seconds; past it a decision is undecided. Re-check: `python -m temporal audit`

DX = 0.005
DT = 0.005
LAPSE_MIX = 0.02  # pyddm's uniform lapse mixture, its default; stated so _sampler can undo it

# Conditions are snapped to these steps so repeated draws hit _SOLUTION_CACHE.
VALUE_STEP = 0.05
COST_STEP = 0.05


# One drift form at every node. What the person makes of the set they just played
# reaches it through `value`, as the prediction error carried out of that set
# (conditions.value_at_choice_point, conditions.stage_conditions) -- not as terms of its own.
def initiation_drift(value, cost, f_val, k_val):
    return f_val * value - k_val * cost


# The two-parameter model (f, k). `nondecision` is seconds: a number fixes it, a
# parameter name fits it.
def build_model(parameters, nondecision=0):
    return pyddm.gddm(
        drift=initiation_drift,
        noise=SIGMA,
        bound=BOUND,
        starting_position=0,
        nondecision=nondecision,
        mixture_coef=LAPSE_MIX,
        T_dur=T_DUR,
        dx=DX,
        dt=DT,
        parameters=parameters,
        conditions=["value", "cost"],
    )


# `value` carries a prediction error, so it can be negative; round() handles that.
def snap_value(value):
    return round(round(float(value) / VALUE_STEP) * VALUE_STEP, 4)


def snap_cost(cost):
    return round(round(float(cost) / COST_STEP) * COST_STEP, 4)


_MODEL_CACHE = {}
_SOLUTION_CACHE = {}
_SAMPLER_CACHE = {}


def clear_caches():
    _MODEL_CACHE.clear()
    _SOLUTION_CACHE.clear()
    _SAMPLER_CACHE.clear()


def _r(x):
    return round(float(x), 6)


def _model(f_val, k_val, t0=0.0):
    key = (_r(f_val), _r(k_val), _r(t0))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_model({"f_val": key[0], "k_val": key[1]}, nondecision=key[2])
    return _MODEL_CACHE[key]


# The solved choice and RT distributions for one person in one condition, cached.
def solution(f_val, k_val, value, cost, t0=0.0):
    key = (_r(f_val), _r(k_val), value, _r(cost), _r(t0))
    if key not in _SOLUTION_CACHE:
        _SOLUTION_CACHE[key] = _model(f_val, k_val, t0).solve(
            conditions={"value": value, "cost": cost})
    return _SOLUTION_CACHE[key]


# pyddm's Solution.sample(1), with its tables built once per condition. It draws from
# numpy's global generator in the same calls and order, plus one uniform on an undecided
# draw for where the person ends up (_upper_after_deadline).
def _sampler(key):
    if key not in _SAMPLER_CACHE:
        sol = solution(*key)
        upper = sol.pdf("_top")[:-1]  # the last point would land past T_DUR once jittered
        upper[-1] += sol.pdf("_top")[-1]
        lower = sol.pdf("_bottom")[:-1]
        lower[-1] += sol.pdf("_bottom")[-1]
        t = sol.t_domain[:-1]
        shift = np.max(t) + 1  # lower-bound times sit past the upper ones
        domain = np.array(list(t) + list(t + shift) + [-1])
        probs = np.array(list(upper * sol.dt) + list(lower * sol.dt) + [sol.prob_undecided()])
        drift = initiation_drift(key[2], key[3], key[0], key[1])
        _SAMPLER_CACHE[key] = (domain, probs, shift, sol.dt, _upper_after_deadline(sol, drift))
    return _SAMPLER_CACHE[key]


# P(upper first, from 0) for a diffusion that runs on with no deadline.
def _upper_eventually(drift):
    a = 2.0 * drift / SIGMA**2
    if abs(a) < 1e-9:
        return 0.5
    if a > 0:
        return float(-np.expm1(-a * BOUND) / -np.expm1(-2.0 * a * BOUND))
    return float(np.exp(a * BOUND) * -np.expm1(a * BOUND) / -np.expm1(2.0 * a * BOUND))


# P(upper | no bound by T_DUR): where an undecided decision ends up if the person keeps
# deliberating, as a person at a prompt with no deadline does. From
# P(upper eventually) = P(upper by T_DUR) + P(undecided) * P(upper | undecided), with the
# lapse overlay taken back out of the solution (it scales both by 1 - LAPSE_MIX).
def _upper_after_deadline(sol, drift):
    undecided = sol.prob_undecided() / (1 - LAPSE_MIX)
    if undecided < 1e-12:
        return 0.5  # no draw can land here
    upper = (sol.prob("upper") - LAPSE_MIX / 2) / (1 - LAPSE_MIX)
    return float(np.clip((_upper_eventually(drift) - upper) / undecided, 0.0, 1.0))


# (GO, RT, chose) from one draw: GO 1 upper, 0 lower, None undecided by T_DUR. `chose`
# is the choice the person makes: GO when decided, and for an undecided draw the bound
# the diffusion reaches after the deadline. The fit sees only GO and RT.
def _draw(f_val, k_val, value, cost, t0):
    domain, probs, shift, dt, after = _sampler((f_val, k_val, value, cost, t0))
    samp = np.random.choice(domain, p=probs, replace=True, size=1)
    samp = samp[samp != -1]
    samp += np.random.uniform(0, dt, len(samp))  # each point stands for [x, x + dt)
    if len(samp) == 0:
        return None, None, int(np.random.random() < after)
    if samp[0] < shift:
        return 1, float(samp[0]), 1
    return 0, float(samp[0] - shift), 0


# Start the next set: upper = GO (took the effortful option), lower = STAY.
def initiation_decision(value, cost, f_val, k_val, t0=0.0):
    value = snap_value(value)
    cost = snap_cost(cost)
    go, rt, chose = _draw(f_val, k_val, value, cost, t0)

    return {"value": value, "cost": float(cost), "GO": go, "RT": rt, "chose": chose}


# Redo a miss: upper = REPEAT, lower = CONTINUE. The same accumulator as
# initiation_decision, fed different conditions, so both observe one (f, k).
def stage_decision(value, cost, f_val, k_val, t0=0.0):
    record = initiation_decision(value, cost, f_val, k_val, t0=t0)
    record["node"] = "stage"
    return record
