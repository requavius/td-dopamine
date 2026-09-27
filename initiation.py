import pyddm

SIGMA = 0.7
BOUND = 1.5
T_DUR = 7.0 # deadline in seconds; past it a decision is undecided. Re-check: `python main.py audit`

DX = 0.005
DT = 0.005

# Conditions are snapped to these steps so repeated draws hit _SOLUTION_CACHE.
VALUE_STEP = 0.05
COST_STEP = 0.05


# One drift form at every node. What the person makes of the set they just played
# reaches it through `value`, as the prediction error carried out of that set
# (config.value_at_choice_point, config.stage_conditions) -- not as terms of its own.
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
        T_dur=T_DUR,
        dx=DX,
        dt=DT,
        parameters=parameters,
        conditions=["value", "cost"],
    )

_MODEL_CACHE = {}
_SOLUTION_CACHE = {}


# `value` carries a prediction error, so it can be negative; round() handles that.
def snap_value(value):
    return round(round(float(value) / VALUE_STEP) * VALUE_STEP, 4)


def snap_cost(cost):
    return round(round(float(cost) / COST_STEP) * COST_STEP, 4)


def clear_caches():
    _MODEL_CACHE.clear()
    _SOLUTION_CACHE.clear()


def _r(x):
    return round(float(x), 6)


def _cached_model(f_val, k_val, t0=0.0):
    key = (_r(f_val), _r(k_val), _r(t0))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_model({"f_val": key[0], "k_val": key[1]}, nondecision=key[2])
    return _MODEL_CACHE[key]


def _cached_solution(f_val, k_val, value, cost, t0=0.0):
    key = (_r(f_val), _r(k_val), value, _r(cost), _r(t0))
    if key not in _SOLUTION_CACHE:
        _SOLUTION_CACHE[key] = _cached_model(f_val, k_val, t0).solve(
            conditions={"value": value, "cost": cost})
    return _SOLUTION_CACHE[key]


# Repeat-or-continue at a failed stage: upper = REPEAT, lower = CONTINUE. The same
# accumulator as initiation_decision, fed different conditions, so both observe one (f, k).
def stage_decision(value, cost, f_val, k_val, t0=0.0):
    record = initiation_decision(value, cost, f_val, k_val, t0=t0)
    record["node"] = "stage"
    if record["GO"] is None:
        record["resolved"] = 0  # ran out of deliberation -> did not repeat
    else:
        record["resolved"] = record["GO"]
    return record


def initiation_decision(value, cost, f_val, k_val, t0=0.0):
    value = snap_value(value)
    cost = snap_cost(cost)
    samp = _cached_solution(f_val, k_val, value, cost, t0).sample(1)

    if len(samp.choice_upper) > 0:
        go, rt = 1, float(samp.choice_upper[0]) # GO: took the effortful option
    elif len(samp.choice_lower) > 0:
        go, rt = 0, float(samp.choice_lower[0]) # STAY
    else:
        go, rt = None, None # neither bound reached by T_DUR

    return {"value": value, "cost": float(cost), "GO": go, "RT": rt}
