import pyddm

SIGMA = 0.7
BOUND = 1.5
T_DUR = 3.0

DX = 0.005
DT = 0.005

VALUE_STEP = 0.05


def initiation_drift(value, cost, f_val, k_val):
    return f_val * value - k_val * cost


def build_model(parameters):
    return pyddm.gddm(
        drift=initiation_drift,
        noise=SIGMA,
        bound=BOUND,
        starting_position=0,
        T_dur=T_DUR,
        dx=DX,
        dt=DT,
        parameters=parameters,
        conditions=["value", "cost"],
    )

_MODEL_CACHE = {}
_SOLUTION_CACHE = {}


def snap_value(value):
    return round(round(float(value) / VALUE_STEP) * VALUE_STEP, 4)


def clear_caches():
    _MODEL_CACHE.clear()
    _SOLUTION_CACHE.clear()


def _cached_model(f_val, k_val):
    key = (round(float(f_val), 6), round(float(k_val), 6))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_model({"f_val": key[0], "k_val": key[1]})
    return _MODEL_CACHE[key]


def _cached_solution(f_val, k_val, value, cost):
    key = (round(float(f_val), 6), round(float(k_val), 6), value, round(float(cost), 6))
    if key not in _SOLUTION_CACHE:
        model = _cached_model(f_val, k_val)
        _SOLUTION_CACHE[key] = model.solve(conditions={"value": value, "cost": cost})
    return _SOLUTION_CACHE[key]


def initiation_decision(value, cost, f_val, k_val):
    value = snap_value(value)
    samp = _cached_solution(f_val, k_val, value, cost).sample(1)

    if len(samp.choice_upper) > 0:
        go, rt = 1, float(samp.choice_upper[0])
    elif len(samp.choice_lower) > 0:
        go, rt = 0, float(samp.choice_lower[0])
    else:
        go, rt = None, None

    return {"value": value, "cost": float(cost), "GO": go, "RT": rt}
