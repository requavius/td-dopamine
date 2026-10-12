"""Seeded runs whose output is pinned in golden.json.

Any change to the model, the controllers or the app's bookkeeping shows up here as a
mismatch. When a change is meant to alter behaviour, re-record and say why in the commit:

    uv run pytest tests --record
"""
import json
import random
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

from temporal.app.web import Root, WebApp
from temporal.controller import CANDIDATES, Controller, Environment, InfoController, measured, run
from temporal.inference import fit, passages_to_frame
from temporal.model.params import UserParams
from temporal.model.persistence import attempts_to_frame, fit_patience, neg_log_likelihood
from temporal.model.questions import Ratings
from temporal.model.simulation import Simulation
from temporal.recovery import simulate_dataset

GOLDEN = Path(__file__).with_name("golden.json")
FIT_TOLERANCE = 1e-3  # pyddm's optimiser stops a few 1e-5 apart from run to run


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _decisions(passages):
    return [[p["GO"], p["RT"], p["value"], p["cost"]] for p in passages]


def _learner(review_mode):
    _seed(1)
    sim = Simulation(UserParams(f=0.8, k=0.8), reentry_cost=0.6, review_mode=review_mode)
    st = sim.run(max_decisions=60)
    return {"starts": _decisions(st.initiation_passages), "redos": _decisions(st.first_passage),
            "theta": st.theta.tolist(), "reward": st.total_reward, "episodes": st.episodes,
            "ability": st.ability}


def _person(sim):
    st = sim.state
    return {"starts": _decisions(st.initiation_passages), "redos": _decisions(st.first_passage),
            "attempts": [[a["p"], a["rt"], a["skipped"]] for a in st.persistence],
            "reward": st.total_reward, "episodes": st.episodes}


def scenario_learner_scheduled():
    return _learner("scheduled")


def scenario_learner_choice():
    return _learner("choice")


def scenario_person_fixed():
    return _person(run(UserParams(f=1.0, k=0.6), env=Environment(), n_decisions=80, seed=2))


def scenario_person_redo_prompts():
    env = Environment(0.45, 0.0, True, False, True)
    return _person(run(UserParams(f=1.0, k=0.6), env=env, n_decisions=80, seed=3))


def scenario_reward_controller_probes():
    ctl = Controller(explore=1.0, seed=4, known=(0.8, 0.8))
    out = _person(run(UserParams(f=0.8, k=0.8), controller=ctl, n_decisions=80, seed=4))
    out["chosen"] = [str(u.chosen) for u in ctl.log]
    return out


def scenario_info_controller():
    # A subset of the settings keeps this fast; every redo-prompt setting stays in.
    ctl = InfoController(candidates=CANDIDATES[::12] + CANDIDATES[-12:], seed=5,
                         known=(1.2, 0.4))
    sim = run(UserParams(f=1.2, k=0.4), controller=ctl, n_decisions=60, seed=5)
    out = _person(sim)
    out["chosen"] = [str(u.chosen) for u in ctl.log]
    out["scores"] = [sorted(u.scores.values())[-3:] for u in ctl.log]
    return out


def scenario_persistence_likelihood():
    sim = run(UserParams(f=0.8, k=0.8, patience=5.0), env=Environment(0.3, 0.25, True),
              n_decisions=80, seed=6)
    attempts = attempts_to_frame(sim.state.persistence)
    return {"nll": [neg_log_likelihood(f, k, attempts, p)
                    for f, k, p in ((0.8, 0.8, 5.0), (1.2, 0.4, 10.0), (0.4, 1.5, 2.0))],
            "patience": fit_patience(0.8, 0.8, attempts), "questions": len(attempts)}


def scenario_fit():
    _seed(7)
    df = simulate_dataset(0.8, 0.6, n_per_cell=100)
    f_hat, k_hat = fit(df)
    _seed(8)
    sim = run(UserParams(f=0.8, k=0.8), env=Environment(), n_decisions=120, seed=8)
    frame = passages_to_frame(sim.state.initiation_passages)
    return {"designed": [f_hat, k_hat], "rows": len(df),
            "undecided": sorted(map(list, df.attrs["undecided"].items())),
            "learner": list(fit(frame)), "learner_rows": len(frame)}


def scenario_measured_state():
    _seed(9)
    sim = run(UserParams(f=0.8, k=0.8), env=Environment(), n_decisions=30, seed=9)
    ratings = Ratings()
    for t, correct in ((0, 1), (3, 0), (5, 1), (5, 0), (8, 0)):
        ratings.update(t, correct)
    st = measured(sim.state, ratings)
    return {"ability": st.ability, "type_diffs": st.type_diffs.tolist()}


# One scripted session of the app, through the same code the browser drives.
def scenario_app_session():
    _seed(10)
    skip = {"time", "seconds_since_launch", "last_active", "gap_seconds"}
    with tempfile.TemporaryDirectory() as d:
        app = WebApp(Root(threading.RLock()), Path(d), panel=True)
        for i in range(40):
            view = app.view["screen"]
            if view == "question":
                if i % 7 == 6:
                    app.on_skip(rt=3.0)
                else:
                    app.on_answer(str(app.answer) if i % 3 else "0", rt=1.5 + i % 4)
            elif view == "decision":
                app.decide(0 if i > 35 else 1, 0.8 + (i % 3) / 2)
            else:
                break
        events = [{k: v for k, v in e.items() if k not in skip} for e in app.store.events()]
        return {"events": events, "ratings": app.ratings.to_dict()}


SCENARIOS = {name[len("scenario_"):]: fn for name, fn in globals().items()
             if name.startswith("scenario_")}
FITTED = {"fit": ("designed", "learner"), "persistence_likelihood": ("patience",)}


def _close(got, want, path, tol):
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path}: keys differ"
        for k in want:
            _close(got[k], want[k], f"{path}.{k}", tol.get(k, 0) if isinstance(tol, dict) else tol)
    elif isinstance(want, list):
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _close(g, w, f"{path}[{i}]", tol)
    elif isinstance(want, float) and not isinstance(got, bool):
        assert got == pytest.approx(want, rel=1e-9, abs=tol or 1e-12), f"{path}: {got} != {want}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario(name, record):
    got = _jsonable(SCENARIOS[name]())
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if record:
        golden[name] = got
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        return
    assert name in golden, f"no golden output for {name}: run with --record"
    tol = {k: FIT_TOLERANCE for k in FITTED.get(name, ())}
    _close(got, golden[name], name, tol)
