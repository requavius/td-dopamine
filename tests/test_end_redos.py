"""End-of-set redos: with feedback hidden, the redo prompts move to the end-of-set screen.

Off by default, so every other seeded output is unchanged (tests/test_regression.py).
"""
from dataclasses import replace

import numpy as np

from temporal.controller import Environment, run
from temporal.model.accumulator import snap_value
from temporal.model.conditions import carried_delta
from temporal.model.params import UserParams


def _redos(env, seed=3):
    sim = run(UserParams(f=0.8, k=0.6), env=env, n_decisions=60, seed=seed)
    return sim.state.first_passage, sim.state.stage_amt


def test_hidden_feedback_without_end_redos_asks_nothing():
    redos, _ = _redos(Environment(0.45, 0.0, False, False, True))
    assert redos == []


def test_end_redos_fire_after_the_set_with_the_sets_error():
    redos, n = _redos(Environment(0.45, 0.0, False, False, True, True))
    assert redos, "a hidden-feedback set at a 45% target should have misses to redo"
    for r in redos:
        assert r["end"] and r["stage"] == n
        # the slot share is undiscounted; the error carried is the whole set's
        assert r["value"] == snap_value(1.0 / n + carried_delta(r["slot_delta"]))


def test_feedback_shown_keeps_redos_mid_set():
    redos, _ = _redos(Environment(0.45, 0.0, True, False, True, True))
    assert redos and not any(r.get("end") for r in redos)


def test_a_shown_end_redo_lands_what_it_earns():
    env = Environment(0.45, 0.0, False, False, True, True)
    hidden = run(UserParams(f=0.8, k=0.6), env=env, n_decisions=60, seed=3)
    shown = run(UserParams(f=0.8, k=0.6), env=replace(env, end_feedback=True),
                n_decisions=60, seed=3)
    # the same draws until a redo is answered right, after which what it earned has moved
    # the value weights
    assert hidden.state.first_passage[0] == shown.state.first_passage[0]
    assert not np.allclose(hidden.state.theta, shown.state.theta)
