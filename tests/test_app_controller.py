"""The app's controller (controller.AppController) and the model settings it chooses."""
import io
from contextlib import redirect_stderr

import numpy as np
import pytest

from temporal.controller import (APP_CANDIDATES, AppController, Environment, fisher, measured,
                                 run)
from temporal.inference import fit_closed_form
from temporal.model.params import UserParams


def test_every_setting_once():
    assert len(APP_CANDIDATES) == 1104 == len(set(APP_CANDIDATES))
    assert {e.stage_amt for e in APP_CANDIDATES} == {2, 4, 6, 8}
    assert {e.countdown for e in APP_CANDIDATES if e.pressure} == {2.0, 3.0, 5.0}
    # a countdown only on redo prompts, an end-of-set redo only with feedback hidden
    assert all(e.redo_prompts for e in APP_CANDIDATES if e.pressure)
    assert all(not e.feedback for e in APP_CANDIDATES if e.end_redos)


def test_forced_repeats_redo_every_seen_miss_undecided():
    sim = run(UserParams(f=0.8, k=0.8), env=Environment(0.3, 0.0, forced=True),
              n_decisions=30, seed=5)
    assert sim.state.first_passage == []  # no redo decision is ever recorded
    assert sim.state.corrections > 0      # but misses were redone


def test_set_length_and_score_reach_the_simulator():
    sim = run(UserParams(f=0.8, k=0.8), env=Environment(stage_amt=6, show_score=False),
              n_decisions=10, seed=5)
    assert sim.state.stage_amt == 6 and not sim.show_score
    assert {a["stage_amt"] for a in sim.state.persistence} == {6}


def test_closed_form_fit_recovers_the_countdown_response():
    env = Environment(0.45, 0.0, True, False, True, pressure=True, countdown=3.0)
    sim = run(UserParams(f=0.9, k=0.7, pressure=0.5), env=env, n_decisions=150, seed=6)
    f, k, t0, lam = fit_closed_form(sim.state.initiation_passages + sim.state.first_passage)
    assert abs(lam - 0.5) < 0.1 and abs(f - 0.9) < 0.2 and abs(k - 0.7) < 0.2


def test_untimed_information_is_info_controllers():
    sim = run(UserParams(f=0.8, k=0.8), env=Environment(0.45, 0.0, True, False, True),
              n_decisions=30, seed=7)
    ps = sim.state.initiation_passages + sim.state.first_passage
    m = AppController().information(ps, 0.8, 0.8, 0.7)
    assert np.allclose(m[:2, :2], fisher(ps, 0.8, 0.8)) and not m[:2, 2].any()


def test_a_plan_chooses_one_of_its_settings():
    env = Environment(0.45, 0.0, True, False, True, pressure=True, countdown=2.0)
    sim = run(UserParams(f=0.8, k=0.8, pressure=0.6), env=env, n_decisions=40, seed=8)
    some = APP_CANDIDATES[::23] + APP_CANDIDATES[15:23]  # one of each kind of setting
    ctl = AppController(candidates=some, finalists=4, seed=8, known=(0.8, 0.8, 0.6))
    with redirect_stderr(io.StringIO()):
        update = ctl.plan(measured(sim.state, sim.ratings) if sim.ratings else sim.state,
                          UserParams(f=0.0, k=0.0))
    assert update.chosen in some and len(update.scores) == 4
    assert update.reason.endswith(f"from {len(some)} settings")
    assert update.estimate.lam == pytest.approx(0.6)
