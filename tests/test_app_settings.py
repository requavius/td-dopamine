"""The app's score, its redo countdown, and the settings its controller sees."""
import random
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

from temporal.app.web import Root, WebApp
from temporal.controller import AppController


@pytest.fixture
def app():
    random.seed(31)
    np.random.seed(31)
    with tempfile.TemporaryDirectory() as d:
        session = WebApp(Root(threading.RLock()), Path(d))
        yield session
        session.on_close()  # before the folder goes, or the exit hook writes into nothing


def _right_set(app):
    for _ in range(app.state.stage_amt):
        app.on_answer(str(app.answer), rt=2.5)


def test_the_score_is_the_td_reward_and_lasts_the_session(app):
    assert app.view["header"].endswith("score 0.00")
    _right_set(app)
    first = app.score
    assert first > 0  # four right: each landed as 4 (1 - p) / 4
    app.decide(1, 1.0)
    assert app.view["header"].endswith(f"score {first:.2f}")  # carried into the next set
    _right_set(app)
    assert app.score > first


def test_a_hidden_score_pays_the_fraction_right(app):
    app.v_show_score.set(False)
    _right_set(app)  # the first set started with the score shown
    shown = app.score
    app.decide(1, 1.0)
    assert "score" not in app.view["header"]
    _right_set(app)
    assert app.score - shown == pytest.approx(1.0)  # 4 right of 4, 1/4 each


def test_a_hidden_score_with_hidden_feedback_ends_on_how_many_were_right(app):
    _right_set(app)
    app.v_show_score.set(False)
    app.v_feedback.set(False)
    app.decide(1, 1.0)
    _right_set(app)
    n = app.state.stage_amt
    assert app.view["total"] == f"{n} of {n} right"
    assert all(r["text"].endswith("correct") for r in app.view["rows"])  # no points hidden


def _miss_with_countdown(app, seconds=3.0):
    app.v_countdown.set(seconds)
    app.v_review.set(0.0)
    while app.view["screen"] == "question":
        app.on_answer("x", rt=2.0)  # feedback on and redo prompts on: a timed prompt
        if app.view["screen"] == "decision" and app.view.get("deadline"):
            return
    raise AssertionError("no timed redo prompt")


def test_a_countdown_that_runs_out_closes_the_prompt(app):
    _miss_with_countdown(app)
    assert app.view["deadline"] == 3.0
    app.decide(None, 3.0)
    (rec,) = [e for e in app.store.events() if e["node"] == "stage"]
    assert rec["GO"] is None and rec["RT"] is None and rec["deadline"] == 3.0
    assert app.view["screen"] in ("question", "decision") and not app.redo  # moved on


def test_a_click_after_the_countdown_counts_as_running_out(app):
    _miss_with_countdown(app, 2.0)
    app.decide(1, 2.4)  # the page's click raced the countdown and lost
    (rec,) = [e for e in app.store.events() if e["node"] == "stage"]
    assert rec["GO"] is None and app.decision_now["decided"]


def test_a_click_in_time_is_a_timed_decision(app):
    _miss_with_countdown(app, 5.0)
    app.decide(1, 1.2)
    (rec,) = [e for e in app.store.events() if e["node"] == "stage"]
    assert rec["GO"] == 1 and rec["RT"] == 1.2 and rec["deadline"] == 5.0
    assert app.redo  # the question is back on screen


def test_settings_map_to_the_controllers(app):
    app.v_countdown.set(2.0)
    env = app.current_environment()
    assert env.redo_prompts and env.pressure and env.countdown == 2.0
    app.v_show_choice.set(False)
    app.v_forced.set("repeat")
    app.v_stages.set(6)
    app.v_show_score.set(False)
    env = app.current_environment()
    assert env.forced and env.stage_amt == 6 and not env.show_score and not env.pressure


def test_the_controller_status_says_what_it_is_doing(app):
    assert app.controller_text() == "off"
    app.v_controller.set(True)
    assert app.controller_text() == "on, first plan when the next set starts"
    # A saved controller that planned one start decision ago waits four more.
    app.controller = AppController(min_spacing=5)
    app.controller.last_plan = len(app.state.initiation_passages) - 1
    assert app.controller_text() == "on, next plan in 4 sets"
    app.controller.last_plan -= 4
    assert app.controller_text() == "on, plans when the next set starts"
