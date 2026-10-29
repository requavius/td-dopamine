"""The app's end-of-set screen: a set played with feedback hidden ends on a summary of how
each question went, and with end-of-set redos on each new miss is offered after it, its
outcome shown, as Simulation(end_redos=True, end_feedback=True) does
(tests/test_end_redos.py).
"""
import random
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

from temporal.app.session import rebuild_from_record
from temporal.app.web import Root, WebApp
from temporal.model.accumulator import snap_value
from temporal.model.conditions import carried_delta
from temporal.model.value import ACHIEVEMENT_SCALE


@pytest.fixture
def app():
    random.seed(21)
    np.random.seed(21)
    with tempfile.TemporaryDirectory() as d:
        session = WebApp(Root(threading.RLock()), Path(d))
        yield session
        session.on_close()  # before the folder goes, or the exit hook writes into nothing


# The first set (feedback on) answered right, then the next one hidden, every answer wrong.
def _hidden_set_all_wrong(app, end_redos):
    app.v_review.set(0.0)  # every slot a new question
    app.v_feedback.set(False)
    app.v_end_redos.set(end_redos)
    for _ in range(app.state.stage_amt):
        app.on_answer(str(app.answer), rt=2.5)
    assert app.view["prompt"] == "start the next set?"
    app.decide(1, 1.0)
    for _ in range(app.state.stage_amt):
        assert app.view["screen"] == "question"
        app.on_answer("x", rt=3.0)
    return app.payload()


def _events(app, node):
    return [e for e in app.store.events() if e.get("node") == node]


def test_feedback_shown_has_no_score_screen(app):
    for _ in range(app.state.stage_amt):
        app.on_answer(str(app.answer), rt=2.5)
    assert app.view["prompt"] == "start the next set?"
    assert not _events(app, "score")


def test_hidden_set_ends_on_a_summary(app):
    out = _hidden_set_all_wrong(app, end_redos=False)
    (score,) = _events(app, "score")
    # the set's points (none: all wrong) and the session's score, which is the TD reward
    assert score["points"] == 0.0 and score["score"] == pytest.approx(app.score)
    view = out["view"]
    assert view["screen"] == "summary" and view["total"] == f"+0.00 this set, score {app.score:.2f}"
    answers = [e for e in _events(app, "answer") if e["set"] == score["set"]]
    assert [r["text"] for r in view["rows"]] == [
        f"question {i}: {a['question']} = x  incorrect" for i, a in enumerate(answers, start=1)]
    assert {r["outcome"] for r in view["rows"]} == {"incorrect"}
    app.proceed()
    assert app.view["prompt"] == "start the next set?"
    assert score["offered"] == 0 and score["passed"] == 0
    assert not any(e.get("end") for e in _events(app, "stage"))


def test_the_summary_gives_each_right_answer_its_points(app):
    app.v_review.set(0.0)
    app.v_feedback.set(False)
    for _ in range(app.state.stage_amt):
        app.on_answer(str(app.answer), rt=2.5)
    app.decide(1, 1.0)
    for i in range(app.state.stage_amt):
        app.on_answer(str(app.answer) if i % 2 else "x", rt=2.5)
    rows, n = app.view["rows"], app.state.stage_amt
    answers = [e for e in _events(app, "answer")][-n:]
    for row, a in zip(rows, answers):
        if a["correct"]:
            points = ACHIEVEMENT_SCALE * (1 - a["p"]) / n
            assert row["outcome"] == "correct" and row["text"].endswith(f"correct  +{points:.2f}")
        else:
            assert row["outcome"] == "incorrect" and row["text"].endswith("incorrect")
    (score,) = _events(app, "score")
    # the points listed add up to what the set paid
    assert score["points"] == pytest.approx(
        sum(ACHIEVEMENT_SCALE * (1 - a["p"]) / n for a in answers if a["correct"]))


def test_end_redos_follow_the_summary_as_in_the_simulator(app):
    _hidden_set_all_wrong(app, end_redos=True)
    n = app.state.stage_amt
    assert app.view["screen"] == "summary"
    (score,) = _events(app, "score")
    assert score["offered"] == n
    set_delta = score["set_delta"]
    app.proceed()

    # One prompt per miss: take the first (answered right, quickly), pass on the second,
    # take the third and give up on it, pass on the fourth.
    asked, score_before = [], app.score
    for i in range(n):
        assert app.view["screen"] == "decision"
        asked.append(app.problem[0])
        assert app.view["prompt"] == f"redo question {i + 1}: {app.problem[0]}?"
        app.decide(1 if i in (0, 2) else 0, 1.0)
        if i == 0:
            assert app.view["header"].startswith(f"end of set: question {i + 1}")
            app.on_answer(str(app.answer), rt=0.5)
            earned = ACHIEVEMENT_SCALE * (1 - _events(app, "answer")[-1]["p"]) / n
            # its outcome and points are shown, and the points land on the score
            assert app.payload()["flashes"] == [{"text": f"correct +{earned:.2f}",
                                                 "kind": "correct"}]
            assert app.score == pytest.approx(score_before + earned)
        elif i == 2:
            app.on_skip(rt=2.0)
    assert app.view["prompt"] == "start the next set?"

    redos = [e for e in _events(app, "stage") if e.get("end")]
    assert [r["question"] for r in redos] == asked
    # the first prompt carries the set's error; after the right redo, that plus what landed
    carried = [set_delta] + [set_delta + earned] * (n - 1)
    for r, c in zip(redos, carried):
        # the same node at the last slot: undiscounted share, the error carried
        assert r["stage"] == n and r["slot_delta"] == pytest.approx(c)
        assert r["value"] == snap_value(1.0 / n + carried_delta(r["slot_delta"]))

    after = [e for e in app.store.events() if e.get("node") in ("answer", "skip")
             and e.get("slot") == n]
    assert len(after) == 2 and all(e["redo"] and e["review"] for e in after)
    assert after[0]["delta"] == set_delta and after[1]["delta"] == pytest.approx(set_delta + earned)
    assert after[0]["correct"] and not after[0]["slip"]  # never a typo after the score

    end = _events(app, "set_end")[-1]
    assert end["passed"] == 0.0  # the redo after the score is not in the set's score
    assert end["review_load"] == 2 / n  # both redos taken add to it, as in the simulator
    assert end["set_delta"] == pytest.approx(set_delta + earned)  # the redo's reward landed
    assert app.state.corrections == 1
    assert sum(len(qs) for qs in app.owed.values()) == n - 1

    ratings, _ = rebuild_from_record(app.store.events())
    assert ratings.to_dict() == app.ratings.to_dict()


def test_the_setting_maps_to_an_end_of_set_environment(app):
    app.v_feedback.set(False)
    app.v_end_redos.set(True)
    env = app.current_environment()
    assert env.end_redos and env.end_feedback  # the app shows how each redo went
    app.v_feedback.set(True)  # with feedback shown the miss is seen at once
    assert not app.current_environment().end_redos
