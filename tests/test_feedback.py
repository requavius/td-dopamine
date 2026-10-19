"""Feedback decided per question, what a hidden outcome is worth, and leaving from a question."""
import math
import random
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from temporal.app.web import Root, WebApp
from temporal.controller import Environment, run
from temporal.model import feedback as fb
from temporal.model.params import THETA_START, ModelState, UserParams
from temporal.model.persistence import attempts_to_frame, neg_log_likelihood
from temporal.model.simulation import Simulation
from temporal.model.value import ACHIEVEMENT_SCALE, V, slot_reward


def test_true_and_false_are_on_and_off():
    assert fb.policy(True) == "on" and fb.policy(False) == "off"
    assert Environment(0.75, 0.25, False).feedback == "off"
    with pytest.raises(ValueError):
        fb.policy("sometimes")


# Hiding moves when a set's reward lands, never how much: the slots sum to its achievement.
@pytest.mark.parametrize("seed", range(5))
def test_a_set_totals_the_same_shown_or_hidden(seed):
    rng = random.Random(seed)
    st = ModelState(t=1, stage_amt=5)
    outcomes = [(rng.random() < 0.5, rng.random()) for _ in range(st.stage_amt)]
    totals = []
    for hide in ([False] * 5, [True] * 5, [rng.random() < 0.5 for _ in range(5)]):
        achievement = credited = paid = 0.0
        for s, ((right, p), hidden) in enumerate(zip(outcomes, hide)):
            worth = (1 - p) if right else 0.0
            achievement += worth
            reward, credited = slot_reward(st, worth, not hidden, achievement, credited,
                                           s == st.stage_amt - 1)
            paid += reward
        totals.append(paid)
    assert totals == pytest.approx([ACHIEVEMENT_SCALE * achievement / st.stage_amt] * 3)


# Nothing pending, nothing changed: V is position alone and the hidden weights never move.
def test_with_nothing_hidden_the_hidden_weights_never_move():
    random.seed(0)
    np.random.seed(0)
    sim = Simulation(UserParams(f=0.8, k=0.8), feedback="on")
    sim.run(max_decisions=40)
    assert sim.state.theta[2:].tolist() == list(THETA_START[2:])


# What one pending outcome at a 50% chance is worth after `n` start decisions, for a
# person on question types (chances vary, so value sees pending outcomes across them).
# `misses` hides every miss before the last slot and shows every right; otherwise
# everything is hidden.
def _pending_worth(misses, n=200, seeds=4):
    out = []
    for seed in range(seeds):
        st = run(UserParams(f=0.8, k=0.8), env=Environment(0.5, 0.25, "guard" if misses else "off"),
                 n_decisions=n, seed=seed).state
        out.append(V(st, 1, [0.5]) - V(st, 1, []))
    return float(np.mean(out))


# Hide only misses and pending comes to be worth nothing; hide everything and it is worth
# about an outcome at its chance. Learned by TD, at the value's own rate.
def test_hiding_only_misses_is_learned_and_hiding_everything_is_not(monkeypatch):
    real = fb.shows

    def every_miss(state, params, pol, s, success, p, target_p, guard):
        if pol == "guard":
            return success or s == state.stage_amt - 1
        return real(state, params, pol, s, success, p, target_p, guard)

    monkeypatch.setattr(fb, "shows", every_miss)
    # From a start of 0.25 (worth its chance, params.THETA_START), at the value's own rate.
    assert _pending_worth(misses=True) < 0.1
    assert _pending_worth(misses=False) > 0.2


def _fragile():
    # A person who values the task: one more seen miss would make leaving likely.
    st = ModelState(t=1, stage_amt=4)
    st.theta = np.array([0.6, 0.0, 0.0, 0.0])
    return st


# The guard hides a miss only while hiding is worth something: with pending worth nothing
# (its weights at 0, as hiding only misses teaches), showing and hiding come out equal.
def test_the_guard_stops_hiding_once_hiding_reads_as_a_miss():
    st, params = _fragile(), UserParams(f=1.2, k=0.8)
    guard = (1.2, 0.8, 3.0)
    assert not fb.guard_hides(st, params, 0, 0.3, 0.3, guard)
    st.theta = np.array([0.6, 0.0, 0.0, 1.0])  # pending worth an outcome at its chance
    assert fb.guard_hides(st, params, 0, 0.3, 0.3, guard)


def test_cover_hides_rights_at_the_rate_misses_were_hidden():
    st, params = _fragile(), UserParams(f=1.2, k=0.8)
    st.miss_hide_rate = 0.25
    shown = [fb.shows(st, params, "guard+cover", 0, True, 0.5, 0.3, (1.2, 0.8, 3.0))
             for _ in range(400)]
    assert shown.count(False) == 100
    assert fb.shows(st, params, "guard", 0, True, 0.5, 0.3, (1.2, 0.8, 3.0))
    assert not fb.shows(st, params, "off", st.stage_amt - 1, True, 0.5, 0.3, (1.2, 0.8, 3.0))


# Leaving abandons the set partway, and the next start decision is cold.
def test_a_simulated_leave_ends_the_session():
    sim = run(UserParams(f=0.8, k=0.8, patience=2.0), env=Environment(0.3, 0.25, "on"),
              n_decisions=150, seed=1)
    st = sim.state
    left = [i for i, a in enumerate(st.persistence) if a.get("left")]
    assert left and st.abandoned >= len(left)
    assert all(st.persistence[i]["leave_v"] is not None for i in left)
    assert all(not st.persistence[i]["skipped"] for i in left)


def _frame(n=600, seed=0, leave_v=True):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rec = {"p": float(rng.uniform(0.2, 0.9)), "rt": float(rng.uniform(1, 20)),
               "skipped": bool(rng.random() < 0.1), "stage_amt": 4,
               "delta": float(rng.normal(0, 0.1))}
        if leave_v:
            rec.update(leave_v=float(rng.uniform(0.2, 1.0)), leave_cost=0.2,
                       left=bool(not rec["skipped"] and rng.random() < 0.05))
        rows.append(rec)
    return attempts_to_frame(rows)


# A record from before leaving existed is scored exactly as it was: no leave term.
def test_records_without_leave_conditions_keep_the_old_likelihood():
    frame = _frame(leave_v=False)
    assert not frame["left"].any() and frame["leave_value"].isna().all()
    old = frame[["value", "cost", "skipped", "seconds"]]
    assert neg_log_likelihood(0.8, 0.8, frame, 5.0) == pytest.approx(
        neg_log_likelihood(0.8, 0.8, old, 5.0))
    # And an old abandon from a question still counts as giving up on it.
    gone = attempts_to_frame([{"p": 0.5, "rt": 3.0, "skipped": True, "left": True}])
    assert bool(gone["skipped"].iloc[0]) and not bool(gone["left"].iloc[0])


def test_leave_terms_change_the_likelihood_and_track_value():
    frame = _frame()
    plain = frame.assign(leave_value=np.nan, left=False,
                         skipped=frame["skipped"] | frame["left"])
    assert neg_log_likelihood(0.8, 0.8, frame, 5.0) != neg_log_likelihood(0.8, 0.8, plain, 5.0)
    # Leaving when staying was worth more is less likely: the leave density falls with value.
    one = frame.iloc[[0]].assign(left=True, skipped=False)
    lo, hi = one.assign(leave_value=0.1), one.assign(leave_value=1.0)
    assert neg_log_likelihood(0.8, 0.8, hi, 5.0) > neg_log_likelihood(0.8, 0.8, lo, 5.0)


def test_the_app_logs_a_leave_from_a_question():
    random.seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as d:
        app = WebApp(Root(threading.RLock()), Path(d))
        app.on_answer(str(app.answer), rt=2.0)
        app.on_leave(rt=3.5)
        events = app.store.events()
        leave = next(e for e in events if e["node"] == "leave")
        assert leave["rt"] == 3.5 and leave["left"] and leave["leave_v"] is not None
        assert events[-1]["node"] == "set_end" and events[-1]["left"]
        frame = attempts_to_frame(app.state.persistence)
        assert bool(frame["left"].iloc[-1]) and math.isfinite(frame["leave_value"].iloc[-1])
        again = WebApp(Root(threading.RLock()), Path(d))  # closed cleanly: no unlogged exit
        assert not any(e.get("unlogged") for e in again.store.events())
        again.on_close()


def test_feedback_off_tells_the_score_at_the_next_start_prompt():
    random.seed(1)
    np.random.seed(1)
    with tempfile.TemporaryDirectory() as d:
        app = WebApp(Root(threading.RLock()), Path(d))
        app.v_feedback.set("off")
        app.end_episode(abandoned=False)  # the first set began under the default
        app.decide(1, 1.0)
        while app.view["screen"] == "question":
            app.on_answer(str(app.answer), rt=2.0)
        assert app.view["prompt"].startswith("this set: ")
        assert not app.flashes  # nothing was marked as it was answered
        ends = pd.DataFrame([e for e in app.store.events() if e["node"] == "set_end"])
        assert ends["hidden"].iloc[-1] == app.state.stage_amt
        app.on_close()
