"""The simulator records only what the app could: one stop per session, no cold starts.

Until 2026-10-25 it recorded up to three stops in a row inside a session and every cold
start after it, none of which a person in the app can produce (RETIRED.md, V3).
"""
import random

import numpy as np
import pytest

from temporal.controller import Environment, run
from temporal.model.params import UserParams
from temporal.model.simulation import Simulation


def _check(starts):
    assert starts, "nothing recorded"
    assert all(p["in_session"] for p in starts), "a cold start was recorded"
    ids = [p["episode_id"] for p in starts]
    # A stop ends the session, so the next recorded decision comes after a set played
    # on coming back: never two recorded start decisions with no set between them.
    assert all(b > a for a, b in zip(ids, ids[1:])), "two start decisions with no set between"


@pytest.mark.parametrize("review_mode", ["scheduled", "choice"])
def test_learner_one_stop_per_session(review_mode):
    random.seed(11)
    np.random.seed(11)
    sim = Simulation(UserParams(f=0.6, k=1.2), reentry_cost=0.6, review_mode=review_mode)
    st = sim.run(max_decisions=200)
    _check(st.initiation_passages)
    assert st.stuck > 0 and st.away > 0  # it did stop, and was away


def test_person_one_stop_per_session():
    # Skips on: leaving at the first question of a cold set is another way to end one.
    sim = run(UserParams(f=0.6, k=1.2), env=Environment(), n_ticks=300, seed=12)
    _check(sim.state.initiation_passages)
    assert sim.state.t == 301  # n_ticks ticks after the free first set
