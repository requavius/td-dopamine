"""Confidence (model/confidence.py): the rules, and that eta = 0 is no change at all."""
import random

import numpy as np

from temporal.controller import Environment, current_confidence, measured, run
from temporal.model import confidence as conf
from temporal.model.params import UserParams
from temporal.model.questions import Ratings
from temporal.model.simulation import Simulation


def test_perceived_is_identity_at_zero():
    for p in (0.001, 0.3, 0.75, 0.999):
        assert conf.perceived(p, 0.0) == p


def test_rules():
    assert conf.update(0.0, 0, 0.5, 0.0) == 0.0  # no dynamics
    assert conf.update(0.0, 1, 0.9, 1.0, "elo") == 1.0 * (1 - 0.9)  # expected win: small
    assert conf.update(0.0, 0, 0.9, 1.0, "elo") == -0.9            # surprising loss: large
    assert conf.update(0.0, 1, 0.9, 1.0, "count", 0.75) == (0.25) * (1 - conf.REVERT)
    assert conf.update(0.0, 0, 0.5, 1.0, "asym") == -1.0           # loss counts twice
    assert conf.update(2.9, 1, 0.1, 5.0, "elo") == conf.C_CLIP


def test_trajectory_is_before_each_question():
    before, now = conf.trajectory([(0.5, 0), (0.5, 1)], ("elo", 1.0, 0.75))
    assert before[0] == 0.0 and before[1] == -0.5
    assert now == before[1] + (1 - conf.perceived(0.5, -0.5))


def test_eta_zero_keeps_confidence_at_zero():
    sim = run(UserParams(f=0.8, k=0.8), env=Environment(), n_decisions=40, seed=1)
    assert sim.state.confidence == 0.0
    assert all(a["outcome"] in (0, 1) for a in sim.state.persistence)


def test_confidence_moves_and_costs_give_ups():
    env = Environment(0.45, 0.25, True)
    base = run(UserParams(f=0.8, k=0.8), env=env, n_decisions=60, seed=2).state
    low = run(UserParams(f=0.8, k=0.8, eta=1.0, conf_rule="count", conf_ref=0.9),
              env=env, n_decisions=60, seed=2).state
    assert low.confidence < 0  # missing at 45% against a 90% reference
    assert low.skips / len(low.persistence) > base.skips / len(base.persistence)


def test_controller_cannot_see_confidence():
    sim = run(UserParams(f=0.8, k=0.8, eta=1.0, conf_rule="count", conf_ref=0.9),
              env=Environment(0.45), n_decisions=40, seed=3)
    assert sim.state.confidence != 0.0
    assert measured(sim.state, Ratings()).confidence == 0.0


def test_current_confidence_replays_the_log():
    # Without ratings the log carries the true chance, so replaying it under the true
    # rule rebuilds confidence exactly. (With ratings it carries the environment's
    # estimate, and the replay is what the environment can know.)
    random.seed(4)
    np.random.seed(4)
    sim = Simulation(UserParams(f=0.8, k=0.8, eta=0.5, conf_rule="elo"), skips=True)
    sim.run(max_decisions=30)
    (_, _, c), = current_confidence(sim.state, [(("elo", 0.5, 0.75), 1.0, 0.0)])
    assert sim.state.confidence != 0.0 and c == sim.state.confidence
