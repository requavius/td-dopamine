"""Competence: the ceiling keeps a simulated person inside the question bank."""
import math

import numpy as np

from temporal.controller import Environment, run
from temporal.model import simulation
from temporal.model.competence import (CORRECTION_GAIN, ability_ceiling, correction_gain,
                                       sigmoid, success_prob)
from temporal.model.params import ModelState, UserParams


def test_no_ceiling_is_the_old_gain():
    assert correction_gain(ModelState(t=1, ability=50.0)) == CORRECTION_GAIN


def test_gain_shrinks_to_nothing_at_the_ceiling():
    gains = [correction_gain(ModelState(t=1, ability=a, ability_ceiling=2.0))
             for a in (0.0, 1.0, 1.9, 2.0, 3.0)]
    assert gains == sorted(gains, reverse=True) and gains[-2:] == [0.0, 0.0]
    assert gains[0] < CORRECTION_GAIN


def test_ceiling_leaves_the_hardest_type_at_twenty_percent():
    assert math.isclose(sigmoid(ability_ceiling([-2.0, 0.5, 3.0]) - 3.0), 0.2)


# Over 1000 start decisions the chance served stays near the target; without the ceiling
# the person outgrows the bank and every question is a near-certain right answer.
def _served(env, ceiling):
    served = []
    new_item = simulation.Simulation._new_item

    def spy(self, s):
        item = new_item(self, s)
        served.append(success_prob(self.state, item))
        return item

    simulation.Simulation._new_item = spy
    try:
        sim = run(UserParams(f=0.8, k=0.8), env=env, n_decisions=1000, seed=0, ceiling=ceiling)
    finally:
        simulation.Simulation._new_item = new_item
    return sim.state, np.mean(served[len(served) // 2:])


def test_served_chance_stays_near_target():
    st, late = _served(Environment(0.5, 0.25, "off"), True)
    assert st.ability <= st.ability_ceiling
    assert abs(late - 0.5) < 0.1


def test_without_a_ceiling_the_person_outgrows_the_bank():
    st, late = _served(Environment(0.5, 0.25, "off"), False)
    assert st.ability > max(st.type_diffs) + 2 and late > 0.95
