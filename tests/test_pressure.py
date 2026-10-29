"""Time pressure on the redo prompt (model/pressure.py). Off by default."""
import numpy as np

from temporal import fastlik
from temporal.controller import Environment, run
from temporal.model import accumulator as acc
from temporal.model.params import UserParams
from temporal.model.pressure import REDO_DEADLINE, pressured_decision


def test_no_countdown_by_default():
    sim = run(UserParams(f=0.8, k=0.6), env=Environment(0.45, 0.0, True, False, True),
              n_decisions=40, seed=3)
    assert sim.state.first_passage and not any("deadline" in r for r in sim.state.first_passage)


def test_a_countdown_closes_the_prompt():
    sim = run(UserParams(f=0.8, k=0.6, pressure=0.6),
              env=Environment(0.45, 0.0, True, False, True, pressure=True), n_decisions=60, seed=3)
    redos = sim.state.first_passage
    assert redos
    for r in redos:
        assert r["deadline"] == REDO_DEADLINE
        if r["GO"] is None:
            assert r["chose"] == 0  # undecided at the deadline: they move on
        else:
            assert r["RT"] < REDO_DEADLINE and r["chose"] == r["GO"]


def test_draws_follow_the_closed_form():
    np.random.seed(0)
    f, k, value, cost, scale = 0.8, 0.8, 0.4, 0.45, 0.7
    draws = [pressured_decision(value, cost, f, k, scale=scale) for _ in range(4000)]
    up, lo, und = fastlik.outcome_probs([f * value - k * cost], acc.BOUND * scale,
                                        t_dur=REDO_DEADLINE, dt=acc.DT)
    assert abs(np.mean([d["GO"] == 1 for d in draws]) - up.sum()) < 0.03
    assert abs(np.mean([d["GO"] is None for d in draws]) - und[0]) < 0.03


def test_a_lower_bound_decides_sooner():
    def mean_rt(scale):
        np.random.seed(1)
        rts = [pressured_decision(0.3, 0.3, 0.8, 0.8, scale=scale)["RT"] for _ in range(2000)]
        return np.mean([t for t in rts if t is not None])
    assert mean_rt(0.5) < mean_rt(1.0)
