"""The closed-form likelihood against pyddm's Fokker-Planck solution."""
import numpy as np
import pyddm
import pytest

from temporal import fastlik
from temporal.model import accumulator as acc


@pytest.mark.parametrize("bound,t0", [(1.0, 0.0), (1.5, 0.23), (2.4, 0.1)])
def test_matches_pyddm(bound, t0):
    model = pyddm.gddm(drift=lambda mu: mu, noise=acc.SIGMA, bound=bound, starting_position=0,
                       nondecision=t0, mixture_coef=acc.LAPSE_MIX, T_dur=acc.T_DUR,
                       dx=acc.DX, dt=acc.DT, parameters={}, conditions=["mu"])
    for mu in (-2.5, -0.3, 0.0, 0.6, 3.0):
        sol = model.solve(conditions={"mu": mu})
        t = sol.t_domain
        away = t > t0 + 0.05  # pyddm shifts by whole time steps
        for upper, name in ((True, "upper"), (False, "lower")):
            mine = ((1 - acc.LAPSE_MIX) * fastlik.passage_density(t - t0, np.full(t.shape, mu), upper, bound)
                    + 0.5 * acc.LAPSE_MIX / acc.T_DUR)
            peak = sol.pdf(name)[away].max()
            assert np.max(np.abs(mine[away] - sol.pdf(name)[away])) < 2e-3 * peak
        undecided = (1 - acc.LAPSE_MIX) * fastlik.survival(acc.T_DUR - t0, mu, bound)[0]
        assert undecided == pytest.approx(sol.prob_undecided(), abs=1e-3)
