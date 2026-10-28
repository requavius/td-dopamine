"""The α replay (experiments.recovery_alpha) against the shadow learners it replaces."""
import numpy as np

from experiments.common import Quiet
from experiments.gamma_profile import run_app
from experiments.recovery_alpha import SET_ABANDONED, EventLog, replay, value_of

GRID = (0.5, 0.7, 0.9, 0.99)


# A controller-free run with feedback switched every 10 starts, so the log holds redos,
# sets played with feedback hidden, and sets abandoned on a cold first question.
def test_replay_matches_shadows():
    sims = []
    with Quiet():
        data = run_app(0.3, 1.4, 0.8, seed=2, n=150, blocks=10, grid=GRID,
                       cls=EventLog, sim_out=sims)
    kind, slot, reward = sims[0].arrays()
    assert (kind == SET_ABANDONED).any()
    assert any(p.get("node") == "stage" for p in data)
    assert len({p.get("feedback") for p in data if p.get("node") != "stage"}) == 2
    parts = replay(kind, slot, reward, [p["event"] for p in data], GRID, [0.05])
    shadow = np.array([[sum(p["parts_by_g"][g]) for g in GRID] for p in data])
    assert np.abs(value_of(parts[:, :, 0]) - shadow).max() < 1e-12
