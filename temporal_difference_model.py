from config import ModelState, UserParams, value_of_stage
from ddm import effective_delta, modelsolve


class Simulation:
    """One learner: bundles the mutable ModelState with the fixed UserParams so
    they don't have to be threaded through every call."""

    def __init__(self, params: UserParams, skill=0.1):
        self.params = params
        self.state = ModelState(t=1, skill=skill)

    def useddm(self, s):
        state, p = self.state, self.params
        delta = value_of_stage(state, p, s)
        solved, _dt, _ = modelsolve(delta, s, state, p.f, p.k, p.b)
        # draw a real (choice, first-passage time) pair from the solved DDM so the
        # RT we later fit to actually comes from the diffusion, not a constant dt.
        sampled = solved.sample(1)
        if len(sampled.choice_upper) > 0: # hit upper bound -> disengaged
            disengaged, rt = True, float(sampled.choice_upper[0])
        elif len(sampled.choice_lower) > 0: # hit lower bound -> stayed engaged
            disengaged, rt = False, float(sampled.choice_lower[0])
        else: # undecided within T_dur -> no crossing
            disengaged, rt = False, None

        # skill grows with practice but saturates
        learning_gain = max(0, delta) * (1.0 - state.skill)
        state.skill += min(learning_gain / state.stage_amt, 1)
        return delta, disengaged, rt, solved

    def simulate(self):
        state = self.state
        disengaged = False
        for s in range(state.stage_amt):
            delta, disengaged, rt, _ = self.useddm(s)
            state.rpe[s] = delta
            if rt is not None: # skip undecided stages (no observed first passage)
                state.first_passage.append({
                    "Disengagment": disengaged,
                    "Engaged T": rt,
                    "delta": round(effective_delta(delta, s), 2),
                    'timestep': state.t
                })
            if disengaged:
                break
        return disengaged

