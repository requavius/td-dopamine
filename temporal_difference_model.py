from config import (
    ModelState,
    UserParams,
    V,
    draw_outcome,
    update_ability,
    value_at_choice_point,
    value_of_stage,
)
from initiation import initiation_decision


class Simulation:

    def __init__(self, params: UserParams, ability=None, reentry_cost=None):
        self.params = params
        self.state = ModelState(t=1)
        if ability is not None:
            self.state.ability = ability
        if reentry_cost is not None:
            self.state.reentry_cost = reentry_cost

    def step_stage(self, s, reward=0.0):
        delta = value_of_stage(self.state, self.params, s, reward)
        self.state.rpe[s] = delta
        return delta

    def run_episode(self):
        state = self.state
        last = state.stage_amt - 1

        deltas = []
        for s in range(state.stage_amt):
            if s == last:
                success = draw_outcome(state)
                deltas.append(self.step_stage(s, reward=float(success)))
                update_ability(state, success)
            else:
                deltas.append(self.step_stage(s))

        state.episodes += 1
        return deltas

    def initiate(self):
        state, p = self.state, self.params

        record = initiation_decision(
            value=value_at_choice_point(state),
            cost=state.reentry_cost,
            f_val=p.f,
            k_val=p.k,
        )
        record["episode"] = state.episodes
        record["timestep"] = state.t
        state.initiation_passages.append(record)

        if record["GO"] != 1:
            # STAY or timeout: the cheap repeat action consumes this tick and
            # teaches the learner nothing. Counted, never silently skipped.
            state.stuck += 1
            return False
        return True

    def converged(self, low_rpe_streak):
        state = self.state
        average_v = sum(V(state, s) for s in range(state.stage_amt)) / state.stage_amt
        return low_rpe_streak >= state.stage_amt and average_v > 0.1

    def run(self, max_decisions=None, max_episodes=None, max_ticks=20000,
            stop_on_convergence=False):
        state = self.state
        low_rpe_streak = 0

        # The learner is already in the task; first episode is not gated.
        self.run_episode()

        while True:
            self.initiate()
            state.t += 1

            if max_decisions is not None and len(state.initiation_passages) >= max_decisions:
                break
            if max_episodes is not None and state.episodes >= max_episodes:
                break
            if state.t >= max_ticks:
                break

            if state.initiation_passages[-1]["GO"] == 1:
                self.run_episode()

                max_rpe = max(abs(x) for x in state.rpe.values())
                low_rpe_streak = low_rpe_streak + 1 if max_rpe < 0.05 else 0
                if stop_on_convergence and self.converged(low_rpe_streak):
                    break

        return state
