from config import ModelState, np, stage_amt, drift_rate
import logging
import pyddm

logging.getLogger("pyddm").setLevel(logging.WARNING)

pyddm.set_N_cpus(4)


def effective_delta(delta, state: ModelState, s, decay_rate=0.3):
    if state.t > 1:
        idx = (state.t - 1) * abs(stage_amt - 1)
        if idx < len(state.stage_log):
            return state.stage_log[idx]["delta"] * np.exp(-decay_rate * s)
    return delta


class ddm:
    def modelsolve(delta, s, state: ModelState, f_val, k_val, b_val):
        dt = 1 / stage_amt
        eff_delta = effective_delta(delta, state, s)
        model = pyddm.gddm(
            drift=drift_rate,
            noise=0.5,
            bound=lambda b_val: (1 - b_val) * 5,
            starting_position=lambda k_val: k_val * stage_amt / 10,
            T_dur=dt,
            parameters={"f_val": f_val, "k_val": k_val, "b_val": b_val},
            conditions=["delta"],
        )
        sol = model.solve(conditions={"delta": eff_delta})
        return sol, dt, model

    def sample_pandas(self, state):
        df = state.first_passage.copy()
        df["Disengagment"] = df["Disengagment"].astype(int)
        return pyddm.Sample.from_pandas_dataframe(
            df, rt_column_name="Engaged T", choice_column_name="Disengagment"
        )

    def fit(self, state: ModelState):
        model_to_fit = pyddm.gddm(
            drift=drift_rate,
            noise=0.5,
            bound=lambda b_val: (1 - b_val) * 5,
            starting_position=lambda k_val: k_val * stage_amt / 10,
            T_dur=(1.0 / stage_amt) + 1.0,
            parameters={"f_val": (0, 1), "k_val": (0, 1), "b_val": (0, 1)},
            conditions=["delta"],
        )
        sample = self.sample_pandas(state)
        model_to_fit.fit(sample, verbose=False)
        return model_to_fit

    def loglikelihood(self, state):
        model = self.fit(state)
        sample = self.sample_pandas(state)
        lossfunc = pyddm.get_model_loss(model, sample)
        return lossfunc, model.parameters()
