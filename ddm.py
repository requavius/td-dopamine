from config import *
import pyddm, pyddm.plot
import matplotlib.pyplot as plt

def drift_rate(delta, f_val, state: ModelState, t, s):
    decay_rate = 0.3 
    if state.t > 1:
        delta = state.stage_log[(state.t-1)*stage_amt-1]['delta'] * np.exp(-decay_rate * s)
    fatigue = np.log1p(.3 * t)
    reward_signal = np.log1p(10*(f_val * delta))
    return fatigue - reward_signal

def ddm(delta, s, state: ModelState, f_val, k_val, b_val):
    dt = 1/stage_amt
    time = state.t + dt*s 
    model = pyddm.gddm(drift=drift_rate,
                       noise = .01,
                       bound= lambda b_val : (1-b_val)*stage_amt,
                       starting_position = lambda k_val : k_val * stage_amt/10,
                       T_dur = float(state.t + dt*s), 
                       dt = dt * 0.01,
                       parameters={'f_val' : f_val,'k_val' : k_val,'b_val' : b_val}, 
                       conditions=['delta', 'state', 's'])
    sol = model.solve(conditions={'delta' : delta, 'state' : state, 's' : s})
    return sol.pdf('correct')[-1], time
def ddmtofit(delta, state: ModelState):
    dt = float(1/stage_amt)
    model_to_fit = pyddm.gddm(drift=drift_rate,
                       noise = .01,
                       bound= lambda b_val : (1-b_val)*stage_amt,
                       starting_position = lambda k_val : k_val * stage_amt/10,
                       T_dur = float(state.t), 
                       dt = dt,
                       parameters={'f_val' : (0,1),'k_val' : (0,1),'b_val' : (0,1)}, 
                       conditions=['delta', 'state', 's'])
    samp = pyddm.Sample.from_pandas_dataframe(state.first_passage, rt_column_name="Engaged T", choice_column_name="Engagment")
    fitted = model_to_fit.fit()


                       