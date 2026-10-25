"""How precisely f and k can be measured, by the kind of decisions in the data.

Expected standard deviation of f and k from 300 start decisions (and the redo decisions
among them): sqrt(diag(I^-1)), with I the Fisher information of the decisions
(`controller.fisher`) at the true f and k. The 16 people of the 4 x 4 grid x 2 seeds,
question-type task, one run each with the default setting (start decisions only) and one
with a fixed redo-prompt setting (starts and redos). Warm-only designs keep the in-session
starts and scale their information to 300 start decisions, as a person in the app gives
(the fit never sees a cold start from a human). Current model: gamma 0.9,
ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s.

README: the table in "How value and cost together decide what can be measured". Paper
Figure 6d.

    python -m experiments.precision            # run, then summarize
    python -m experiments.precision --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from temporal.controller import Environment, fisher, run
from temporal.model.params import UserParams

from .common import LEVELS, args, out_dir, run_pool

OUT = out_dir("precision")
DESIGNS = (("starts", "start decisions, cold and warm"), ("warm", "start decisions, warm only"),
           ("starts_redo", "starts and redos, cold and warm"), ("warm_redo", "starts and redos, warm only"))


def _one(task):
    f, k, seed = task
    plain = run(UserParams(f=f, k=k), env=Environment(), n_decisions=300, seed=seed).state
    redo = run(UserParams(f=f, k=k), env=Environment(0.6, 0.0, True, False, True),
               n_decisions=300, seed=seed).state
    rows = []
    for name, st, with_redo, warm_only in (("starts", plain, False, False),
                                           ("warm", plain, False, True),
                                           ("starts_redo", redo, True, False),
                                           ("warm_redo", redo, True, True)):
        starts = [p for p in st.initiation_passages if not warm_only or p["in_session"]]
        data = starts + (st.first_passage if with_redo else [])
        info = fisher(data, f, k) * (300 / max(len(starts), 1))
        sd = np.sqrt(np.diag(np.linalg.inv(info)))
        rows.append(dict(design=name, f_true=f, k_true=k, seed=seed, sd_f=sd[0], sd_k=sd[1],
                         starts=len(starts), cost_sd=float(np.std([p["cost"] for p in data]))))
    return rows


def summarize(df):
    print("== expected SD from 300 start decisions, medians over 32 runs ==")
    print(f"{'decisions in the data':>32}  SD of f  SD of k  sd of cost")
    for name, label in DESIGNS:
        d = df[df.design == name]
        print(f"{label:>32}  {d.sd_f.median():7.3f}  {d.sd_k.median():7.3f}  {d.cost_sd.median():10.3f}")


if __name__ == "__main__":
    a = args(__doc__)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        tasks = [(f, k, s) for f in LEVELS for k in LEVELS for s in (0, 1)]
        df = run_pool(_one, tasks, OUT / "runs.csv")
        df = df.sort_values(["f_true", "k_true", "seed", "design"])
        df.to_csv(OUT / "runs.csv", index=False)
    summarize(df)
