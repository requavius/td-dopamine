"""Can the discount factor γ be recovered from the app's decisions? A profile likelihood.

The person learns at their true γ (α 0.05). Shadow TD learners, one per grid γ, run on
the same rewards, as the app could replay a person's history; each gives the value every
recorded start and redo decision would have had under that γ. (f, k) is fitted at each
grid γ, and the grid γ with the lowest -log L is the estimate. BOUND, SIGMA at the
pinned values, T0 at 0 (the simulated people have none), so γ is the only thing wrong.

The task is the app's, as in experiments.misspecification: question types with Elo
ratings, skips on, InfoController steering from the default setting (redo prompts and
feedback among its settings), 300 start decisions plus their redo decisions. The
controller sees value at γ = 0.9, as the app computes it.

People (f, k): (0.8, 0.8), (1.2, 0.4), (0.6, 1.2). True γ: 0.7, 0.8, 0.9, 0.95.
Grid: 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99. Constants: ACHIEVEMENT_SCALE 4,
continuing bootstrap, T_DUR 7.0. Each (person, γ, rep) has its own seed, so runs split
across machines by --gammas and --reps never overlap, and --replot merges every
results/gamma_profile/runs_*.csv.

Claims, fixed before the first run (2026-10-28):
    P1  the grid γ with the lowest -log L is the true γ or its grid neighbour in at least
        80% of runs
    P2  the profile is not flat: every grid γ more than 0.1 from the estimate costs at
        least 1.92 in -log L (a 95% likelihood-ratio interval inside ±0.1), in at least
        80% of runs (the 80% was added on 2026-10-28, before any result was read)

The simulated person is credited the end-of-set reward with feedback hidden, as the app's
score has shown it since 2026-10-28.

    python -m experiments.gamma_profile --gammas 0.7,0.8 --reps 0,1 --tag server
    python -m experiments.gamma_profile --replot
"""
import argparse
import contextlib
import dataclasses
import random

import numpy as np
import pandas as pd

from temporal.controller import Environment, InfoController, apply, measured
from temporal.inference import CONDITIONS, build_sample, passages_to_frame
from temporal.model import accumulator as acc
from temporal.model.conditions import carried_delta
from temporal.model.params import UserParams
from temporal.model.questions import Ratings, synthetic_type_diffs
from temporal.model.simulation import Simulation
from temporal.model.competence import success_prob
from temporal.model.value import ACHIEVEMENT_SCALE, V, late_reward, slot_reward, value_of_stage

from .common import Quiet, _design_args, design, out_dir, run_pool, save_conditions

D = design("gamma_profile", people=((0.8, 0.8), (1.2, 0.4), (0.6, 1.2)),
           true_g=(0.7, 0.8, 0.9, 0.95), reps=5, decisions=300,
           grid=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99))
OUT = out_dir(D.out)
GRID = D.grid
APP_G = 0.9  # what the app computes value with
PEOPLE = list(D.people)
TRUE_G = D.true_g
N_APP = D.decisions
CACHE_CAP = 15000
LR_95 = 1.92  # half the chi-square(1) 95% point
# The fixed setting of the feedback-block follow-up: misses come back only through redo
# prompts, which are offered only with feedback on.
BLOCK_ENV = Environment(target_p=0.6, review_rate=0.0, feedback=True, return_skipped=False,
                        redo_prompts=True)


class GridShadow(Simulation):
    """The person learns at their own γ and α; one shadow TD learner per grid γ, at the
    app's α, runs on the rewards the model assumes. Every start and redo record gets
    `v_by_g`, its value under each grid γ, and `parts_by_g`, that value's two parts before
    snapping: (V(0), carried error) at a start, (slot share, carried error) at a redo.

    With `reward="achievement"` the shadows are paid exactly what the person is. With
    another reward the person learns from it, and the shadows are paid the achievement
    reward, as the app would compute it."""

    def __init__(self, params, grid, **kw):
        super().__init__(params, **kw)
        self.grid = grid
        self.shadow = {g: {"theta": np.zeros_like(self.state.theta), "slot": 0.0,
                           "set": 0.0, "deltas": []} for g in grid}
        self.model_credited = 0.0

    def _V0(self, g):
        own, self.state.theta = self.state.theta, self.shadow[g]["theta"]
        v = V(self.state, 0)
        self.state.theta = own
        return v

    def step_stage(self, s, reward=0.0):
        delta = super().step_stage(s, reward)
        if self.reward != "achievement":
            reward, self.model_credited = slot_reward(
                self.state, self.achievement_so_far, self.model_credited, self.feedback,
                s == self.state.stage_amt - 1)
        for g, sh in self.shadow.items():
            own, self.state.theta = self.state.theta, sh["theta"]
            sh["slot"] = value_of_stage(self.state, UserParams(f=1.0, k=1.0, g=g), s, reward)
            sh["theta"], self.state.theta = self.state.theta, own
            sh["deltas"].append(sh["slot"])
        return delta

    def run_episode(self):
        for sh in self.shadow.values():
            sh["deltas"] = []
        self.model_credited = 0.0
        abandoned = self.state.abandoned
        out = super().run_episode()
        for sh in self.shadow.values():
            sh["set"] = 0.0 if self.state.abandoned > abandoned else float(sum(sh["deltas"]))
        return out

    def initiate(self):
        parts = {g: (self._V0(g), carried_delta(sh["set"])) for g, sh in self.shadow.items()}
        out = super().initiate()
        rec = self.state.initiation_passages[-1]
        rec["parts_by_g"] = parts
        rec["v_by_g"] = {g: acc.snap_value(v0 + d) for g, (v0, d) in parts.items()}
        return out

    def _redoes(self, s, item):
        n = self.state.stage_amt
        parts = {g: ((1.0 / n) * g ** (n - 1 - s), carried_delta(sh["slot"]))
                 for g, sh in self.shadow.items()}
        out = super()._redoes(s, item)
        rec = self.state.first_passage[-1]
        rec["parts_by_g"] = parts
        rec["v_by_g"] = {g: acc.snap_value(a + d) for g, (a, d) in parts.items()}
        return out

    # An end-of-set redo whose outcome is shown (`end_feedback`) lands its reward on the
    # set's last step, and the error adds to the set's; each shadow takes the same, at its
    # own γ and the app's α, paid the achievement reward as the app would compute it.
    def _extra_question(self, item, set_delta):
        p_item = success_prob(self.state, item)
        landed = super()._extra_question(item, set_delta)
        if landed:
            n = self.state.stage_amt
            reward = landed if self.reward == "achievement" else ACHIEVEMENT_SCALE * (1.0 - p_item) / n
            for g, sh in self.shadow.items():
                own, self.state.theta = self.state.theta, sh["theta"]
                sh["deltas"].append(late_reward(self.state, UserParams(f=1.0, k=1.0, g=g),
                                                n - 1, reward))
                sh["theta"], self.state.theta = self.state.theta, own
        return landed

    # An end-of-set redo: the undiscounted slot share, and each shadow's error for the set.
    def _redoes_at_end(self, item, set_delta):
        n = self.state.stage_amt
        parts = {g: (1.0 / n, carried_delta(float(sum(sh["deltas"]))))
                 for g, sh in self.shadow.items()}
        out = super()._redoes_at_end(item, set_delta)
        rec = self.state.first_passage[-1]
        rec["parts_by_g"] = parts
        rec["v_by_g"] = {g: acc.snap_value(a + d) for g, (a, d) in parts.items()}
        return out

    # What the app holds: value from the γ = 0.9 learner, not the person's own.
    def app_view(self, ratings):
        view = measured(self.state, ratings)
        sh = self.shadow[APP_G]
        view.theta = sh["theta"].copy()
        view.slot_delta, view.last_set_delta = sh["slot"], sh["set"]
        view.initiation_passages = [dict(p, value=p["v_by_g"][APP_G])
                                    for p in self.state.initiation_passages]
        view.first_passage = [dict(p, value=p["v_by_g"][APP_G]) for p in self.state.first_passage]
        return view


# The app's task for one person. The defaults are this experiment's design; the options
# are the follow-ups':
#   a        the person's TD learning rate (the shadows keep the app's 0.05)
#   reward   what the person values, Simulation's `reward` (the shadows keep achievement)
#   person   misspecification.PersonConstants: the person's own SIGMA and BOUND
#   blocks   None: InfoController steers. B: no controller; BLOCK_ENV with feedback on for
#            B start decisions, then off for B, and so on
#   grid     the shadow learners' γ values (GRID); t0 the person's non-decision time
def run_app(f, k, g_true, seed, n=N_APP, a=0.05, reward="achievement", person=None,
            blocks=None, grid=GRID, t0=0.0):
    random.seed(seed)
    np.random.seed(seed)
    ctl = None if blocks else InfoController(seed=seed)
    start = BLOCK_ENV if blocks else ctl.env
    own = person or contextlib.nullcontext
    ratings = Ratings()
    sim = GridShadow(UserParams(f=f, k=k, g=g_true, a=a, t0=t0), grid, reentry_cost=0.6,
                     review_rate=start.review_rate, review_mode=start.review_mode,
                     target_p=start.target_p, feedback=start.feedback, ratings=ratings,
                     skips=True, return_skipped=start.return_skipped, reward=reward)
    st = sim.state
    st.type_diffs = np.array(synthetic_type_diffs(1.0), dtype=float)
    st.ability = 0.0
    st.in_session = True
    with own():
        sim.run_episode()
    while len(st.initiation_passages) < n:
        with own():
            sim.tick()
        if person is not None:
            person.trim()
        elif len(acc._SOLUTION_CACHE) > CACHE_CAP:
            acc.clear_caches()
        if blocks:
            on = (len(st.initiation_passages) // blocks) % 2 == 0
            apply(sim, dataclasses.replace(BLOCK_ENV, feedback=on))
        elif ctl.due(st):
            ctl.plan(sim.app_view(ratings), UserParams(f=f, k=k))
            apply(sim, ctl.env)
    starts, redos = st.initiation_passages, st.first_passage
    upto = starts[n - 1]["timestep"]
    return starts[:n] + [r for r in redos if r["timestep"] <= upto]


# (f, k) at a given value, with its -log L.
def fit_nll(df):
    model = acc.build_model({"f_val": (0.0, 3.0), "k_val": (0.0, 3.0)})
    data = df[[*CONDITIONS, "GO", "RT"]].copy()
    data.attrs["undecided"] = df.attrs.get("undecided", {})
    model.fit(build_sample(data), verbose=False)
    drift = model.parameters()["drift"]
    return float(drift["f_val"]), float(drift["k_val"]), float(model.get_fit_result().value())


def seed_of(f, k, g_true, rep):
    return rep * 10**6 + int(round(100 * g_true)) * 10**4 + int(round(10 * f)) * 100 + int(round(10 * k))


# One row per grid γ: (f, k) fitted with every recorded decision valued at that γ.
def profile(data):
    fits = []
    for g in GRID:
        df = passages_to_frame([dict(p, value=p["v_by_g"][g]) for p in data])
        fits.append((g, *fit_nll(df)))
    redo_slots = np.bincount([p["stage"] for p in data if p.get("node") == "stage"], minlength=4)
    starts = [p for p in data if p.get("node") != "stage"]
    return [dict(g=g, f_hat=fh, k_hat=kh, nll=nll, starts=len(starts),
                 feedback_share=float(np.mean([p.get("feedback", 1) for p in starts])),
                 redos=int(redo_slots.sum()), **{f"redo_slot{i}": int(c) for i, c in enumerate(redo_slots)})
            for g, fh, kh, nll in fits]


def _one(task):
    f, k, g_true, rep = task
    seed = seed_of(f, k, g_true, rep)
    acc.clear_caches()
    with Quiet():
        rows = profile(run_app(f, k, g_true, seed))
    return [dict(f_true=f, k_true=k, g_true=g_true, rep=rep, seed=seed, **r) for r in rows]


RUN = ["f_true", "k_true", "g_true", "rep"]


# One row per run: the estimate, P1, P2, the 95% likelihood-ratio interval on the grid,
# and the bias in f at the estimate and at the app's γ. `keys` names a run.
def per_run(df, keys=RUN):
    grid = sorted(float(g) for g in df.g.unique())
    runs = []
    for key, r in df.groupby(keys):
        r = r.sort_values("g")
        info = dict(zip(keys, key))
        f_true, g_true = info["f_true"], info["g_true"]
        best = r.loc[r.nll.idxmin()]
        far = r[(r.g - best.g).abs() > 0.1 + 1e-9]
        near = r[r.nll - best.nll <= LR_95]
        runs.append(dict(info, g_hat=best.g,
                         p1=abs(grid.index(best.g) - grid.index(g_true)) <= 1,
                         p2=bool(len(far) == 0 or (far.nll - best.nll).min() >= LR_95),
                         lo=near.g.min(), hi=near.g.max(),
                         f_bias=100 * (best.f_hat - f_true) / f_true,
                         f_bias_app=100 * (r[r.g == APP_G].f_hat.iloc[0] - f_true) / f_true,
                         k_bias=100 * (best.k_hat - info["k_true"]) / info["k_true"],
                         redos=best.redos, feedback_share=best.feedback_share))
    return pd.DataFrame(runs), grid


def summarize(df):
    runs, grid = per_run(df)
    print(f"\n== γ profile: {len(runs)} runs, grid {grid} ==")
    print(f"P1 (estimate within one grid step): {runs.p1.mean():.0%} of runs (claim: ≥ 80%)")
    print(f"P2 (95% interval inside ±0.1):     {runs.p2.mean():.0%} of runs (claim: ≥ 80%)")
    print(f"\n{'true γ':>6} {'runs':>4} {'γ̂ (each run)':>32} {'95% intervals':>30} "
          f"{'f bias at γ̂':>12} {'at 0.9':>8} {'redos':>5}")
    for g_true, r in runs.groupby("g_true"):
        hats = ",".join(f"{x:g}" for x in r.g_hat)
        ivs = " ".join(f"[{a:g},{b:g}]" for a, b in zip(r.lo, r.hi))
        print(f"{g_true:>6g} {len(r):>4} {hats:>32} {ivs[:30]:>30} {r.f_bias.mean():+11.0f}% "
              f"{r.f_bias_app.mean():+7.0f}% {r.redos.mean():5.0f}")
    print("\nmean Δ(-log L) from the best grid γ, by true γ:")
    df = df.merge(df.groupby(RUN).nll.min().rename("best"), on=RUN)
    df["d"] = df.nll - df.best
    print(df.pivot_table(index="g_true", columns="g", values="d", aggfunc="mean").round(1).to_string())
    return runs


def floats(s):
    return tuple(float(x) for x in s.split(","))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="merge and summarize the saved results")
    p.add_argument("--gammas", type=floats, default=TRUE_G, help="true γ values to run")
    p.add_argument("--reps", type=lambda s: tuple(int(x) for x in s.split(",")),
                   default=tuple(range(D.reps)), help="which reps to run, for example 0,1")
    p.add_argument("--tag", default="local", help="names this machine's file, runs_<tag>.csv")
    _design_args(p)
    a = p.parse_args()
    if not a.replot:
        save_conditions(D)
        tasks = [(f, k, g, rep) for g in a.gammas for f, k in PEOPLE for rep in a.reps]
        run_pool(_one, tasks, OUT / f"runs_{a.tag}.csv", fresh=True,
                 label=lambda rows: f"f {rows[0]['f_true']} k {rows[0]['k_true']} γ {rows[0]['g_true']}")
    files = sorted(OUT.glob("runs_*.csv"))
    summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
