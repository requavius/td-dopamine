"""Fitting (f, k) with the model's fixed constants to people who do not share them.

The person differs from the model in one constant at a time: the discount factor γ, the
TD learning rate α, the accumulator's noise SIGMA or its BOUND. Everything that measures
them keeps the defaults (γ 0.9, α 0.05, SIGMA 0.7, BOUND 1.5):

- γ and α never enter the fit directly. They shape the person's learned value, and the fit
  takes value as an observed condition, which the app computes at the defaults. So a
  shadow TD learner at the defaults runs on the same rewards as the person, and its value
  is what the "analyst" fit sees. The "oracle" fit gets the person's own value instead,
  so the difference between the two fits is the cost of assuming the wrong γ or α.
- SIGMA and BOUND apply to the person's own start, redo and give-up draws only. Every fit,
  and the controller's rollouts, use the pinned values. Here analyst and oracle coincide.

Two parts:
    app     the question-type task, skips on, the InfoController steering it from the
            default setting (redo prompts among its 96 candidates); 300 start decisions
            plus the redo decisions up to the 300th, as in experiments.controllers
    single  the single-diff task of `recovery.collect_single`: one learner, 1200
            decisions, mid cold-start cost, scheduled review, so no redo decisions

People (f, k): (0.8, 0.8), (1.2, 0.4), (0.6, 1.2); 5 seeds each, the same at every level.
The default level is run once per part and shared by all four constants. Bias is
(estimate − true) / true, mean ± SE over the 15 fits. Constants of the fit: γ 0.9,
ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s. Any claim about a difference
follows the 2·SE rule.

README: "The person may not share the model's constants", under "Recovery demonstrates
identifiability, not accuracy".

    python -m experiments.misspecification            # run (hours: the app part), then summarize
    python -m experiments.misspecification --replot   # summarize the saved results
"""
import contextlib
import dataclasses
import itertools
import random

import numpy as np
import pandas as pd

from temporal.controller import InfoController, apply, measured
from temporal.inference import fit, passages_to_frame
from temporal.model import accumulator as acc
from temporal.model import persistence
from temporal.model.conditions import carried_delta
from temporal.model.params import REENTRY_COST_MID, UserParams
from temporal.model.questions import Ratings, synthetic_type_diffs
from temporal.model.simulation import Simulation
from temporal.model.value import V, value_of_stage

from .common import Quiet, args, out_dir, run_pool, se

OUT = out_dir("misspecification")
DEFAULTS = {"g": 0.9, "a": 0.05, "sigma": acc.SIGMA, "bound": acc.BOUND}
LEVELS = {
    "g": (0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
    "a": (0.01, 0.025, 0.05, 0.1, 0.2),
    "sigma": (0.5, 0.6, 0.7, 0.85, 1.0),
    "bound": (1.0, 1.25, 1.5, 1.75, 2.0),
}
NAMES = {"g": "γ", "a": "α", "sigma": "SIGMA", "bound": "BOUND"}
PEOPLE = [(0.8, 0.8), (1.2, 0.4), (0.6, 1.2)]
REPS = 5
N_APP, N_SINGLE = 300, 1200
# Solved conditions kept per process. The InfoController's posterior draws keep adding new
# (f, k), and uncapped the caches reached ~7 GB a worker. A solution is deterministic, so
# clearing them changes no draw, only time.
CACHE_CAP = 15000


class ShadowSimulation(Simulation):
    """The person learns at their own γ and α; a shadow TD learner at the defaults runs on
    the same rewards, and its value is recorded beside theirs as `value_assumed`."""

    def __init__(self, params, assumed, **kw):
        super().__init__(params, **kw)
        self.assumed = assumed
        self.shadow_theta = np.zeros_like(self.state.theta)
        self.shadow_slot_delta = 0.0
        self.shadow_set_delta = 0.0
        self._shadow_deltas = []

    def _shadow_V0(self):
        own, self.state.theta = self.state.theta, self.shadow_theta
        v = V(self.state, 0)
        self.state.theta = own
        return v

    def step_stage(self, s, reward=0.0):
        delta = super().step_stage(s, reward)
        own, self.state.theta = self.state.theta, self.shadow_theta
        self.shadow_slot_delta = value_of_stage(self.state, self.assumed, s, reward)
        self.shadow_theta, self.state.theta = self.state.theta, own
        self._shadow_deltas.append(self.shadow_slot_delta)
        return delta

    def run_episode(self):
        self._shadow_deltas = []
        abandoned = self.state.abandoned
        out = super().run_episode()
        self.shadow_set_delta = (0.0 if self.state.abandoned > abandoned
                                 else float(sum(self._shadow_deltas)))
        return out

    def initiate(self):
        assumed = self._shadow_V0() + carried_delta(self.shadow_set_delta)
        out = super().initiate()
        self.state.initiation_passages[-1]["value_assumed"] = acc.snap_value(assumed)
        return out

    def _redoes(self, s, item):
        remaining = self.state.stage_amt - 1 - s
        assumed = ((1.0 / self.state.stage_amt) * self.assumed.g ** remaining
                   + carried_delta(self.shadow_slot_delta))
        out = super()._redoes(s, item)
        self.state.first_passage[-1]["value_assumed"] = acc.snap_value(assumed)
        return out

    # Give-up records keep the analyst's carried error too, for the patience fit.
    def _gives_up(self, item, p_true, review):
        gave_up = super()._gives_up(item, p_true, review)
        self.state.persistence[-1]["delta_assumed"] = self.shadow_slot_delta
        return gave_up


class PersonConstants:
    """The person's SIGMA and BOUND, with solution caches of their own, for the person's
    draws only. Outside `with person():` the pinned constants and caches are back."""

    def __init__(self, sigma, bound):
        self.sigma, self.bound = sigma, bound
        self.caches = ({}, {}, {})
        self.default = (sigma, bound) == (DEFAULTS["sigma"], DEFAULTS["bound"])

    @contextlib.contextmanager
    def __call__(self):
        if self.default:
            yield
            return
        saved = acc._MODEL_CACHE, acc._SOLUTION_CACHE, acc._SAMPLER_CACHE
        acc._MODEL_CACHE, acc._SOLUTION_CACHE, acc._SAMPLER_CACHE = self.caches
        acc.SIGMA = persistence.SIGMA = self.sigma
        acc.BOUND = persistence.BOUND = self.bound
        try:
            yield
        finally:
            acc.SIGMA = persistence.SIGMA = DEFAULTS["sigma"]
            acc.BOUND = persistence.BOUND = DEFAULTS["bound"]
            acc._MODEL_CACHE, acc._SOLUTION_CACHE, acc._SAMPLER_CACHE = saved

    # Called outside the context: empties whichever cache set has passed CACHE_CAP.
    def trim(self):
        if len(acc._SOLUTION_CACHE) > CACHE_CAP:
            acc.clear_caches()
        if len(self.caches[1]) > CACHE_CAP:
            for cache in self.caches:
                cache.clear()


def _assumed(passages):
    return [dict(p, value=p["value_assumed"]) for p in passages]


# What the app would hold: measured ability and ratings, and value from the defaults.
def analyst_view(sim, ratings):
    st = sim.state
    return dataclasses.replace(
        measured(st, ratings), theta=sim.shadow_theta.copy(),
        slot_delta=sim.shadow_slot_delta, last_set_delta=sim.shadow_set_delta,
        initiation_passages=_assumed(st.initiation_passages),
        first_passage=_assumed(st.first_passage),
        persistence=[dict(p, delta=p["delta_assumed"]) for p in st.persistence])


# controller.run with an InfoController, keeping the person's constants and the analyst's
# view apart. At the defaults it is controller.run draw for draw.
def run_app(f, k, settings, seed, n=N_APP):
    random.seed(seed)
    np.random.seed(seed)
    params = UserParams(f=f, k=k, g=settings["g"], a=settings["a"])
    assumed = UserParams(f=f, k=k)  # plan() reads only g and a from it
    person = PersonConstants(settings["sigma"], settings["bound"])
    ctl = InfoController(seed=seed)
    start = ctl.env
    ratings = Ratings()
    sim = ShadowSimulation(params, assumed, reentry_cost=0.6, review_rate=start.review_rate,
                           review_mode=start.review_mode, target_p=start.target_p,
                           feedback=start.feedback, ratings=ratings, skips=True,
                           return_skipped=start.return_skipped)
    st = sim.state
    st.type_diffs = np.array(synthetic_type_diffs(1.0), dtype=float)
    st.ability = 0.0

    st.in_session = True
    with person():
        sim.run_episode()
    while len(st.initiation_passages) < n:
        with person():
            sim.tick()
        person.trim()
        if ctl.due(st):
            ctl.plan(analyst_view(sim, ratings), assumed)
            apply(sim, ctl.env)

    starts, redos = st.initiation_passages, st.first_passage
    upto = starts[n - 1]["timestep"]
    return starts[:n] + [r for r in redos if r["timestep"] <= upto], len(redos)


# recovery.collect_single's learner, with the shadow beside it.
def run_single(f, k, settings, seed, n=N_SINGLE):
    random.seed(seed)
    np.random.seed(seed)
    person = PersonConstants(settings["sigma"], settings["bound"])
    sim = ShadowSimulation(UserParams(f=f, k=k, g=settings["g"], a=settings["a"]),
                           assumed=UserParams(f=f, k=k), reentry_cost=REENTRY_COST_MID)
    with person():
        sim.run(max_decisions=n)
    return sim.state.initiation_passages + sim.state.first_passage, len(sim.state.first_passage)


def _one(task):
    part, factor, level, f, k, rep = task
    settings = dict(DEFAULTS, **{factor: level})
    seed = 1000 * rep + int(10 * f) * 10 + int(10 * k)
    acc.clear_caches()
    with Quiet():
        data, redos = (run_app if part == "app" else run_single)(f, k, settings, seed)
        oracle = passages_to_frame(data)
        f_or, k_or = fit(oracle)
        f_an, k_an = fit(passages_to_frame(_assumed(data)))
    v_or = np.array([p["value"] for p in data], float)
    v_an = np.array([p["value_assumed"] for p in data], float)
    undecided = sum(oracle.attrs["undecided"].values())
    return dict(part=part, factor=factor, level=level, f_true=f, k_true=k, rep=rep, seed=seed,
                rows=len(oracle), undecided=undecided, redo_prompts=redos,
                p_go=float(oracle["GO"].mean()), value_mean=float(v_or.mean()),
                value_assumed_mean=float(v_an.mean()),
                value_corr=(float(np.corrcoef(v_or, v_an)[0, 1])
                            if v_or.std() and v_an.std() else 1.0),
                f_oracle=f_or, k_oracle=k_or, f_analyst=f_an, k_analyst=k_an)


def tasks():
    out = []
    for part in ("app", "single"):
        levels = [("g", DEFAULTS["g"])] + [(fa, lv) for fa, lvs in LEVELS.items()
                                           for lv in lvs if lv != DEFAULTS[fa]]
        out += [(part, fa, lv, f, k, rep) for fa, lv in levels
                for (f, k), rep in itertools.product(PEOPLE, range(REPS))]
    return out


# The default runs, saved once under γ, count at every constant's default level.
def with_defaults(df):
    base = df[(df.factor == "g") & (df.level == DEFAULTS["g"])]
    return pd.concat([df] + [base.assign(factor=fa, level=DEFAULTS[fa])
                             for fa in LEVELS if fa != "g"], ignore_index=True)


def summarize(df):
    df = with_defaults(df)
    for part, title in (("app", f"app setting, InfoController, {N_APP} start decisions"),
                        ("single", f"single-diff task, {N_SINGLE} decisions")):
        p = df[df.part == part]
        print(f"\n== {title}: bias (estimate - true) / true, mean ± SE over "
              f"{len(PEOPLE) * REPS} fits ==")
        print(f"{'':>13} {'f analyst':>13} {'k analyst':>13} {'f/k analyst':>13} {'f oracle':>9}"
              f" {'k oracle':>9} {'undecided':>9} {'starts GO':>9} {'redos':>6}")
        for factor in LEVELS:
            for level, g in p[p.factor == factor].groupby("level"):
                cells = []
                for col in ("f_analyst", "k_analyst", "f_oracle", "k_oracle"):
                    rel = 100 * (g[col] - g[col[0] + "_true"]) / g[col[0] + "_true"]
                    cells.append(f"{rel.mean():+.0f} ± {se(rel):.0f}%" if "analyst" in col
                                 else f"{rel.mean():+.0f}%")
                # A wrong BOUND or SIGMA scales f and k together, so this ratio barely moves.
                # Undefined where a fit put k at 0 (single-diff task, γ = 0.99).
                if (g.k_analyst > 1e-6).all():
                    ratio = 100 * ((g.f_analyst / g.k_analyst) / (g.f_true / g.k_true) - 1)
                    cells.insert(2, f"{ratio.mean():+.0f} ± {se(ratio):.0f}%")
                else:
                    cells.insert(2, "k = 0")
                share = 100 * (g.undecided / (g.rows + g.undecided)).mean()
                print(f"{NAMES[factor]:>6} {level:<6g} {cells[0]:>13} {cells[1]:>13} {cells[2]:>13} "
                      f"{cells[3]:>9} {cells[4]:>9} {share:8.1f}% {g.p_go.mean():9.2f} "
                      f"{g.redo_prompts.mean():6.0f}")


if __name__ == "__main__":
    a = args(__doc__)
    if a.replot:
        df = pd.read_csv(OUT / "runs.csv")
    else:
        df = run_pool(_one, tasks(), OUT / "runs.csv", fresh=True)
        df = df.sort_values(["part", "factor", "level", "f_true", "k_true", "rep"])
        df.to_csv(OUT / "runs.csv", index=False)
    summarize(df)
