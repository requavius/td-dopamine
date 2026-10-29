"""Can the learning rate α, and which reward a person values, be measured as γ is?

γ is fitted by replaying a person's rewards through TD learners at candidate γ
(experiments.gamma_profile, recovery_lhc). α and the reward are the two constants still
assumed: α 0.05, and right answers worth 1 − p ("achievement"), against the fraction
right ("accuracy"), the earlier model. Both can be replayed the same way. Here shadow TD
learners run at every γ from 0.50 to 0.99 (steps of 0.01), every α in ALPHAS and both
rewards, on the rewards each reward form would pay for the same answers, as the app could
replay a person's record. Each recorded decision then has a value under every (γ, α,
reward).

People: a seeded Latin hypercube over f and k 0.2 to 1.5, γ 0.6 to 0.97, BOUND 1.0 to
2.0, T0 0 to 0.4 s, and log α from log 0.01 to log 0.2. Even-numbered people value
achievement, odd-numbered ones accuracy. Each plays the app's task (question types with
Elo ratings, skips on, InfoController steering as the app runs it, with γ 0.9, α 0.05
and achievement) for 300 start decisions plus their redo decisions. SIGMA 0.7,
ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0.

The fit. For each grid α and each reward, (f, k, γ, BOUND, T0) is fitted by maximum
likelihood, γ continuous by a cubic spline (recovery_lhc.Person.fit). That gives a
profile of -log L over α for each reward.
- α̂ for a reward: the vertex of the parabola in log α through the lowest grid point and
  its two neighbours (the grid point itself when it is at an end of the grid).
- The 95% interval for α: where the profile, linearly interpolated in log α, is within
  1.92 of its lowest grid value (Wilks, χ² with 1 df), clipped to the grid.
- The reward form: the one whose profile has the lower minimum. The evidence is the
  difference of the two minima.

Claims, fixed before any run (2026-10-28):
    A1  under the person's true reward, α̂ is within a factor of 1.5 of the true α in at
        least 80% of runs
    A2  under the true reward, the 95% interval for α contains the true α in at least 90%
        of runs (runs whose true α lies inside the grid, which is all of them)
    R1  the reward form with the lower profile minimum is the true one in at least 90% of
        runs
    R2  none: report the bias in f and k (i) the current way (α 0.05, achievement),
        (ii) α fitted under the assumed achievement reward, (iii) α and reward both chosen
        by the profile

    python -m experiments.alpha_reward --people 0-31 --tag server
    python -m experiments.alpha_reward --replot
"""
import argparse
import random

import numpy as np
import pandas as pd
from scipy.stats import qmc

from temporal.controller import InfoController, apply, measured
from temporal.model import accumulator as acc
from temporal.model.conditions import carried_delta
from temporal.model.params import UserParams
from temporal.model.questions import Ratings, synthetic_type_diffs
from temporal.model.simulation import REWARDS, Simulation
from temporal.model.value import ACHIEVEMENT_SCALE, V, slot_reward, value_of_stage

from .common import Quiet, _design_args, design, out_dir, run_pool, save_conditions, se
from .gamma_profile import APP_G, CACHE_CAP
from .misspecification import PersonConstants
from .recovery_lhc import G_FINE, NAMES, Person, span

# `ranges["a"]` is drawn uniformly in log α; it must stay last, as the hypercube's order.
D = design("alpha_reward", people=64, seed=20261030, decisions=300,
           alphas=(0.01, 0.015, 0.025, 0.035, 0.05, 0.07, 0.1, 0.14, 0.2),  # about √2 apart
           ranges={"f": (0.2, 1.5), "k": (0.2, 1.5), "g": (0.6, 0.97), "bound": (1.0, 2.0),
                   "t0": (0.0, 0.4), "a": (0.01, 0.2)})
OUT = out_dir(D.out)
N_PEOPLE = D.people
SEED = D.seed
N_STARTS = D.decisions
ALPHAS = D.alphas
APP_A = 0.05
RANGES = {**{k: v for k, v in D.ranges.items() if k != "a"},
          "log_a": tuple(np.log(D.ranges["a"]))}
LR_95 = 1.92


def people():
    unit = qmc.LatinHypercube(d=len(RANGES), seed=SEED).random(N_PEOPLE)
    lo = np.array([r[0] for r in RANGES.values()])
    hi = np.array([r[1] for r in RANGES.values()])
    out = []
    for i, row in enumerate(lo + unit * (hi - lo)):
        p = dict(zip(RANGES, row))
        p["a"] = float(np.exp(p.pop("log_a")))
        p["reward"] = REWARDS[i % 2]
        out.append(p)
    return out


class Replay(Simulation):
    """The person learns at their own γ, α and reward. One shadow TD learner per (γ, α,
    reward) runs on what that reward form pays for the same answers, at the same moments.
    Every start and redo record gets `parts`, its value's two parts before snapping at
    each key: (V(0), carried error) at a start, (slot share, carried error) at a redo."""

    def __init__(self, params, keys, **kw):
        super().__init__(params, **kw)
        self.keys = keys
        self.shadow = {key: {"theta": np.zeros_like(self.state.theta), "slot": 0.0,
                             "set": 0.0, "deltas": [],
                             "params": UserParams(f=1.0, k=1.0, g=key[0], a=key[1])}
                       for key in keys}

    def step_stage(self, s, reward=0.0):
        delta = super().step_stage(s, reward)
        last = s == self.state.stage_amt - 1
        pay = {}
        for r in REWARDS:
            earned = (self.achievement_so_far if r == "achievement"
                      else self.passed_so_far / ACHIEVEMENT_SCALE)
            pay[r], self.credited[r] = slot_reward(self.state, earned, self.credited[r],
                                                   self.feedback, last)
        own = self.state.theta
        for (_, _, r), sh in self.shadow.items():
            self.state.theta = sh["theta"]
            sh["slot"] = value_of_stage(self.state, sh["params"], s, pay[r])
            sh["theta"] = self.state.theta
            sh["deltas"].append(sh["slot"])
        self.state.theta = own
        return delta

    def run_episode(self):
        self.credited = {r: 0.0 for r in REWARDS}
        for sh in self.shadow.values():
            sh["deltas"] = []
        abandoned = self.state.abandoned
        out = super().run_episode()
        for sh in self.shadow.values():
            sh["set"] = 0.0 if self.state.abandoned > abandoned else float(sum(sh["deltas"]))
        return out

    def initiate(self):
        own = self.state.theta
        parts = {}
        for key, sh in self.shadow.items():
            self.state.theta = sh["theta"]
            parts[key] = (V(self.state, 0), carried_delta(sh["set"]))
        self.state.theta = own
        out = super().initiate()
        self.state.initiation_passages[-1]["parts"] = parts
        return out

    def _redoes(self, s, item):
        n = self.state.stage_amt
        parts = {(g, a, r): ((1.0 / n) * g ** (n - 1 - s), carried_delta(sh["slot"]))
                 for (g, a, r), sh in self.shadow.items()}
        out = super()._redoes(s, item)
        self.state.first_passage[-1]["parts"] = parts
        return out

    def _redoes_at_end(self, item, set_delta):
        n = self.state.stage_amt
        parts = {key: (1.0 / n, carried_delta(float(sum(sh["deltas"]))))
                 for key, sh in self.shadow.items()}
        out = super()._redoes_at_end(item, set_delta)
        self.state.first_passage[-1]["parts"] = parts
        return out

    # What the app holds: value from the learner at γ 0.9, α 0.05, achievement.
    def app_view(self, ratings):
        view = measured(self.state, ratings)
        sh = self.shadow[(APP_G, APP_A, "achievement")]
        view.theta = sh["theta"].copy()
        view.slot_delta, view.last_set_delta = sh["slot"], sh["set"]

        def at_app(p):
            v0, d = p["parts"][(APP_G, APP_A, "achievement")]
            return dict(p, value=acc.snap_value(v0 + d))
        view.initiation_passages = [at_app(p) for p in self.state.initiation_passages]
        view.first_passage = [at_app(p) for p in self.state.first_passage]
        return view


def simulate(i):
    truth = people()[i]
    seed = SEED + i
    random.seed(seed)
    np.random.seed(seed)
    keys = [(g, a, r) for g in G_FINE for a in ALPHAS for r in REWARDS]
    ctl = InfoController(seed=seed)
    start = ctl.env
    ratings = Ratings()
    person = PersonConstants(acc.SIGMA, truth["bound"])
    sim = Replay(UserParams(f=truth["f"], k=truth["k"], g=truth["g"], a=truth["a"],
                            t0=truth["t0"]), keys, reentry_cost=0.6,
                 review_rate=start.review_rate, review_mode=start.review_mode,
                 target_p=start.target_p, feedback=start.feedback, ratings=ratings,
                 skips=True, return_skipped=start.return_skipped, reward=truth["reward"])
    st = sim.state
    st.type_diffs = np.array(synthetic_type_diffs(1.0), dtype=float)
    st.ability = 0.0
    st.in_session = True
    with person():
        sim.run_episode()
    while len(st.initiation_passages) < N_STARTS:
        with person():
            sim.tick()
        person.trim()
        if len(acc._SOLUTION_CACHE) > CACHE_CAP:
            acc.clear_caches()
        if ctl.due(st):
            ctl.plan(sim.app_view(ratings), UserParams(f=truth["f"], k=truth["k"]))
            apply(sim, ctl.env)
    upto = st.initiation_passages[N_STARTS - 1]["timestep"]
    return truth, (st.initiation_passages[:N_STARTS]
                   + [r for r in st.first_passage if r["timestep"] <= upto])


def _person(records, a, r, i):
    node = np.array([p.get("node") == "stage" for p in records], dtype=np.int8)
    go = np.array([-1 if p["GO"] is None else p["GO"] for p in records], dtype=np.int8)
    rt = np.array([np.nan if p["RT"] is None else p["RT"] for p in records])
    parts = np.array([[p["parts"][(g, a, r)] for g in G_FINE] for p in records])
    h = dict(node=node, go=go, rt=rt, cost=np.array([p["cost"] for p in records]),
             base=parts[:, :, 0], delta=parts[:, :, 1], g_grid=np.array(G_FINE),
             **{f"true_{n}": 0.0 for n in NAMES})
    return Person(i, h=h)


def _one(i):
    acc.clear_caches()
    with Quiet():
        truth, records = simulate(i)
        rows = []
        for r in REWARDS:
            for a in ALPHAS:
                est, nll = _person(records, a, r, i).fit()
                rows.append(dict(person=i, reward_true=truth["reward"], a_true=truth["a"],
                                 **{f"true_{n}": truth[n] for n in NAMES},
                                 reward=r, a=a, nll=nll, decisions=len(records),
                                 **{f"est_{n}": est[j] for j, n in enumerate(NAMES)}))
    return rows


# α̂ and the 95% interval from one profile (a over the grid, its -log L).
def alpha_hat(alphas, nll):
    x, y = np.log(np.asarray(alphas)), np.asarray(nll)
    j = int(np.argmin(y))
    if 0 < j < len(x) - 1:
        c = np.polyfit(x[j - 1:j + 2], y[j - 1:j + 2], 2)
        best = -c[1] / (2 * c[0]) if c[0] > 0 else x[j]
        best = float(np.clip(best, x[j - 1], x[j + 1]))
    else:
        best = x[j]
    lim = y[j] + LR_95
    lo, hi = x[0], x[-1]
    for m in range(j, 0, -1):  # walk left to the first crossing
        if y[m - 1] > lim:
            lo = x[m - 1] + (lim - y[m - 1]) * (x[m] - x[m - 1]) / (y[m] - y[m - 1])
            break
    for m in range(j, len(x) - 1):
        if y[m + 1] > lim:
            hi = x[m] + (lim - y[m]) * (x[m + 1] - x[m]) / (y[m + 1] - y[m])
            break
    return float(np.exp(best)), float(np.exp(lo)), float(np.exp(hi))


def per_person(df):
    out = []
    for i, g in df.groupby("person"):
        row = dict(person=i, reward_true=g.reward_true.iloc[0], a_true=g.a_true.iloc[0],
                   **{f"true_{n}": g[f"true_{n}"].iloc[0] for n in NAMES})
        best = {}
        for r, p in g.groupby("reward"):
            p = p.sort_values("a")
            a_hat, lo, hi = alpha_hat(p.a, p.nll)
            best[r] = p.nll.min()
            row.update({f"a_hat_{r}": a_hat, f"a_lo_{r}": lo, f"a_hi_{r}": hi,
                        f"min_{r}": p.nll.min()})
            at = p.loc[p.nll.idxmin()]
            row.update({f"{n}_afit_{r}": at[f"est_{n}"] for n in NAMES})
        row["reward_hat"] = min(best, key=best.get)
        row["lr_reward"] = best["accuracy"] - best["achievement"]  # > 0 favours achievement
        cur = g[(g.reward == "achievement") & np.isclose(g.a, APP_A)].iloc[0]
        row.update({f"{n}_current": cur[f"est_{n}"] for n in NAMES})
        row.update({f"{n}_both": row[f"{n}_afit_{row['reward_hat']}"] for n in NAMES})
        out.append(row)
    return pd.DataFrame(out)


def summarize(df):
    p = per_person(df)
    print(f"== α and reward: {len(p)} people ({(p.reward_true == 'achievement').sum()} value "
          f"achievement, {(p.reward_true == 'accuracy').sum()} accuracy) ==")
    own = np.array([p.loc[i, f"a_hat_{p.loc[i, 'reward_true']}"] for i in p.index])
    lo = np.array([p.loc[i, f"a_lo_{p.loc[i, 'reward_true']}"] for i in p.index])
    hi = np.array([p.loc[i, f"a_hi_{p.loc[i, 'reward_true']}"] for i in p.index])
    ratio = own / p.a_true.to_numpy()
    a1 = np.mean((ratio < 1.5) & (ratio > 1 / 1.5))
    a2 = np.mean((p.a_true.to_numpy() >= lo) & (p.a_true.to_numpy() <= hi))
    print(f"A1 α̂ within a factor of 1.5 of the truth (true reward): {a1:.0%} "
          f"({'holds' if a1 >= 0.8 else 'does not hold'})")
    print(f"   log(α̂ / α): RMSE {np.sqrt(np.mean(np.log(ratio) ** 2)):.2f}, "
          f"median |ratio − 1| {np.median(np.abs(ratio - 1)):.0%}, "
          f"r(log α̂, log α) {np.corrcoef(np.log(own), np.log(p.a_true))[0, 1]:.2f}")
    print(f"A2 95% interval covers the true α: {a2:.0%} "
          f"({'holds' if a2 >= 0.9 else 'does not hold'}); median width, a factor of "
          f"{np.median(hi / lo):.1f}")
    for r, g in p.groupby("reward_true"):
        ok = (g.reward_hat == r).mean()
        lr = g.lr_reward if r == "achievement" else -g.lr_reward
        print(f"R1 true {r}: chosen in {ok:.0%} of {len(g)}; Δ min -log L in its favour, "
              f"median {lr.median():.1f} (5th percentile {lr.quantile(0.05):.1f})")
    r1 = (p.reward_hat == p.reward_true).mean()
    print(f"R1 overall: {r1:.0%} ({'holds' if r1 >= 0.9 else 'does not hold'})")
    print("\nR2 bias (mean of (estimate − true) / true, ± SE) and RMSE, by true reward:")
    for r, g in p.groupby("reward_true"):
        for label, col in (("current (α 0.05, achievement)", "current"),
                           ("α fitted, achievement assumed", "afit_achievement"),
                           ("α and reward chosen", "both")):
            cells = []
            for n in ("f", "k", "g"):
                rel = (g[f"{n}_{col}"] - g[f"true_{n}"]) / g[f"true_{n}"] * 100
                err = g[f"{n}_{col}"] - g[f"true_{n}"]
                cells.append(f"{n} {rel.mean():+5.1f} ± {se(rel):.1f}% (RMSE {np.sqrt(np.mean(err ** 2)):.3f})")
            print(f"  {r:>11} | {label:<30} | " + "  ".join(cells))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true", help="merge and summarize the saved results")
    ap.add_argument("--people", type=span, default=list(range(N_PEOPLE)), help="for example 0-31")
    ap.add_argument("--tag", default="local", help="names this machine's file, runs_<tag>.csv")
    _design_args(ap)
    a = ap.parse_args()
    if not a.replot:
        save_conditions(D)
        run_pool(_one, a.people, OUT / f"runs_{a.tag}.csv", fresh=True,
                 label=lambda rows: f"person {rows[0]['person']}")
    files = sorted(OUT.glob("runs_*.csv"))
    summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
