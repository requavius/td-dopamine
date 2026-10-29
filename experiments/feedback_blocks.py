"""Feedback in blocks as a controller setting, with redos mid-set or at the end of the set.

The controllers keep feedback on: redo prompts measure k best, and with feedback hidden a
miss is never shown, so no redo is offered. Hiding feedback is also what measures γ
("Feedback switched in blocks"). Two changes let a controller have both:

- **Blocks.** BlockInfoController is InfoController with feedback on a schedule: shown for
  one plan's 25 start decisions, hidden for the next, alternating. Within a block it
  chooses among the candidates with that block's feedback, as InfoController does.
- **Redos at the end of the set.** With feedback hidden, the misses are shown on the
  end-of-set screen with the score, and each can be redone there (Simulation `end_redos`,
  controller.REDO_END). The redo node is the same accumulator at the set's last slot: value
  is the undiscounted slot share 1/4 plus the error the score just produced (the whole
  set's), cost is 0.15 · min(1/p, 8) as always. A redo taken there is an extra question.

Three arms, paired on person and seed:
    current      InfoController as the app runs it (its 96 settings)
    blocks       BlockInfoController over the same 96: in hidden blocks a redo-prompt
                 setting asks nothing
    blocks_end   BlockInfoController with the 6 hidden redo-prompt settings replaced by
                 REDO_END: in hidden blocks the redos move to the end of the set

128 people from a seeded Latin hypercube over the ranges of experiments.recovery_lhc (f, k
0.2 to 1.5, γ 0.6 to 0.97, BOUND 1.0 to 2.0, T0 0 to 0.4 s), a seed of its own. Each plays
the app's task (question types with Elo ratings, skips on, cold-start cost 0.6) for 300
ticks under each arm. α 0.05 for person, shadows and controllers; SIGMA 0.7,
ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0. The simulated person sees the score
at the end of a hidden set, as the app shows it since 2026-10-28. At 100, 200 and 300
ticks the data recorded so far are fitted the same way in every arm: f, k, γ, BOUND and T0
by maximum likelihood (recovery_lhc.Person.fit), value at any γ from shadow learners.

Claims, fixed before any run (2026-10-28), paired on person, at 300 ticks:
    E1  squared error in k, blocks_end − blocks, is below zero beyond 2 SE
    E2  squared error in γ, blocks_end − current, is below zero beyond 2 SE
    E3  squared error in k, blocks_end − current, is not above zero beyond 2 SE
    E4  P(start next set), blocks_end − current, is not below zero beyond 2 SE
    E5  none: f, BOUND and T0; blocks − current; redo decisions, end-of-set redo
        decisions, reward, time away; corr(value, cost) over the decisions recorded

    python -m experiments.feedback_blocks --people 0-63 --tag server
    python -m experiments.feedback_blocks --replot
"""
import argparse
import random

import numpy as np
import pandas as pd
from scipy.stats import qmc

from temporal.controller import (CANDIDATES, REDO, REDO_END, SCHEDULED, InfoController,
                                 apply)
from temporal.model import accumulator as acc
from temporal.model.params import UserParams
from temporal.model.questions import Ratings, synthetic_type_diffs

from . import recovery_lhc
from .common import Quiet, _design_args, design, out_dir, run_pool, save_conditions, se
from .controller_full import _fit_records, _records_until
from .gamma_profile import CACHE_CAP, GridShadow
from .misspecification import PersonConstants
from .recovery_lhc import G_FINE, NAMES, span

# The run lasts until the last checkpoint. Its own seed; recovery_lhc's ranges.
D = design("feedback_blocks", people=128, seed=20261031, checkpoints=(100, 200, 300),
           ranges=recovery_lhc.D.paper["ranges"])
OUT = out_dir(D.out)
N_PEOPLE = D.people
SEED = D.seed
RANGES = D.ranges
CHECKPOINTS = D.checkpoints
N_TICKS = CHECKPOINTS[-1]
ARMS = ("current", "blocks", "blocks_end")
WITH_END = SCHEDULED + tuple(e for e in REDO if e.feedback) + REDO_END


def people():
    unit = qmc.LatinHypercube(d=len(RANGES), seed=SEED).random(N_PEOPLE)
    lo = np.array([r[0] for r in RANGES.values()])
    hi = np.array([r[1] for r in RANGES.values()])
    return [dict(zip(RANGES, row)) for row in lo + unit * (hi - lo)]


class BlockInfoController(InfoController):
    """InfoController with feedback on a schedule: on for the first plan's block of start
    decisions, off for the next, and so on. Within a block it chooses among the candidates
    with that block's feedback."""

    def __init__(self, candidates, **kw):
        super().__init__(candidates=candidates, **kw)
        self.all = tuple(candidates)

    def _plan(self, state, params):
        on = len(self.log) % 2 == 0
        self.candidates = tuple(c for c in self.all if c.feedback == on)
        return super()._plan(state, params)


def controller(arm, seed):
    if arm == "current":
        return InfoController(seed=seed)
    return BlockInfoController(CANDIDATES if arm == "blocks" else WITH_END, seed=seed)


def _one(task):
    arm, i = task
    truth = people()[i]
    seed = SEED + i
    acc.clear_caches()
    random.seed(seed)
    np.random.seed(seed)
    ctl = controller(arm, seed)
    start = ctl.env
    ratings = Ratings()
    person = PersonConstants(acc.SIGMA, truth["bound"])
    sim = GridShadow(UserParams(f=truth["f"], k=truth["k"], g=truth["g"], t0=truth["t0"]),
                     G_FINE, reentry_cost=0.6, review_rate=start.review_rate,
                     review_mode=start.review_mode, target_p=start.target_p,
                     feedback=start.feedback, ratings=ratings, skips=True,
                     return_skipped=start.return_skipped, end_redos=start.end_redos)
    st = sim.state
    st.type_diffs = np.array(synthetic_type_diffs(1.0), dtype=float)
    st.ability = 0.0
    st.in_session = True
    blocks = []
    with Quiet():
        with person():
            sim.run_episode()
        t_start = st.t
        mark = dict(reward=st.total_reward, away=st.away, starts=0, ticks=0)
        while st.t < t_start + N_TICKS:
            with person():
                sim.tick()
            person.trim()
            if len(acc._SOLUTION_CACHE) > CACHE_CAP:
                acc.clear_caches()
            if ctl.due(st):
                ctl.plan(sim.app_view(ratings), UserParams(f=truth["f"], k=truth["k"]))
                apply(sim, ctl.env)
            if st.t - t_start in CHECKPOINTS:
                n = st.t - t_start
                starts = st.initiation_passages[mark["starts"]:]
                blocks.append(dict(ticks=n, t=st.t, reward=st.total_reward - mark["reward"],
                                   away=(st.away - mark["away"]) / (n - mark["ticks"]),
                                   p_go=float(np.mean([p["chose"] == 1 for p in starts]))
                                   if starts else np.nan))
                mark = dict(reward=st.total_reward, away=st.away,
                            starts=len(st.initiation_passages), ticks=n)
        recs = st.initiation_passages + st.first_passage
        v = np.array([p["value"] for p in recs])
        c = np.array([p["cost"] for p in recs])
        redo = np.array([p.get("node") == "stage" for p in recs])
        row = dict(arm=arm, person=i, seed=seed, **{f"true_{n}": truth[n] for n in NAMES},
                   reward=sum(b["reward"] for b in blocks), away=st.away / N_TICKS,
                   decisions=len(st.initiation_passages), redos=len(st.first_passage),
                   end_redos=sum(bool(p.get("end")) for p in st.first_passage),
                   p_go=float(np.mean([p["chose"] == 1 for p in st.initiation_passages])),
                   feedback_share=float(np.mean([p["feedback"] for p in st.initiation_passages])),
                   corr_vc=float(np.corrcoef(v, c)[0, 1]) if len(recs) > 2 else np.nan,
                   corr_vc_redo=float(np.corrcoef(v[redo], c[redo])[0, 1])
                   if redo.sum() > 2 else np.nan,
                   target=float(np.mean([u.chosen.target_p for u in ctl.log])))
        for b in blocks:
            n = b["ticks"]
            part = _records_until(st, b["t"])
            row.update({f"reward_{n}": b["reward"], f"away_{n}": b["away"], f"pgo_{n}": b["p_go"],
                        f"rows_{n}": len(part)})
            est, nll = _fit_records(part, i) if len(part) >= 10 else (np.full(5, np.nan), np.nan)
            row.update({f"{name}_{n}": est[j] for j, name in enumerate(NAMES)})
            row[f"nll_{n}"] = nll
    return row


def _paired(a, b, col):
    d = (a[col] - b[col]).dropna()
    return d.mean(), se(d)


def _sq(x, name, n=300):
    return (x[f"{name}_{n}"] - x[f"true_{name}"]) ** 2


def summarize(df):
    arms = {arm: g.set_index("person").sort_index() for arm, g in df.groupby("arm")}
    common = sorted(set.intersection(*(set(g.index) for g in arms.values())))
    arms = {a: g.loc[common] for a, g in arms.items()}
    cur, blk, end = arms["current"], arms["blocks"], arms["blocks_end"]
    print(f"== feedback in blocks, redos mid-set or at the end: {len(common)} paired people, "
          f"{N_TICKS} ticks ==")
    print(f"{'RMSE at 100 / 200 / 300':>24} " + "".join(f"{a:>24}" for a in ARMS))
    for name in NAMES:
        cells = []
        for a in ARMS:
            x = arms[a]
            cells.append(" / ".join(f"{np.sqrt(np.nanmean(_sq(x, name, n))):.3f}" for n in CHECKPOINTS))
        print(f"{name:>24} " + "".join(f"{c:>24}" for c in cells))

    def claim(label, d, below=True, absent=False):
        m, e = d.mean(), se(d)
        ok = (m > -2 * e) if absent and not below else (m < 2 * e) if absent else (m < -2 * e)
        print(f"  {label}: {m:+.4f} ± {e:.4f} ({'holds' if ok else 'does not hold'})")
    print("\nclaims, paired at 300 ticks:")
    claim("E1 k sq. error, blocks_end − blocks", (_sq(end, "k") - _sq(blk, "k")).dropna())
    claim("E2 γ sq. error, blocks_end − current", (_sq(end, "g") - _sq(cur, "g")).dropna())
    claim("E3 k sq. error, blocks_end − current (not above)",
          (_sq(end, "k") - _sq(cur, "k")).dropna(), below=True, absent=True)
    claim("E4 P(start), blocks_end − current (not below)",
          (end.p_go - cur.p_go).dropna(), below=False, absent=True)

    print("\nE5 (descriptive), paired differences at 300 ticks:")
    for a, b, label in ((blk, cur, "blocks − current"), (end, cur, "blocks_end − current"),
                        (end, blk, "blocks_end − blocks")):
        cells = [f"{n} {(_sq(a, n) - _sq(b, n)).mean():+.4f} ± {se((_sq(a, n) - _sq(b, n)).dropna()):.4f}"
                 for n in NAMES]
        print(f"  sq. error, {label:<22} " + "  ".join(cells))
    print(f"\n{'mean per person':>16} " + "".join(f"{a:>12}" for a in ARMS))
    for col in ("reward", "away", "p_go", "decisions", "redos", "end_redos", "feedback_share",
                "target", "corr_vc", "corr_vc_redo"):
        print(f"{col:>16} " + "".join(f"{arms[a][col].mean():12.3f}" for a in ARMS))
    for a, b, label in ((end, cur, "blocks_end − current"), (blk, cur, "blocks − current")):
        cells = [f"{col} {_paired(a, b, col)[0]:+.3f} ± {_paired(a, b, col)[1]:.3f}"
                 for col in ("reward", "away", "p_go", "decisions")]
        print(f"  {label}: " + ", ".join(cells))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="merge and summarize the saved results")
    p.add_argument("--people", type=span, default=list(range(N_PEOPLE)), help="for example 0-63")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--tag", default="local", help="names this machine's file, runs_<tag>.csv")
    _design_args(p)
    a = p.parse_args()
    if not a.replot:
        save_conditions(D)
        tasks = [(arm, i) for i in a.people for arm in a.arms.split(",")]
        run_pool(_one, tasks, OUT / f"runs_{a.tag}.csv", fresh=True,
                 label=lambda r: f"{r['arm']} person {r['person']}")
    files = sorted(OUT.glob("runs_*.csv"))
    summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
