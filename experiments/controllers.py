"""Measuring against planning for reward: six ways of choosing settings, on the same people.

The 32 simulated people of the 4 x 4 grid of f, k in {0.2, 0.6, 1.0, 1.5}, 2 seeds each,
question-type task, skips on, 300 ticks (a tick is a start decision in a session, or a
chance to come back while away; a person who stops more gives fewer decisions), true f and
k hidden from every planner. Current model (V4): gamma 0.9, ACHIEVEMENT_SCALE 4,
continuing bootstrap, T_DUR 7.0 s.

Arms:
    fixed             the default setting throughout
    random            a random setting every 25 start decisions, from the 84 scheduled-review settings
    random_redo       the same over all 96 (redo-prompt settings included)
    reward_scheduled  reward planning over the 84 (the app's controller until 2026-10-11)
    reward            reward planning over all 96
    information       InfoController over all 96 (the app's controller)

Each run records, at 100, 200 and 300 ticks: f and k fitted from the start and redo
decisions recorded so far, the spread (sd) of cost in that data, and the reward and
P(start next set) of the block of 100 ticks just ended.

Pass conditions, fixed before the run (paired on person and seed, at the end of the run):
  P1  squared error in k, information - reward, is below zero by more than twice its
      standard error, and squared error in f is not above zero by more than twice its.
  P2  reward, information - fixed, is above zero by more than twice its standard error.
Everything else is reported without a claim.

README: "It found the redo prompts on its own", "Measuring costs no detectable reward",
"Over time" and "What measuring costs, block by block". Paper Figure 8.

    python -m experiments.controllers            # run (hours), then summarize
    python -m experiments.controllers --replot   # summarize the saved results
"""
import numpy as np
import pandas as pd

from temporal.controller import SCHEDULED, Controller, Environment, InfoController, run
from temporal.inference import fit, passages_to_frame
from temporal.model.params import UserParams

from .common import LEVELS, Quiet, args, out_dir, run_pool, se

OUT = out_dir("controllers")
REPS = 2
N = 300
CHECKPOINTS = (100, 200, 300)
ARMS = ("fixed", "random", "random_redo", "reward_scheduled", "reward", "information")
KEY = ["f_true", "k_true", "seed"]


def controller(arm, f, k, seed):
    if arm == "fixed":
        return None
    if arm == "random":
        return Controller(candidates=SCHEDULED, explore=1.0, seed=seed, known=(f, k))
    if arm == "random_redo":
        # Its own random stream per person: with one stream per seed, as "random" has,
        # both streams happened never to draw one of the 12 redo settings.
        return Controller(explore=1.0, seed=1000 * seed + int(f * 100) * 10 + int(k * 10),
                          known=(f, k))
    if arm == "reward_scheduled":
        return Controller(candidates=SCHEDULED, seed=seed)
    if arm == "reward":
        return Controller(seed=seed)
    return InfoController(seed=seed)


def _one(task):
    arm, f, k, seed = task
    ctl = controller(arm, f, k, seed)
    reward_at = {}

    def mark(sim):
        n = sim.state.t - 1  # ticks so far; the first set is free
        if n in CHECKPOINTS:
            reward_at.setdefault(n, sim.state.total_reward)

    with Quiet():
        sim = run(UserParams(f=f, k=k), controller=ctl, env=Environment(), n_ticks=N,
                  seed=seed, each=mark)
        starts, redos = sim.state.initiation_passages, sim.state.first_passage
        log = ctl.log if ctl else []
        out = dict(arm=arm, f_true=f, k_true=k, seed=seed, reward=sim.state.total_reward,
                   sets=sim.state.episodes, decisions=len(starts), away=sim.state.away / N,
                   p_go=float(np.mean([p["chose"] == 1 for p in starts])),
                   redo_prompts=len(redos),
                   redo_share=float(np.mean([u.chosen.redo_prompts for u in log])) if log else 0.0)
        prev = 0.0
        for i, n in enumerate(CHECKPOINTS):
            # Everything recorded in the first n ticks. A tick's set runs after it, at n + 1.
            block = [p for p in starts if p["timestep"] <= n]
            data = block + [r for r in redos if r["timestep"] <= n + 1]
            out[f"f_{n}"], out[f"k_{n}"] = fit(passages_to_frame(data))
            out[f"rows_{n}"] = len(data)
            out[f"csd_{n}"] = float(np.std([p["cost"] for p in data]))
            out[f"reward_{n}"] = reward_at[n] - prev
            prev = reward_at[n]
            lo = CHECKPOINTS[i - 1] if i else 0
            go = [p["chose"] == 1 for p in block if p["timestep"] > lo]
            out[f"pgo_{n}"] = float(np.mean(go)) if go else np.nan
    return out


def rmse(d, p, n=300):
    return float(np.sqrt(((d[f"{p}_{n}"] - d[f"{p}_true"]) ** 2).mean()))


def paired(res, a, b, col):
    x = res[res.arm == a].set_index(KEY)[col]
    y = res[res.arm == b].set_index(KEY)[col]
    d = (x - y).dropna()
    return float(d.mean()), se(d)


def summarize(res):
    res = res.assign(ek=(res.k_300 - res.k_true) ** 2, ef=(res.f_300 - res.f_true) ** 2)
    by = {arm: res[res.arm == arm] for arm in ARMS}

    print("== redo settings chosen, share of updates, and redo prompts per person ==")
    for arm in ARMS[1:]:
        print(f"{arm:>17}: {by[arm].redo_share.mean():.0%} of updates, "
              f"{by[arm].redo_prompts.mean():.0f} redo prompts")

    print(f"\n== after {N} ticks (32 people) ==")
    print(f"{'arm':>17}  f RMSE  k RMSE  reward  P(start)  decisions  away")
    for arm in ("fixed", "reward_scheduled", "reward", "information"):
        d = by[arm]
        print(f"{arm:>17}  {rmse(d, 'f'):6.3f}  {rmse(d, 'k'):6.3f}  {d.reward.mean():6.1f}  "
              f"{d.p_go.mean():8.2f}  {d.decisions.mean():9.0f}  {d.away.mean():4.0%}")

    print("\n== over time (ticks) ==")
    print(f"{'arm':>17}  k RMSE at 100 / 200 / 300   f RMSE at 100 / 200 / 300   sd of cost at 300"
          "   decisions in the fit at 100 / 200 / 300")
    for arm in ("fixed", "random", "reward_scheduled", "reward", "information"):
        d = by[arm]
        ks = " / ".join(f"{rmse(d, 'k', n):.3f}" for n in CHECKPOINTS)
        fs = " / ".join(f"{rmse(d, 'f', n):.3f}" for n in CHECKPOINTS)
        rows = " / ".join(f"{d[f'rows_{n}'].mean():.0f}" for n in CHECKPOINTS)
        print(f"{arm:>17}  {ks:>25}   {fs:>25}   {d.csd_300.mean():.3f}   {rows}")

    print("\n== block by block ==")
    print(f"{'arm':>17}  reward, ticks 1-100 / 101-200 / 201-300   P(start), the same blocks")
    for arm in ("fixed", "random", "reward_scheduled", "reward", "information"):
        d = by[arm]
        rw = " / ".join(f"{d[f'reward_{n}'].mean():.1f}" for n in CHECKPOINTS)
        pg = " / ".join(f"{d[f'pgo_{n}'].mean():.2f}" for n in CHECKPOINTS)
        print(f"{arm:>17}  {rw:>30}   {pg:>30}")
    print("information - reward planning (84 / 96 settings), per block:")
    for n in CHECKPOINTS:
        parts = []
        for col, fmt in ((f"reward_{n}", "{:+.1f} ± {:.1f}"), (f"pgo_{n}", "{:+.3f} ± {:.3f}")):
            parts.append(" / ".join(fmt.format(*paired(res, "information", o, col))
                                    for o in ("reward_scheduled", "reward")))
        print(f"  to {n}: reward {parts[0]};  P(start) {parts[1]}")

    print("\n== P1: squared error, information - reward (96 settings) ==")
    mk, sk = paired(res, "information", "reward", "ek")
    mf, sf = paired(res, "information", "reward", "ef")
    print(f"k {mk:+.4f} ± {sk:.4f} (2 SE {2 * sk:.4f});  f {mf:+.4f} ± {sf:.4f}")
    print("P1 PASSES" if mk < -2 * sk and mf <= 2 * sf else "P1 DOES NOT PASS")
    m, e = paired(res, "information", "reward_scheduled", "ek")
    print(f"k against the previous controller (84 settings): {m:+.4f} ± {e:.4f} ({-m / e:.1f} SE)")
    print("\n== P2: reward, information - fixed ==")
    m, e = paired(res, "information", "fixed", "reward")
    print(f"{m:+.1f} ± {e:.1f} -> {'P2 PASSES' if m > 2 * e else 'P2 DOES NOT PASS'}")

    print("\n== reported without a claim ==")
    for a, b in (("information", "reward"), ("information", "reward_scheduled"),
                 ("reward", "reward_scheduled")):
        print(f"{a} - {b}: reward {'{:+.1f} ± {:.1f}'.format(*paired(res, a, b, 'reward'))}; "
              f"k squared error {'{:+.4f} ± {:.4f}'.format(*paired(res, a, b, 'ek'))}")
    for arm in ("reward_scheduled", "reward", "information"):
        low = by[arm][by[arm].k_300 < 0.05]
        print(f"{arm}: k below 0.05 for {len(low)} of 32 (true k {sorted(set(low.k_true))})")


if __name__ == "__main__":
    a = args(__doc__)
    if a.replot:
        res = pd.read_csv(OUT / "runs.csv")
    else:
        tasks = [(arm, f, k, s) for arm in ARMS for f in LEVELS for k in LEVELS for s in range(REPS)]
        tasks.sort(key=lambda t: t[0] in ("fixed", "random", "random_redo"))  # slow arms first
        res = run_pool(_one, tasks, OUT / "runs.csv",
                       label=lambda r: f"{r['arm']} f {r['f_true']} k {r['k_true']} seed {r['seed']}")
    summarize(res)
