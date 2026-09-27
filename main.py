import argparse
import json

import pandas as pd

from config import COST_PROFILES, UserParams, value_at_choice_point
from inference import DEADLINE_TOLERANCE, audit_rts, sanity_check, sanity_check_coupled
from recovery import (
    DESIGN,
    GRID,
    PATIENCE_GRID,
    PERSISTENCE_GRID,
    T0_GRID,
    collect_coupled,
    collect_single,
    recovery,
    recovery_coupled,
    recovery_patience,
    recovery_persistence,
    recovery_reseeded,
    recovery_single,
    recovery_t0,
    simulate_dataset,
)
from temporal_difference_model import REVIEW_MODES, Simulation


def run_experiment(params: UserParams, cost=None, max_decisions=200, debug=False,
                   review_mode="scheduled", review_rate=0.25, diff_spread=0.0):
    sim = Simulation(params, reentry_cost=cost, review_mode=review_mode,
                     review_rate=review_rate, diff_spread=diff_spread)
    state = sim.run(max_decisions=max_decisions)

    passages = pd.DataFrame(state.initiation_passages)
    stages = pd.DataFrame(state.first_passage)
    go_rate = (passages["GO"] == 1).mean() if len(passages) else float("nan")
    repeat_rate = stages["resolved"].mean() if len(stages) else float("nan")

    if debug:
        print(f"true params: f={params.f:.3f} k={params.k:.3f}")
        print(f"cold-start cost: {state.reentry_cost} (warm decisions cost far less)")
        print(f"cost actually faced: {passages['cost'].min():.2f}-{passages['cost'].max():.2f} "
              f"(sd {passages['cost'].std():.3f}), in-session on {passages['in_session'].mean():.0%} of decisions")
        print(f"ticks={state.t} episodes={state.episodes} stuck={state.stuck}")
        print(f"review: mode={review_mode} rate={review_rate}, "
              f"{passages['last_review_load'].mean():.0%} of slots on average, "
              f"{passages['last_episode_passed'].mean():.0%} of questions right")
        print(f"corrections={state.corrections} missed backlog={len(state.missed)} "
              f"ability={state.ability:.2f}")
        if review_mode == "choice":
            print(f"redo offers={len(stages)} P(accept)={repeat_rate:.3f}")
        print(f"initiation decisions={len(passages)} P(GO)={go_rate:.3f}")
        print(f"V at re-entry: {value_at_choice_point(state):.3f}")
        print("RPE:", [round(float(state.rpe[s]), 3) for s in range(state.stage_amt)])
        if len(passages):
            print("\nlast 10 re-entry decisions:")
            print(passages.tail(10).to_string(index=False))

    return {
        "true_f": params.f,
        "true_k": params.k,
        "reentry_cost": state.reentry_cost,
        "abandoned": state.abandoned,
        "ticks": state.t,
        "episodes": state.episodes,
        "stuck": state.stuck,
        "decisions": len(passages),
        "p_go": go_rate,
        "stage_draws": len(stages),
        "p_repeat": repeat_rate,
        "stages_passed": passages["last_episode_passed"].mean() if len(passages) else float("nan"),
        "corrections": state.corrections,
        "missed": len(state.missed),
        "review_load": passages["last_review_load"].mean() if len(passages) else float("nan"),
    }


def cmd_run(args):
    params = UserParams()
    if args.values:
        params.f, params.k = map(float, args.values.split(",")[:2])

    run_experiment(
        params,
        cost=COST_PROFILES[args.cost],
        max_decisions=args.decisions,
        debug=True,
        review_mode=args.review_mode,
        review_rate=args.review_rate,
        diff_spread=args.diff_spread,
    )

def _report(stats, label):
    print(f"\n{label}: " + "  ".join(
        f"{name}: r = {s['r']:.3f}, RMSE = {s['rmse']:.3f}"
        for name, s in stats.items()
    ))


def cmd_recovery(args):
    if args.mode in ("single", "all"):
        print("=== single-run sanity check (f = 0.8, k = 0.8) ===")
        demo = collect_single(0.8, 0.8)
        print(f"{len(demo)} rows")
        sanity_check_coupled(demo)

        print(f"\n=== single-run recovery ({len(GRID)} points x {args.reps} reps) ===")
        res = recovery_single(GRID, reps=args.reps)
        _report(res.attrs["stats"], "single-run")

    if args.mode in ("uncoupled", "all"):
        print("=== uncoupled sanity check (f = 0.8, k = 0.8) ===")
        demo = simulate_dataset(0.8, 0.8, DESIGN, n_per_cell=200)
        print(f"{len(demo)} rows")
        sanity_check(demo)

        print(f"\n=== uncoupled recovery ({len(GRID)} points x {args.reps} reps) ===")
        res = recovery(GRID, reps=args.reps)
        _report(res.attrs["stats"], "uncoupled")

    if args.mode in ("coupled", "all"):
        print("\n=== coupled sanity check (f = 0.8, k = 0.8) ===")
        demo = collect_coupled(0.8, 0.8)
        print(f"{len(demo)} rows")
        sanity_check_coupled(demo)

        print(f"\n=== coupled recovery ({len(GRID)} points x {args.reps} reps) ===")
        res = recovery_coupled(GRID, reps=args.reps)
        _report(res.attrs["stats"], "coupled")

    if args.mode in ("persistence", "all"):
        print(f"\n=== persistence recovery ({len(PERSISTENCE_GRID)} people x {args.reps} reps) ===")
        res = recovery_persistence(PERSISTENCE_GRID, reps=args.reps)
        for source, s in res.attrs["stats"].items():
            print(f"{source:>12}: f RMSE = {s['f']:.3f}  k RMSE = {s['k']:.3f}")

    if args.mode in ("patience", "all"):
        print(f"\n=== patience recovery ({len(PATIENCE_GRID)} people x {args.reps} reps) ===")
        res = recovery_patience(PATIENCE_GRID, reps=args.reps)
        for source, x in res.attrs["stats"].items():
            print(f"patience {source}: typically off by {x:.2f}x")

    if args.mode in ("t0", "all"):
        print(f"\n=== non-decision time ({len(T0_GRID)} values x {args.reps} reps) ===")
        res = recovery_t0(T0_GRID, reps=args.reps)
        print("\nmean over reps:")
        print(res.groupby("t0_true")[["f_pinned", "k_pinned", "f_free", "k_free", "t0_hat"]]
              .mean().round(3).to_string())

    if args.mode in ("reseeded", "all"):
        print(f"\n=== one person at four points on the learning curve "
              f"({len(GRID)} points x {args.reps} reps) ===")
        res = recovery_reseeded(GRID, reps=args.reps)
        _report(res.attrs["stats"], "reseeded")

def cmd_audit(args):
    events = []
    with open(args.path) as fh:
        for line in fh:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut off by a crash
    table = audit_rts([e for e in events if e.get("node") in ("start", "stage")])
    deadline = table.pop("deadline")
    print(f"decision times in {args.path}, against the model's deadline T_DUR = {deadline} s\n")
    for name, s in table.items():
        if not s["n"]:
            print(f"{name:>6}: none")
            continue
        print(f"{name:>6}: n {s['n']:3d}  min {s['min']:.2f}  median {s['median']:.2f}  "
              f"90% {s['q90']:.2f}  max {s['max']:.2f}  over the deadline {s['over']} "
              f"({s['share_over']:.1%})")
    share = table["all"].get("share_over", 0.0)
    if share > DEADLINE_TOLERANCE:
        print(f"\n{share:.1%} of decisions are past the deadline (tolerance "
              f"{DEADLINE_TOLERANCE:.0%}): the fit discards their choices. Move T_DUR in "
              f"initiation.py past {table['all']['max']:.2f} s.")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="simulate one learner (default)")
    p_run.add_argument("--values", help="f,k as comma separated floats; random if omitted")
    p_run.add_argument("--cost", choices=tuple(COST_PROFILES), default="mid",
                       help="cold-start cost, paid when resuming from outside a "
                            "session; warm decisions cost far less")
    p_run.add_argument("--review-mode", choices=REVIEW_MODES, default="scheduled",
                       help="who decides whether a missed question comes back: "
                            "the controller (scheduled), the learner (choice), "
                            "or nobody (quiet)")
    p_run.add_argument("--review-rate", type=float, default=0.25,
                       help="scheduled mode: chance each slot revisits a missed question")
    p_run.add_argument("--diff-spread", type=float, default=0.0,
                       help="per-question difficulty spread around the base difficulty")
    p_run.add_argument("--decisions", type=int, default=200,
                       help="how many initiation decisions to collect")
    p_run.set_defaults(func=cmd_run)

    p_rec = sub.add_parser("recovery", help="parameter recovery sweep and plot")
    p_rec.add_argument("--mode",
                       choices=("single", "coupled", "uncoupled", "persistence",
                                "patience", "reseeded", "t0", "all"),
                       default="single",
                       help="single: one learner, cost contrast from its own "
                            "sessions and review load (default). "
                            "coupled: two learners pooled at different cold-start "
                            "costs. uncoupled: value as a fixed design constant. "
                            "persistence: f and k from decisions, from "
                            "answer and skip times, and from both, on "
                            "human-sized data. patience: how quickly someone "
                            "gives up, estimated as the controller does. "
                            "reseeded: single-learner data split across four "
                            "learners at different points on the learning curve. "
                            "t0: what assuming away the non-decision time costs, "
                            "and whether freeing it gets the parameters back. "
                            "all: run every pipeline.")
    p_rec.add_argument("--reps", type=int, default=5,
                       help="repetitions per grid point")
    p_rec.set_defaults(func=cmd_recovery)

    p_audit = sub.add_parser("audit", help="human decision times against the model's deadline")
    p_audit.add_argument("--path", default="data/decisions.jsonl")
    p_audit.set_defaults(func=cmd_audit)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        args = parser.parse_args(["run"])

    args.func(args)
