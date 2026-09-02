import argparse

import pandas as pd

from config import COST_PROFILES, UserParams, value_at_choice_point
from inference import sanity_check, sanity_check_coupled
from recovery import (
    DESIGN,
    GRID,
    collect_coupled,
    recovery,
    recovery_coupled,
    simulate_dataset,
)
from temporal_difference_model import Simulation

def run_experiment(params: UserParams, cost=None, max_decisions=200, debug=False):
    sim = Simulation(params, reentry_cost=cost)
    state = sim.run(max_decisions=max_decisions)

    passages = pd.DataFrame(state.initiation_passages)
    go_rate = (passages["GO"] == 1).mean() if len(passages) else float("nan")

    if debug:
        print(f"true params: f={params.f:.3f} k={params.k:.3f}")
        print(f"re-entry cost: {state.reentry_cost}")
        print(f"ticks={state.t} episodes={state.episodes} stuck={state.stuck}")
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
        "ticks": state.t,
        "episodes": state.episodes,
        "stuck": state.stuck,
        "decisions": len(passages),
        "p_go": go_rate,
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
    )

def _report(stats, label):
    print(f"\n{label}: " + "  ".join(
        f"{name}: r = {s['r']:.3f}, RMSE = {s['rmse']:.3f}"
        for name, s in stats.items()
    ))


def cmd_recovery(args):
    if args.mode in ("uncoupled", "both"):
        print("=== uncoupled sanity check (f = 0.8, k = 0.8) ===")
        demo = simulate_dataset(0.8, 0.8, DESIGN, n_per_cell=200)
        print(f"{len(demo)} rows")
        sanity_check(demo)

        print(f"\n=== uncoupled recovery ({len(GRID)} points x {args.reps} reps) ===")
        res = recovery(GRID, reps=args.reps)
        _report(res.attrs["stats"], "uncoupled")

    if args.mode in ("coupled", "both"):
        print("\n=== coupled sanity check (f = 0.8, k = 0.8) ===")
        demo = collect_coupled(0.8, 0.8)
        print(f"{len(demo)} rows")
        sanity_check_coupled(demo)

        print(f"\n=== coupled recovery ({len(GRID)} points x {args.reps} reps) ===")
        res = recovery_coupled(GRID, reps=args.reps)
        _report(res.attrs["stats"], "coupled")

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="simulate one learner (default)")
    p_run.add_argument("--values", help="f,k as comma separated floats; random if omitted")
    p_run.add_argument("--cost", choices=tuple(COST_PROFILES), default="low")
    p_run.add_argument("--decisions", type=int, default=200,
                       help="how many initiation decisions to collect")
    p_run.set_defaults(func=cmd_run)

    p_rec = sub.add_parser("recovery", help="parameter recovery sweep and plot")
    p_rec.add_argument("--mode", choices=("coupled", "uncoupled", "both"),
                       default="coupled",
                       help="coupled: value from the TD learner (default). "
                            "uncoupled: value as a fixed design constant.")
    p_rec.add_argument("--reps", type=int, default=5,
                       help="repetitions per grid point")
    p_rec.set_defaults(func=cmd_recovery)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        args = parser.parse_args(["run"])

    args.func(args)
