# How to run: run with args --function "1-3" --values f,k,b
# Values should be between 0 and 1 and seperated by commas: 0.1,0.1,0.1
# If no value argument they will be random
# Function 3 and no function arg are the same
# Functions: 1: Particle Filter(no value arg), 2: Single DDM(Weiner process), 3: terminal stats for one run, 4: Multiple DDMs(no value arg)
# main.py --function 2 --value 0.1,0.1,0.1
# Parallelism: --workers N controls how many processes the parameter sweep uses
#   (defaults to os.cpu_count()). Use --workers 1 to run the sweep serially.
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pyddm

from config import UserParams, V
from ddm import loglikelihood
from temporal_difference_model import Simulation

FIT_PARAMS = True

def run_learner(sim: Simulation):
    state = sim.state
    low_rpe_streak = 0
    disengaged = False
    while not disengaged:
        disengaged = sim.simulate()

        max_rpe = max(abs(x) for x in state.rpe.values())

        if max_rpe < 0.05:
            low_rpe_streak += 1
        else:
            low_rpe_streak = 0

        average_v = sum(V(state, s) for s in range(state.stage_amt)) / state.stage_amt

        state.t += 1

        if (low_rpe_streak >= state.stage_amt and average_v > 0.1) or state.t >= 2000:
            return


def fit_params(state):
    _, fitted = loglikelihood(state)
    return {
        "f": float(fitted["drift"]["f_val"]),
        "k": float(fitted["bound"]["k_val"]),
        "b": float(fitted["drift"]["b_val"]),
    }


def run_experiment(params: UserParams, debug=False, extra=0, repeat=1, FIT_PARAMS = True):

    sim = Simulation(params)
    state = sim.state

    repeats = {}
    est = None
    for i in range(repeat):
        run_learner(sim)
        if FIT_PARAMS:
            print('fitting...')
            est = fit_params(state)
            repeats[i] = {**est, "t": state.t}

    if debug:
        print(f"stopped after {state.t} trials")
        print("V:", [round(V(state, s), 3) for s in range(state.stage_amt)])
        print("RPE:", [round(state.rpe[s], 3) for s in range(state.stage_amt)])
        if est is not None:
            print("Estimated params:", est)
        print("True params:", sim.params)

    if not extra:
        result = {
            "true_f": params.f,
            "true_k": params.k,
            "true_b": params.b,
            "trials": state.t,
        }
        if est is not None:
            result["est_f"] = est["f"]
            result["est_k"] = est["k"]
            result["est_b"] = est["b"]
        return result
    elif extra == 2:
        return repeats


def _init_worker():
    pyddm.set_N_cpus(1)


def _single_run(job):
    param, val, fixed = job
    if param == "f":
        res = run_experiment(val, fixed, fixed)
    elif param == "k":
        res = run_experiment(fixed, val, fixed)
    else:
        res = run_experiment(fixed, fixed, val)
    return param, val, res


def collect_results(n=60, repeats=1, workers=None):
    if not FIT_PARAMS:
        raise RuntimeError(
            "Parameter recovery sweep requires fitting; set FIT_PARAMS = True."
        )
    sweep = np.linspace(0.05, 0.95, n)
    fixed = 0.5

    jobs = []
    for val in sweep:
        for param in ("f", "k", "b"):
            for _ in range(repeats):
                jobs.append((param, float(val), fixed))

    runs = {}
    total = len(jobs)

    print(f"Starting sweep: {total} runs over {n} values "
          f"({'serial' if workers == 1 else f'{workers or os.cpu_count()} workers'})",
          flush=True)

    def _record(i, param, val, res):
        runs.setdefault((param, round(val, 6)), []).append(res)
        print(f"[{i}/{total}] {param}={val:.3f} -> "
              f"trials={res['trials']}, est_{param}={res[f'est_{param}']:.3f}",
              flush=True)

    if workers == 1:
        for i, job in enumerate(jobs, start=1):
            _record(i, *_single_run(job))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futures = [ex.submit(_single_run, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), start=1):
                _record(i, *fut.result())

    print(f"Sweep complete ({total} runs). Aggregating and plotting...", flush=True)

    results = []
    for val in sweep:
        for param in ("f", "k", "b"):
            rs = runs[(param, round(float(val), 6))]
            results.append(
                {
                    "param": param,
                    "true_f": val if param == "f" else fixed,
                    "true_k": val if param == "k" else fixed,
                    "true_b": val if param == "b" else fixed,
                    "avg_trials": np.mean([r["trials"] for r in rs]),
                    "est_f": np.mean([r["est_f"] for r in rs]),
                    "est_k": np.mean([r["est_k"] for r in rs]),
                    "est_b": np.mean([r["est_b"] for r in rs]),
                }
            )

    return results


def plot_results(results):

    f_est = [(r["true_f"], r["est_f"]) for r in results if r["param"] == "f"]
    k_est = [(r["true_k"], r["est_k"]) for r in results if r["param"] == "k"]
    b_est = [(r["true_b"], r["est_b"]) for r in results if r["param"] == "b"]

    f_sweept = [(r["true_f"], r["avg_trials"]) for r in results if r["param"] == "f"]
    k_sweept = [(r["true_k"], r["avg_trials"]) for r in results if r["param"] == "k"]
    b_sweept = [(r["true_b"], r["avg_trials"]) for r in results if r["param"] == "b"]

    _, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))

    lims = [0.05, 0.95]
    ax2.plot(lims, lims, "k--", alpha=0.4, label="ideal recovery")
    ax2.scatter(*zip(*sorted(f_est)), color="#2196F3", s=15, alpha=0.7, label="est f")
    ax2.scatter(*zip(*sorted(k_est)), color="#FF5722", s=15, alpha=0.7, label="est k")
    ax2.scatter(*zip(*sorted(b_est)), color="#4CAF50", s=15, alpha=0.7, label="est b")
    ax2.set_xlabel("True parameter value")
    ax2.set_ylabel("Estimated parameter value")
    ax2.set_title("Parameter Recovery")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.legend()

    ax3.plot(*zip(*sorted(k_sweept)), color="#FF5722", label="k (effort aversion)")
    ax3.plot(*zip(*sorted(f_sweept)), color="#2196F3", label="f (progress sensitivity)")
    ax3.plot(*zip(*sorted(b_sweept)), color="#4CAF50", label="b (boredom rate)")
    ax3.set_xlabel("Parameter value")
    ax3.set_ylabel("Average Trials completed")
    ax3.set_title("Parameter vs Engagement")
    ax3.legend()

    plt.tight_layout()

    plt.savefig("assets/engagement_by_params.png", dpi=150, bbox_inches="tight")
    plt.show()


def particlesovertime(params):
    data = run_experiment(params, False, 2, 5)
    colors = {"f": "#FF0000", "k": "#2200FF", "b": "#00FF4C"}
    x = sorted(data.keys())

    for key, color in colors.items():
        y = [data[xi][key] for xi in x]
        plt.plot(x, y, marker="o", label=key, color=color)

    plt.xlabel("Trial Num")
    plt.ylabel("Value")
    plt.legend()
    plt.title("")
    plt.show()


def run(args):
    up = UserParams()
    if args.function == "3":
        workers = args.workers if args.workers else None
        plot_results(collect_results(10, 1, workers=workers))
        sys.exit()
    if args.function in ('1', '2'):
        params = args.values if args.values else ""
        if params != "":
            up.f, up.k, up.b = map(float, params.split(","))
        if args.function == "1":
            run_experiment(up, debug=True)
        if args.function == "2":
            particlesovertime(up)
    else:
        run_experiment(up, debug=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--function", default="1")
    parser.add_argument("--values")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="processes for the parameter sweep (default: all cores, 1 = serial)",
    )
    args = parser.parse_args()

    run(args)
