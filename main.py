# How to run: run with args --function "1-4" --values f,k,b
# Values should be between 0 and 1 and seperated by commas: 0.1,0.1,0.1
# If no value argument they will be random
# Function 3 and no function arg are the same
# Functions: 1: Particle Filter(no value arg), 2: Single DDM(Weiner process), 3: terminal stats for one run, 4: Multiple DDMs(no value arg)
# main.py --function 2 --value 0.1,0.1,0.1

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from temporal_difference_model import test_train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--function", default="3")
parser.add_argument("--values")
args = parser.parse_args()

def collect_results(n=60, repeats=5):
    results = []
    sweep = np.linspace(0.05, 0.95, n)
    fixed = 0.5

    for i, val in enumerate(sweep):
        f_runs = [test_train(val, fixed, fixed) for _ in range(repeats)]
        k_runs = [test_train(fixed, val, fixed) for _ in range(repeats)]
        b_runs = [test_train(fixed, fixed, val) for _ in range(repeats)]
        results.append({'param': 'f', 'true_f': val, 'true_k': fixed, 'true_b': fixed,
                        'avg_stages': np.mean([r['avg_stages'] for r in f_runs]),
                        'avg_trials': np.mean([r['trials'] for r in f_runs]),
                        'est_f': np.mean([r['est_f'] for r in f_runs]),
                        'est_k': np.mean([r['est_k'] for r in f_runs]),
                        'est_b': np.mean([r['est_b'] for r in f_runs]),
                        })
        results.append({'param': 'k', 'true_f': fixed, 'true_k': val, 'true_b': fixed,
                        'avg_stages': np.mean([r['avg_stages'] for r in k_runs]),
                        'avg_trials': np.mean([r['trials'] for r in k_runs]),
                        'est_f': np.mean([r['est_f'] for r in k_runs]),
                        'est_k': np.mean([r['est_k'] for r in k_runs]),
                        'est_b': np.mean([r['est_b'] for r in k_runs]),
                        })
        results.append({'param': 'b', 'true_f': fixed, 'true_k': fixed, 'true_b': val,
                        'avg_stages': np.mean([r['avg_stages'] for r in b_runs]),
                        'avg_trials': np.mean([r['trials'] for r in b_runs]),
                        'est_f': np.mean([r['est_f'] for r in b_runs]),
                        'est_k': np.mean([r['est_k'] for r in b_runs]),
                        'est_b': np.mean([r['est_b'] for r in b_runs]),
                        })
        print(f"completed {i+1}/{n}")

    return results

def plot_results(results):
    f_sweep = [(r['true_f'], r['avg_stages']) for r in results if r['param'] == 'f']
    k_sweep = [(r['true_k'], r['avg_stages']) for r in results if r['param'] == 'k']
    b_sweep = [(r['true_b'], r['avg_stages']) for r in results if r['param'] == 'b']

    f_est = [(r['true_f'], r['est_f']) for r in results if r['param'] == 'f']
    k_est = [(r['true_k'], r['est_k']) for r in results if r['param'] == 'k']
    b_est = [(r['true_b'], r['est_b']) for r in results if r['param'] == 'b']
    
    f_sweept = [(r['true_f'], r['avg_trials']) for r in results if r['param'] == 'f']
    k_sweept = [(r['true_k'], r['avg_trials']) for r in results if r['param'] == 'k']
    b_sweept = [(r['true_b'], r['avg_trials']) for r in results if r['param'] == 'b']

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.plot(*zip(*sorted(k_sweep)), color='#FF5722', label='k (effort aversion)')
    ax1.plot(*zip(*sorted(f_sweep)), color='#2196F3', label='f (progress sensitivity)')
    ax1.plot(*zip(*sorted(b_sweep)), color='#4CAF50', label='b (boredom rate)')
    ax1.set_xlabel('Parameter value')
    ax1.set_ylabel('Average stages completed')
    ax1.set_title('Parameter vs Engagement')
    ax1.legend()

    lims = [0.05, 0.95]
    ax2.plot(lims, lims, 'k--', alpha=0.4, label='ideal recovery')
    ax2.scatter(*zip(*sorted(f_est)), color='#2196F3', s=15, alpha=0.7, label='est f')
    ax2.scatter(*zip(*sorted(k_est)), color='#FF5722', s=15, alpha=0.7, label='est k')
    ax2.scatter(*zip(*sorted(b_est)), color='#4CAF50', s=15, alpha=0.7, label='est b')
    ax2.set_xlabel('True parameter value')
    ax2.set_ylabel('Estimated parameter value')
    ax2.set_title('Parameter Recovery')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.legend()
    
    ax3.plot(*zip(*sorted(k_sweept)), color='#FF5722', label='k (effort aversion)')
    ax3.plot(*zip(*sorted(f_sweept)), color='#2196F3', label='f (progress sensitivity)')
    ax3.plot(*zip(*sorted(b_sweept)), color='#4CAF50', label='b (boredom rate)')
    ax3.set_xlabel('Parameter value')
    ax3.set_ylabel('Average Trials completed')
    ax3.set_title('Parameter vs Engagement')
    ax3.legend()

    plt.tight_layout()

    plt.savefig('engagement_by_params.png', dpi=150, bbox_inches='tight')
    plt.show()

def plotweiner(f,k,b):
    fixed, log, t = test_train(f,k,b, debug=False, extra=True)
    x = [ep['dt'] for ep in log]
    y = [ep['position'] for ep in log]
    plt.plot(x,y,color='#FF5722')
    
    plt.axhline(y=1/fixed.k, color="#000000", linestyle='-', linewidth=2,alpha=1)
    plt.xlabel("time")
    plt.ylabel("position")
    params = {k: round(v, 2) for k, v in asdict(fixed).items()}
    plt.title(f"scatter for {t} trials\nTrue Params: {params}")
    plt.show()
def multiweiner(repeats=10):
    values = [0.1, 0.5, 0.9]
    colors = ['#2196F3', '#FF5722', '#4CAF50']
    fixed = 0.5
    param_names = ['f (progress sensitivity)', 'k (effort aversion)', 'b (boredom rate)']

    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (param_label, run_fn) in zip(axes, [
        ('f', lambda v: test_train(v, fixed, fixed, extra=True)),
        ('k', lambda v: test_train(fixed, v, fixed, extra=True)),
        ('b', lambda v: test_train(fixed, fixed, v, extra=True)),
    ]):
        drawn_labels = set()
        for val, color in zip(values, colors):
            for _ in range(repeats):
                fixed_params, log, _ = run_fn(val)
                boundary = 1 / fixed_params.k
                positions = [ep['position'] for ep in log]

                # trim to first boundary crossing so every line ends at the boundary
                cross = next((i for i, p in enumerate(positions) if p >= boundary), len(positions) - 1)
                traj = positions[:cross + 1]

                label = f'{param_label}={val}' if val not in drawn_labels else None
                ax.plot(traj, color=color, alpha=0.4, linewidth=0.9, label=label)
                drawn_labels.add(val)

            ax.axhline(y=boundary, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('stage step')
        ax.set_ylabel('position')
        ax.legend()

    axes[0].set_title(param_names[0])
    axes[1].set_title(param_names[1])
    axes[2].set_title(param_names[2])
    

    plt.tight_layout()
    plt.savefig('multiweiner_frozen_0.5.png', dpi=150, bbox_inches='tight')
    plt.show()


def run():
    f = k = b = choose = None
    if args.function == '1':
        plot_results(collect_results(60,5))
        quit()
    if args.function == '4':
        multiweiner()
        quit()
    if args.function != '1':
        params = args.values if args.values else ''
        if params != "": f,k,b = map(float, params.split(","))
        choose = str(int(args.function) - 1)
        if choose == "1": plotweiner(f,k,b)
        else: test_train(f,k,b, debug=True)
    else:
        test_train(f,k,b, debug=True)
run()

