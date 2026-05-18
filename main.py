# How to run: run with args --function "1-4" --values f,k,b
# Values should be between 0 and 1 and seperated by commas: 0.1,0.1,0.1
# If no value argument they will be random
# Function 3 and no function arg are the same
# Functions: 1: Particle Filter(no value arg), 2: Single DDM(Weiner process), 3: terminal stats for one run, 4: Multiple DDMs(no value arg)
# main.py --function 2 --value 0.1,0.1,0.1
import pyddm, pyddm.plot
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import asdict
from temporal_difference_model import test_train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--function", default="1")
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
                        'avg_trials': np.mean([r['trials'] for r in f_runs]),
                        'est_f': np.mean([r['est_f'] for r in f_runs]),
                        'est_k': np.mean([r['est_k'] for r in f_runs]),
                        'est_b': np.mean([r['est_b'] for r in f_runs]),
                        })
        results.append({'param': 'k', 'true_f': fixed, 'true_k': val, 'true_b': fixed,
                        'avg_trials': np.mean([r['trials'] for r in k_runs]),
                        'est_f': np.mean([r['est_f'] for r in k_runs]),
                        'est_k': np.mean([r['est_k'] for r in k_runs]),
                        'est_b': np.mean([r['est_b'] for r in k_runs]),
                        })
        results.append({'param': 'b', 'true_f': fixed, 'true_k': fixed, 'true_b': val,
                        'avg_trials': np.mean([r['trials'] for r in b_runs]),
                        'est_f': np.mean([r['est_f'] for r in b_runs]),
                        'est_k': np.mean([r['est_k'] for r in b_runs]),
                        'est_b': np.mean([r['est_b'] for r in b_runs]),
                        })
        print(f"completed {i+1}/{n}")

    return results

def plot_results(results):

    f_est = [(r['true_f'], r['est_f']) for r in results if r['param'] == 'f']
    k_est = [(r['true_k'], r['est_k']) for r in results if r['param'] == 'k']
    b_est = [(r['true_b'], r['est_b']) for r in results if r['param'] == 'b']
    
    f_sweept = [(r['true_f'], r['avg_trials']) for r in results if r['param'] == 'f']
    k_sweept = [(r['true_k'], r['avg_trials']) for r in results if r['param'] == 'k']
    b_sweept = [(r['true_b'], r['avg_trials']) for r in results if r['param'] == 'b']

    _, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))

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

def particlesovertime(f,k,b):
    data = test_train(f,k,b, False, 2, 5)
    colors = {'f': "#FF0000", 'k': "#2200FF", 'b': "#00FF4C"}
    x = sorted(data.keys())

    for key, color in colors.items():
        y = [data[xi][key] for xi in x]
        plt.plot(x, y, marker='o', label=key, color=color)

    plt.xlabel('Trial Num')
    plt.ylabel('Value')
    plt.legend()
    plt.title('')
    plt.show()

def run():
    f = k = b = None
    if args.function == '3':
        plot_results(collect_results(60,5))
        quit()
    if args.function == "1" or "2":
        params = args.values if args.values else ''
        if params != "": f,k,b = map(float, params.split(","))
        if args.function == "1": 
            test_train(f,k,b, debug=True)
        if args.function == "2": particlesovertime(f,k,b)
    else:
        test_train(f,k,b, debug=True)
run()

