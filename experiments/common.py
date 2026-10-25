"""What the experiments share: where results go, the grid of people, and the SE rule."""
import argparse
import contextlib
import io
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LEVELS = (0.2, 0.6, 1.0, 1.5)  # the 4 x 4 grid of f and k
WORKERS = max(1, (os.cpu_count() or 2) - 1)


def out_dir(name):
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


# --replot: summarize the saved results instead of running again.
def args(doc):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="re-read the saved results, don't re-run")
    return p.parse_args()


# Run `fn` over `tasks` in parallel, saving the rows to `path` as they come in. `fn`
# returns one row (a dict) or a list of rows. `fresh` starts a new process for every task,
# so nothing one task cached stays in memory for the next.
def run_pool(fn, tasks, path, chunksize=1, label=None, fresh=False):
    rows = []
    with Pool(WORKERS, maxtasksperchild=1 if fresh else None) as pool:
        for i, out in enumerate(pool.imap_unordered(fn, tasks, chunksize=chunksize), start=1):
            rows += out if isinstance(out, list) else [out]
            if i % max(1, len(tasks) // 50) == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"[{i}/{len(tasks)}] {label(out) if label else ''}", flush=True)
    return pd.DataFrame(rows)


# Silences pyddm's progress output inside a fit.
class Quiet(contextlib.ExitStack):
    def __enter__(self):
        super().__enter__()
        self.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return self
