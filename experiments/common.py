"""What the experiments share: where results go, the grid of people, the SE rule, and the
conditions each experiment runs under."""
import argparse
import ast
import contextlib
import datetime
import io
import json
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LEVELS = (0.2, 0.6, 1.0, 1.5)  # the 4 x 4 grid of f and k
# All but one core, or TEMPORAL_WORKERS (for a machine with less memory per core).
WORKERS = int(os.environ.get("TEMPORAL_WORKERS", 0)) or max(1, (os.cpu_count() or 2) - 1)


def out_dir(name):
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


# An experiment's conditions: the paper's, unless this run is that experiment and was
# started with `--set name=value` (any number of them) or `--out folder`. Read from the
# command line when the module is imported, so module constants built from them hold in
# worker processes too (a spawned worker gets the parent's sys.argv back before it
# imports anything). A run under changed conditions must name its own folder, so the
# paper's results are never overwritten.
def design(name, **paper):
    d = SimpleNamespace(name=name, out=name, paper=dict(paper), **paper)
    script = Path(sys.argv[0])
    if script.stem != name or script.parent.name != "experiments":
        return d
    p = argparse.ArgumentParser(add_help=False)
    _design_args(p)
    given, _ = p.parse_known_args()
    for item in given.set or []:
        key, _, value = item.partition("=")
        if key not in paper:
            sys.exit(f"--set {key}: {name} has no condition {key!r}; it has {', '.join(paper)}")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # a bare word, such as a folder name
        if isinstance(paper[key], tuple):
            value = tuple(value) if isinstance(value, (list, tuple)) else (value,)
        setattr(d, key, value)
    changed = {k: getattr(d, k) for k in paper if getattr(d, k) != paper[k]}
    if changed and not given.out:
        sys.exit(f"{name}: conditions changed from the paper's ({', '.join(changed)}); "
                 f"name a folder for the results with --out, so results/{name}/ is kept")
    if given.out:
        d.out = given.out
    return d


def _design_args(p):
    p.add_argument("--set", action="append", metavar="NAME=VALUE",
                   help="change one condition from the paper's, for example --set reps=4 or "
                        "--set 'people=[(0.8, 0.8)]'; needs --out")
    p.add_argument("--out", help="save to results/OUT/ instead of results/<experiment>/")


# The fixed constants a caption states besides an experiment's own conditions.
def model_constants():
    from temporal.model import accumulator as acc
    from temporal.model import value
    from temporal.model.params import UserParams
    return {"g": UserParams.g, "a": UserParams.a, "ACHIEVEMENT_SCALE": value.ACHIEVEMENT_SCALE,
            "bootstrap": "continuing" if value.CONTINUING else "terminal",
            "SIGMA": acc.SIGMA, "BOUND": acc.BOUND, "T_DUR": acc.T_DUR}


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# Writes results/<out>/conditions.json, which `python -m paper` reads for its captions.
def save_conditions(d, note=None):
    conditions = {k: getattr(d, k) for k in d.paper}
    record = {"experiment": d.name, "conditions": conditions,
              "changed": {k: v for k, v in conditions.items() if v != d.paper[k]},
              "model": model_constants(), "date": datetime.date.today().isoformat(),
              "commit": _commit(), "command": " ".join(["python -m", f"experiments.{d.name}",
                                                        *sys.argv[1:]])}
    if note:
        record["note"] = note
    path = out_dir(d.out) / "conditions.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return path


# --replot: summarize the saved results instead of running again. --set and --out change
# the conditions (see `design`); the conditions are saved before a run starts.
def args(doc, d=None):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true", help="re-read the saved results, don't re-run")
    if d is not None:
        _design_args(p)
    a = p.parse_args()
    if d is not None and not a.replot:
        save_conditions(d)
    return a


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
