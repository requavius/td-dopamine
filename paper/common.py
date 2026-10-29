"""Where the figures read from and write to."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"   # written by `python -m experiments.<name>`
OUT = ROOT / "paper" / "figures"

# Experiment name -> the results folder to draw it from, when not results/<name>/ (the
# paper's). Set by `python -m paper --results name=folder`.
SOURCES = {}
_out = OUT


def results(name):
    return RESULTS / SOURCES.get(name, name)


# The conditions an experiment's results were generated under (experiments.common
# writes them), or None for a folder without them.
def conditions(name):
    path = results(name) / "conditions.json"
    return json.loads(path.read_text()) if path.exists() else None


# Draw from other results into paper/figures/<folder>/, so the paper's figures are kept.
def use(sources, folder=None):
    global _out
    SOURCES.update(sources)
    _out = OUT / folder if folder else OUT / "_".join(sources.values()) if sources else OUT


def out(stem=""):
    return _out / stem


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))
