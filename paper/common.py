"""Where the figures read from and write to."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"   # written by `python -m experiments.<name>`
OUT = ROOT / "paper" / "figures"


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))
