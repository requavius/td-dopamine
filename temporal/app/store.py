"""Everything that has to survive closing the app.

decisions.jsonl is the append-only record and is never rewritten; state.json is the
snapshot to resume from (written atomically); controller.pkl holds the controller with
its estimates.
"""
import dataclasses
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from ..model.params import ModelState

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Decision logs live in decisions.jsonl; state.json holds everything else.
LOGS = ("initiation_passages", "first_passage", "persistence")


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON-serialisable: {type(x).__name__}")


def state_to_dict(st):
    out = {}
    for f in dataclasses.fields(st):
        if f.name in LOGS:
            continue
        v = getattr(st, f.name)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif f.name == "rpe":
            v = {str(k): float(x) for k, x in v.items()}
        out[f.name] = v
    return out


def state_from_dict(d):
    names = {f.name for f in dataclasses.fields(ModelState)} - set(LOGS)
    st = ModelState(**{k: v for k, v in d.items() if k in names})
    st.theta = np.array(st.theta, dtype=float)
    for name in ("type_diffs", "stage_diffs"):
        if getattr(st, name) is not None:
            setattr(st, name, np.array(getattr(st, name), dtype=float))
    st.rpe = {int(k): v for k, v in st.rpe.items()}
    return st


# Whether the record's last session ended through the app: "I'm done" between sets or on
# a question, or a logged exit.
def closed_cleanly(events):
    last = events[-1]
    return (last.get("node") == "abandon" or (last.get("node") == "start" and last.get("GO") == 0)
            or (last.get("node") == "set_end" and bool(last.get("left"))))


class Store:
    def __init__(self, root=DATA_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "decisions.jsonl"  # append-only record, never rewritten
        self.state = self.root / "state.json"     # snapshot to resume from
        self.controller = self.root / "controller.pkl"

    def append(self, event):
        event.setdefault("time", time.time())
        with open(self.log, "a") as fh:
            fh.write(json.dumps(event, default=_jsonable) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # on disk before the app moves on

    def events(self):
        if not self.log.exists():
            return []
        out = []
        with open(self.log) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut off by a crash
        return out

    # Written to a temp file and swapped in, so a crash never leaves half a file.
    def _replace(self, path, data):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def save_state(self, payload):
        self._replace(self.state, json.dumps(payload, default=_jsonable).encode())

    def load_state(self):
        if not self.state.exists():
            return None
        try:
            return json.loads(self.state.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def save_controller(self, controller):
        self._replace(self.controller, pickle.dumps(controller))

    # (controller or None, whether a saved one failed to load).
    def load_controller(self):
        if not self.controller.exists():
            return None, False
        try:
            return pickle.loads(self.controller.read_bytes()), False
        except Exception:
            return None, True
