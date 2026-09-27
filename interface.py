# A human-playable version of the simulation: python interface.py. You answer questions,
# choose whether to redo a miss, and whether to start the next set. State lives in data/.

import atexit
import copy
import dataclasses
import json
import os
import pickle
import random
import signal
import threading
import time
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tkinter import ttk

import numpy as np
import pandas as pd

from config import (
    MAX_EPISODE_STEPS,
    ModelState,
    UserParams,
    cost_components,
    settle_skipped,
    slot_reward,
    stage_conditions,
    update_ability,
    value_at_choice_point,
    value_of_stage,
)
from controller import CANDIDATES, REVIEW_LEVELS, Controller, plan_in_worker
from inference import fit, passages_to_frame
from initiation import snap_cost, snap_value
from persistence import attempts_to_frame, fit_patience
from questions import TYPES, Ratings, choose_type

DATA_DIR = Path(__file__).parent / "data"
PLAN_EVERY = 5      # sets (start decisions) between controller plans
MIN_FIT_ROWS = 20
RETIRE_AFTER = 3   # failed returns before a question stops coming back; redos do not count
SLIP_SECONDS = 2.0 # a miss put right on the redo faster than this is a typo, not a miss

BIG = ("Helvetica", 26)
MED = ("Helvetica", 15)
SMALL = ("Helvetica", 11)
MONO = ("Menlo", 11)

# Decision logs live in decisions.jsonl; state.json holds everything else.
LOGS = ("initiation_passages", "first_passage", "persistence")
SETTINGS = ("target", "stages", "cold", "review", "return_skipped", "controller",
            "show_count", "show_score", "feedback", "show_choice", "forced")


# ---------------------------------------------------------------- persistence

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


# Whether the record's last session ended through the app: "I'm done", or a logged exit.
def closed_cleanly(events):
    last = events[-1]
    return last.get("node") == "abandon" or (last.get("node") == "start" and last.get("GO") == 0)


# Version of the rules for rating and retiring. Older saves are replayed forward.
# 3: the reaction terms went; what a set does to the next start decision is carried by
#    the prediction error in `value` instead, so rows recorded under 1-2 hold a `value`
#    that predates the carry and are not comparable with later ones.
RULES = 3


# (ratings, failed returns per question), replayed from the record under current rules.
def rebuild_from_record(events):
    ratings = Ratings()
    failed = {}
    miss = None           # (question, ratings before that miss)
    after_repeat = False
    for e in events:
        node = e.get("node")
        if node == "stage":
            after_repeat = e.get("GO") == 1 # older records do not mark redos; this infers them
            continue
        if node == "set_start":
            miss, after_repeat = None, False
        elif node == "retired":
            failed.pop(e.get("question"), None)
        if node not in ("answer", "skip"):
            continue
        redo = e.get("redo", after_repeat) # newer records mark it outright
        after_repeat = False
        review = bool(e.get("review"))
        t, question = e["type"], e["question"]
        if node == "skip":
            if review and not redo:
                failed[question] = failed.get(question, 0) + 1
            miss = None
            continue
        correct = bool(e.get("correct"))
        slip = (redo and correct and e.get("rt") is not None and e["rt"] < SLIP_SECONDS
                and miss is not None and miss[0] == question)
        if slip:
            ratings = miss[1]
            ratings.update(t, True)
            miss = None
        elif review:
            if not redo and not correct:
                failed[question] = failed.get(question, 0) + 1
            miss = None
        else:
            before = copy.deepcopy(ratings)
            ratings.update(t, correct)
            miss = None if correct else (question, before)
    return ratings, failed


def state_from_dict(d):
    names = {f.name for f in dataclasses.fields(ModelState)} - set(LOGS)
    st = ModelState(**{k: v for k, v in d.items() if k in names})
    st.theta = np.array(st.theta, dtype=float)
    for name in ("type_diffs", "stage_diffs"):
        if getattr(st, name) is not None:
            setattr(st, name, np.array(getattr(st, name), dtype=float))
    st.rpe = {int(k): v for k, v in st.rpe.items()}
    return st


# Everything that has to survive closing the window.
class Store:
    def __init__(self, root=DATA_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "decisions.jsonl" # append-only record, never rewritten
        self.state = self.root / "state.json"    # snapshot to resume from
        self.controller = self.root / "controller.pkl"

    def append(self, event):
        event.setdefault("time", time.time())
        with open(self.log, "a") as fh:
            fh.write(json.dumps(event, default=_jsonable) + "\n")
            fh.flush()
            os.fsync(fh.fileno()) # on disk before the app moves on

    def events(self):
        if not self.log.exists():
            return []
        out = []
        with open(self.log) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass # a line cut off by a crash
        return out

    def _replace(self, path, data):
        # Written to a temp file and swapped in
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

# ---------------------------------------------------------------- the app

class App:
    def __init__(self, root, data_dir=DATA_DIR):
        self.root = root
        root.title("initiation model -- human version")
        root.geometry("880x800")
        root.minsize(880, 800)

        # f and k are what the fit recovers; only g and a are used here.
        self.params = UserParams(f=0.0, k=0.0)
        self.store = Store(data_dir)

        self.v_target = tk.DoubleVar(value=0.75) # chance of getting a question right
        self.v_stages = tk.IntVar(value=4)
        self.v_cold = tk.DoubleVar(value=0.6) # cold-start cost profile
        self.v_review = tk.DoubleVar(value=0.25) # chance a slot brings back an owed question
        self.v_return_skipped = tk.BooleanVar(value=False)  # skipped questions join what review brings back
        self.v_controller = tk.BooleanVar(value=False)
        self.v_show_count = tk.BooleanVar(value=True)
        self.v_show_score = tk.BooleanVar(value=True)
        self.v_feedback = tk.BooleanVar(value=True)
        self.v_show_choice = tk.BooleanVar(value=True)
        self.v_forced = tk.StringVar(value="continue")

        self._timer = None
        self.problem = None # the question on screen: (text, answer)
        self.item = 0  # its type; for a review, the type it was first missed as
        self.p_now = 0.0
        self.shown_at = time.perf_counter()
        self.reviewing = False
        self.source = None # for one that came back: "missed" or "skipped"; None when new
        self.redo = False # the question on screen is an immediate redo of a miss
        self.first_pending = True # no question shown yet since the app was opened
        self.first_of_launch = False
        self.target_now = 0.75
        self.shown_this_set = set()
        self.last_miss = None
        self.retire_pending = None
        self.screen = None # "question", "redo", "start", or None between screens
        self.prompt_at = time.perf_counter()
        self.launched_at = time.time()
        self.closed = False
        self.fit_result = "not fitted yet"

        self.controller = None
        self.controller_status = "off"
        self._pool = None
        self._plan_future = None

        self._load()
        self._build()
        self.start_episode()
        atexit.register(self._record_unclean_exit)

    # ------------------------------------------------------------ load / save

    def settings(self):
        return {name: getattr(self, "v_" + name).get() for name in SETTINGS}

    def _load(self):
        saved = self.store.load_state()
        events = self.store.events()
        if saved is None:
            self.state = ModelState(t=1)
            self.ratings = Ratings()
            self.owed = {}
            self.skipped = {}
            self.failed_returns = {}
            self.set_id = 0
            self.session = 1
            self.store.append({"node": "first_launch", "session": 1})
        else:
            self.state = state_from_dict(saved["model"])
            self.ratings = Ratings.from_dict(saved["ratings"])
            self.owed = {int(t): [tuple(q) for q in qs] for t, qs in saved.get("owed", {}).items()}
            self.skipped = {int(t): [tuple(q) for q in qs]
                            for t, qs in saved.get("skipped", {}).items()}
            self.failed_returns = saved.get("failed_returns", {})
            self.set_id = saved.get("set_id", 0)
            self.session = saved.get("session", 1) + 1
            last = saved.get("last_active")
            if events and not closed_cleanly(events):

                self.store.append({**(saved.get("on_screen") or {}), "node": "abandon",
                                   "session": saved.get("session", 1), "set": self.set_id,
                                   "unlogged": True, "last_active": last, "rt": None})
            self.store.append({"node": "return", "session": self.session,
                               "gap_seconds": time.time() - last if last else None})
            for name, value in saved.get("settings", {}).items():
                var = getattr(self, "v_" + name, None)
                if var is not None:
                    try:
                        var.set(value)
                    except tk.TclError:
                        pass
            if saved.get("rules", 0) < RULES:

                before = self.ratings.ability
                self.ratings, self.failed_returns = rebuild_from_record(events)
                self.store.append({"node": "rebuilt", "session": self.session, "rules": RULES,
                                   "ability_before": before, "ability_after": self.ratings.ability,
                                   "failed_returns": self.failed_returns})
                self._retire_due()

        # The decision history is the record itself, so every fit sees every session.
        for ev in events:
            node = ev.get("node")
            if node == "start":
                self.state.initiation_passages.append(ev)
            elif node == "stage":
                self.state.first_passage.append(ev)
            elif node in ("answer", "skip") or (node == "abandon" and ev.get("screen") == "question"
                                                and ev.get("rt") is not None):
                # Answered: kept going at least that long. Skipped or left: gave up.
                self.state.persistence.append(ev)

        self.sync_measurement()
        controller, failed = self.store.load_controller()
        if failed:
            self.store.append({"node": "controller_reset", "session": self.session})
        if controller is not None:
            controller.candidates = CANDIDATES # a saved one may predate a new setting
        self.controller = controller

    def persist(self):
        self.store.save_state({
            "model": state_to_dict(self.state),
            "ratings": self.ratings.to_dict(),
            "owed": {str(t): [list(q) for q in qs] for t, qs in self.owed.items()},
            "skipped": {str(t): [list(q) for q in qs] for t, qs in self.skipped.items()},
            "failed_returns": self.failed_returns,
            "rules": RULES,
            "on_screen": self.on_screen(),
            "set_id": self.set_id,
            "session": self.session,
            "last_active": time.time(),
            "settings": self.settings(),
        })

    # Competence here is the measurement, not the simulator's corrected-question rule.
    def sync_measurement(self):
        self.state.ability = self.ratings.ability
        self.state.type_diffs = np.array(self.ratings.difficulty, dtype=float)

    # ------------------------------------------------------------ layout
    def _build(self):
        self.left = ttk.Frame(self.root, padding=24)
        self.left.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.left.columnconfigure(0, weight=1)
        self.left.rowconfigure(9, weight=1)  # keeps the task area from jumping

        right = ttk.Frame(self.root, padding=16)
        right.grid(row=0, column=1, sticky="ns")

        ttk.Label(right, text="settings (live)", font=MED).pack(anchor="w")
        ttk.Label(right, text="target chance of getting it right",
                  font=SMALL).pack(anchor="w", pady=(10, 0))
        ttk.Scale(right, from_=0.3, to=0.95, variable=self.v_target, length=200).pack()
        ttk.Label(right, text="stages per set", font=SMALL).pack(anchor="w", pady=(8, 0))
        ttk.Spinbox(right, from_=2, to=8, textvariable=self.v_stages, width=6).pack(anchor="w")
        ttk.Label(right, text="cold-start cost", font=SMALL).pack(anchor="w", pady=(8, 0))
        ttk.Scale(right, from_=0.05, to=1.5, variable=self.v_cold, length=200).pack()
        ttk.Label(right, text="review rate (owed questions return)",
                  font=SMALL).pack(anchor="w", pady=(8, 0))
        ttk.Scale(right, from_=0.0, to=1.0, variable=self.v_review, length=200).pack()
        ttk.Checkbutton(right, text="skipped questions come back too",
                        variable=self.v_return_skipped).pack(anchor="w")
        ttk.Checkbutton(right, text="controller sets target, review, feedback, skipped",
                        variable=self.v_controller).pack(anchor="w", pady=(6, 0))

        for text, var in [
            ("show stage count", self.v_show_count),
            ("show score so far", self.v_show_score),
            ("feedback (from next set)", self.v_feedback),
            ("offer repeat-or-continue", self.v_show_choice),
        ]:
            ttk.Checkbutton(right, text=text, variable=var).pack(anchor="w", pady=1)

        ttk.Label(right, text="when not offered, it:", font=SMALL).pack(anchor="w", pady=(6, 0))
        forced = ttk.Frame(right)
        forced.pack(anchor="w")
        ttk.Radiobutton(forced, text="continues", value="continue",
                        variable=self.v_forced).pack(side="left")
        ttk.Radiobutton(forced, text="repeats", value="repeat",
                        variable=self.v_forced).pack(side="left")

        ttk.Separator(right).pack(fill="x", pady=10)
        ttk.Label(right, text="behind the scenes", font=MED).pack(anchor="w")
        self.stats = ttk.Label(right, text="", font=MONO, justify="left")
        self.stats.pack(anchor="w", pady=(6, 0))
        self.now_label = ttk.Label(right, text="", font=SMALL, justify="left", wraplength=280)
        self.now_label.pack(anchor="w", pady=(4, 0))
        self.ctrl_label = ttk.Label(right, text="", font=SMALL, justify="left", wraplength=280)
        self.ctrl_label.pack(anchor="w", pady=(4, 0))

        ttk.Separator(right).pack(fill="x", pady=10)
        self.fit_label = ttk.Label(right, text=self.fit_result, font=MONO, justify="left")
        self.fit_label.pack(anchor="w")
        self.fit_button = ttk.Button(right, text="fit my parameters", command=self.run_fit)
        self.fit_button.pack(anchor="w", pady=6)
        ttk.Button(right, text="export all decisions to csv", command=self.export).pack(anchor="w")

        self.refresh_stats()

    def clear(self):
        if self._timer:
            self.root.after_cancel(self._timer)
            self._timer = None
        for key in ("1", "2"):
            self.root.unbind(key)
        for w in self.left.winfo_children():
            w.destroy()

    def refresh_stats(self):
        s = self.state
        starts = s.initiation_passages
        yes = sum(p["GO"] == 1 for p in starts)
        no = sum(p["GO"] == 0 for p in starts)
        base, effort = cost_components(s)
        self.stats.config(text="\n".join([
            f"measured ability {self.ratings.ability:6.2f}",
            f"V(next set)      {value_at_choice_point(s):6.3f}",
            f"cost now         {base + effort:6.3f}",
            f"corrections      {s.corrections:6d}",
            f"questions owed   {len(s.missed):6d}",
            f"skipped (total)  {s.skips:6d}",
            f"skipped, pending {len(s.skipped):6d}",
            "",
            f"session          {self.session:6d}",
            f"sets done        {s.episodes:6d}",
            f"start: yes / no  {yes:3d} / {no}",
            f"redo decisions   {len(s.first_passage):6d}",
        ]))
        self.now_label.config(
            text=f"now: {TYPES[self.item].name}, your chance {self.p_now:.0%}")
        self.ctrl_label.config(text=f"controller: {self.controller_status}")

    # ------------------------------------------------------------ screens

    # A two-button decision with no deadline. `on_done(go, rt)`; RT is time to click.
    # A choice slower than T_DUR is kept, and read as undecided when fitting.
    def ask_decision(self, prompt, yes_label, no_label, on_done):
        self.clear()
        ttk.Label(self.left, text=prompt, font=BIG).grid(row=0, column=0, pady=(60, 20))
        row = ttk.Frame(self.left)
        row.grid(row=1, column=0, pady=20)

        start = time.perf_counter()
        self.prompt_at = start
        done = []

        def finish(go):
            if done:
                return
            done.append(True)
            on_done(go, time.perf_counter() - start)

        ttk.Button(row, text=f"1  {yes_label}", command=lambda: finish(1)).pack(side="left", padx=8)
        ttk.Button(row, text=f"2  {no_label}", command=lambda: finish(0)).pack(side="left", padx=8)
        self.root.bind("1", lambda e: finish(1))
        self.root.bind("2", lambda e: finish(0))

    def show_problem(self):
        self.clear()
        if not self.reviewing:
            # A fresh slot: either a question they still owe, or a new one of whichever
            # type is nearest the target chance for this person.
            self.redo = False
            self.last_miss = None
            owed = [(t, q, "missed") for t, qs in self.owed.items() for q in qs]
            if self.v_return_skipped.get():
                owed += [(t, q, "skipped") for t, qs in self.skipped.items() for q in qs]
            # Never the same question twice in one set. (A redo is their own choice.)
            owed = [(t, q, src) for t, q, src in owed if tuple(q) not in self.shown_this_set]
            if owed and random.random() < self.v_review.get():
                self.item, self.problem, self.source = random.choice(owed)
                self.reviewing = True
            else:
                self.item = choose_type(self.ratings.probs(), self.v_target.get())
                self.problem = TYPES[self.item].make()
                self.source = None
        # Otherwise it is a redo: the same question again, already in self.problem.
        self.shown_this_set.add(tuple(self.problem))
        self.first_of_launch, self.first_pending = self.first_pending, False
        self.target_now = self.v_target.get()
        text, self.answer = self.problem
        self.p_now = self.ratings.p(self.item)

        header = []
        if self.v_show_count.get():
            header.append(f"stage {self.s + 1} of {self.state.stage_amt}")
        if self.v_show_score.get() and self.feedback_this_set:
            header.append(f"{self.passed} right")
        if self.reviewing and self.feedback_this_set:
            header.append("review") # flagged only with feedback on; it gives away a past miss
        ttk.Label(self.left, text="   ".join(header), font=SMALL).grid(row=0, column=0, pady=(20, 0))

        ttk.Label(self.left, text=f"{text} = ?", font=BIG).grid(row=1, column=0, pady=30)
        entry = ttk.Entry(self.left, font=BIG, width=10, justify="center")
        entry.grid(row=2, column=0)
        entry.focus_set()
        entry.bind("<Return>", lambda e: self.on_answer(entry.get()))
        entry.bind("<Escape>", lambda e: self.on_skip())
        ttk.Label(self.left, text="type the answer and press enter -- or give up on it",
                  font=SMALL).grid(row=3, column=0, pady=12)
        ttk.Button(self.left, text="skip  (esc)", command=self.on_skip).grid(row=4, column=0)
        self.refresh_stats()
        self.screen = "question"
        self.persist()  # so an exit nothing sees still leaves a record of what was on screen
        self.shown_at = time.perf_counter()

    def flash(self, text, then):
        self.clear()
        ttk.Label(self.left, text=text, font=BIG).grid(row=0, column=0, pady=90)
        self._timer = self.root.after(600, then)

    # ------------------------------------------------------------ the loop
    def start_episode(self):
        s = self.state # settings that would break mid-set are only picked up here
        s.stage_amt = self.v_stages.get()
        s.rpe = {r: 0 for r in range(s.stage_amt)}
        s.in_session = True  # playing a set is being in the task
        self.feedback_this_set = self.v_feedback.get()
        self.reviewing = False
        self.source = None
        self.redo = False
        self.last_miss = None
        self.shown_this_set = set()
        self.review_showings = 0
        self.set_id += 1
        self.s = self.steps = self.passed = self.wrong = self.skips = 0
        self.achievement = self.credited = 0.0 # earned this set, and how much has landed
        self.episode_deltas = []
        s.slot_delta = 0.0 # no slot precedes the first one
        self.store.append({"node": "set_start", "session": self.session, "set": self.set_id,
                           "target": self.v_target.get(), "review_rate": self.v_review.get(),
                           "feedback": self.feedback_this_set,
                           "controller": self.v_controller.get()})
        self.maybe_plan()
        self.next_stage()

    def next_stage(self):
        self.state.reentry_cost = round(self.v_cold.get(), 3)
        if self.steps >= MAX_EPISODE_STEPS:
            self.end_episode(abandoned=True)
        elif self.s >= self.state.stage_amt:
            self.end_episode(abandoned=False)
        else:
            self.show_problem()

    def on_answer(self, raw):
        rt = time.perf_counter() - self.shown_at
        raw = raw.strip()
        try:
            correct = int(raw) == self.answer
        except ValueError:
            correct = False

        self.steps += 1
        self.screen = None
        was_review = self.reviewing
        source = self.source
        redo = self.redo
        self.review_showings += was_review
        miss = self.last_miss
        slip = (redo and correct and rt < SLIP_SECONDS and miss is not None
                and miss["problem"] == self.problem)
        before = None
        if slip:
            # A typo, not a miss: undo what it did to the ratings and the owed
            # questions, and count it right the first time.
            self.ratings = miss["ratings"]
            p = self.ratings.update(self.item, True)
            self._take(self.owed, self.item, self.problem)
            self.state.missed.remove(self.item)
            self.wrong -= 1
            self.review_showings -= 1
        elif was_review:
            p = self.ratings.p(self.item) # the same question again is not a fresh sample
        else:
            # Measure how likely they were to get it, then move the ratings.
            before = copy.deepcopy(self.ratings)
            p = self.ratings.update(self.item, correct)
        self.last_miss = (None if correct or was_review
                          else {"problem": self.problem, "ratings": before})

        if not slip:
            # The model's bookkeeping of what is owed, mirrored with the real questions.
            if source == "skipped":
                settle_skipped(self.state, self.item, correct)
                self._take(self.skipped, self.item, self.problem)
                if not correct:
                    self.owed.setdefault(self.item, []).append(self.problem)
            else:
                update_ability(self.state, self.item, correct, was_review)
                if not was_review and not correct:
                    self.owed.setdefault(self.item, []).append(self.problem)
                elif was_review and correct:
                    self._take(self.owed, self.item, self.problem)
            self.state.challenge += 1.0 - p
        if was_review and not redo and not correct:
            self._failed_return()
        self.sync_measurement()
        if correct:
            self.passed += 1
            self.achievement += 1.0 - p
        else:
            self.wrong += 1
        rec = {"node": "answer", **self.question_record(), "given": raw, "correct": correct,
               "p": p, "rt": rt, "skipped": False, "slip": slip}
        self.state.persistence.append(rec)
        self.store.append(rec)
        self.land()
        self.persist()
        self.reviewing = False
        self.source = None
        self.redo = False

        if correct or not self.feedback_this_set:
            # No redo prompt with feedback hidden: it would give the miss away. Owed
            # questions come back through review instead.
            self.after_feedback(correct, self.advance)
        elif not self.v_show_choice.get():
            if self.v_forced.get() == "repeat":
                self.reviewing = True
                self.source = "missed"
                self.redo = True
                self.after_feedback(correct, self.next_stage)
            else:
                self.after_feedback(correct, self.advance)
        else:
            self.after_feedback(correct, self.ask_repeat)

    # The outcome just seen lands as reward and becomes a prediction error. Run per
    # outcome, not per slot, and before the redo prompt: they see how the question went,
    # that lands, and then they decide whether to take it again. With feedback hidden
    # nothing lands until the end-of-set score, which pays the set in one lump.
    def land(self):
        last = self.state.stage_amt - 1
        reward, self.credited = slot_reward(self.state, self.achievement, self.credited,
                                            self.feedback_this_set, self.s == last)
        delta = value_of_stage(self.state, self.params, self.s, reward)
        self.state.rpe[self.s] = delta
        self.state.slot_delta = delta
        self.episode_deltas.append(delta)

    @staticmethod
    def _take(pool, t, question):
        pool[t].remove(question)
        if not pool[t]:
            del pool[t]

    # They gave up on the question on screen. How long they worked on it first is the
    # measurement; a skip is neither right nor wrong, so the ratings do not move.
    def on_skip(self):
        if self.closed or self.screen != "question":
            return
        rt = time.perf_counter() - self.shown_at
        self.screen = None
        self.steps += 1
        self.skips += 1
        was_review = self.reviewing
        self.review_showings += was_review
        self.state.skips += 1
        if not was_review:
            # A new question joins the skipped ones; one given up on again stays put.
            self.state.skipped.append(self.item)
            self.skipped.setdefault(self.item, []).append(self.problem)
        elif not self.redo:
            self._failed_return()
        rec = {"node": "skip", **self.question_record(), "rt": rt, "skipped": True}
        self.state.persistence.append(rec)
        self.store.append(rec)
        self.land() # the slot is spent and earned nothing, which is itself an error
        self.persist()
        self.reviewing = False
        self.source = None
        self.redo = False
        self.last_miss = None
        self.advance()

    # The question on screen came back and was missed or skipped again.
    def _failed_return(self):
        key = self.problem[0]
        n = self.failed_returns[key] = self.failed_returns.get(key, 0) + 1
        if n >= RETIRE_AFTER:
            # Applied when the set moves on, so a redo in between still works.
            self.retire_pending = (self.item, self.problem)

    # Stop bringing a question back. It stays in the record.
    def _retire(self, item, problem):
        for pool, model_list, name in ((self.owed, self.state.missed, "missed"),
                                       (self.skipped, self.state.skipped, "skipped")):
            if problem in pool.get(item, []):
                self._take(pool, item, problem)
                model_list.remove(item)
                break
        else:
            return  # answered right in the meantime
        self.store.append({"node": "retired", "session": self.session, "set": self.set_id,
                           "type": item, "type_name": TYPES[item].name, "question": problem[0],
                           "answer": problem[1], "from": name,
                           "failed_returns": self.failed_returns.pop(problem[0], 0)})
        self.persist()

    # Retire every owed or skipped question that has already failed RETIRE_AFTER returns.
    def _retire_due(self):
        for t, qs in list(self.owed.items()) + list(self.skipped.items()):
            for q in list(qs):
                if self.failed_returns.get(q[0], 0) >= RETIRE_AFTER:
                    self._retire(t, q)

    # What is known about the question on screen, for the record.
    def question_record(self):
        return {"session": self.session, "set": self.set_id, "slot": self.s,
                "type": self.item, "type_name": TYPES[self.item].name,
                "question": self.problem[0], "answer": self.problem[1], "p": self.p_now,
                "stage_amt": self.state.stage_amt, "review": self.reviewing, "redo": self.redo,
                "delta": self.state.slot_delta,
                "source": self.source, "first_of_launch": self.first_of_launch,
                "feedback": self.feedback_this_set, "target": self.target_now}

    def on_screen(self):
        rec = {"screen": self.screen}
        if self.screen == "question":
            rec.update(self.question_record())
        return rec

    # The abandon event: where they were when they left, and for how long. Leaving from a
    # question is giving up on it, so it joins the persistence data like a skip.
    def exit_record(self):
        rec = {"node": "abandon", "session": self.session, "set": self.set_id,
               "slot": getattr(self, "s", 0), "screen": self.screen,
               "between_sets": self.screen == "start", "owed": len(self.state.missed),
               "seconds_since_launch": time.time() - self.launched_at}
        if self.screen == "question":
            rec.update(self.question_record(), rt=time.perf_counter() - self.shown_at,
                       skipped=True, left=True)
        elif self.screen in ("start", "redo"):
            rec["prompt_seconds"] = time.perf_counter() - self.prompt_at
        return rec

    # Python is exiting and the app never closed. File writes only: Tk may be gone.
    def _record_unclean_exit(self):
        if self.closed:
            return
        self.closed = True
        rec = self.exit_record()
        rec["via"] = "process exit"
        self.store.append(rec)
        try:
            self.state.in_session = False
            self.persist()
        except Exception:
            pass

    def after_feedback(self, correct, then):
        if self.feedback_this_set:
            self.flash("correct" if correct else "wrong", then)
        else:
            then()

    # The repeat-or-continue node (config.stage_conditions). Only reached with feedback on.
    def ask_repeat(self):
        value, cost = stage_conditions(self.state, self.params, self.s, self.item)
        self.screen = "redo"

        def done(go, rt):
            self.screen = None
            rec = {"node": "stage", "session": self.session, "set": self.set_id,
                   "value": snap_value(value), "cost": snap_cost(cost), "GO": go, "RT": rt,
                   "stage": self.s, "item": self.item, "timestep": self.state.t,
                   "slot_delta": self.state.slot_delta}
            self.state.first_passage.append(rec)
            self.store.append(rec)
            self.persist()
            if go == 1:
                self.reviewing = True
                self.source = "missed"
                self.redo = True
                self.next_stage()
            else:
                self.advance()

        self.ask_decision("try this one again?", "repeat it", "move on", done)

    # Move to the next slot and run the TD update for the one just left.
    def advance(self):
        if self.retire_pending is not None:
            self._retire(*self.retire_pending)
            self.retire_pending = None
        self.s += 1
        self.next_stage()

    def end_episode(self, abandoned):
        s = self.state
        n = s.stage_amt
        s.total_reward += self.achievement
        s.last_review_load = self.review_showings / n
        s.last_episode_effort = 1.0 + s.last_review_load
        s.last_episode_passed = min(self.passed / n, 1.0)
        s.last_achievement = self.achievement / n
        s.last_feedback = int(self.feedback_this_set)
        missed_share = min(self.wrong / n, 1.0)
        s.last_seen_miss = missed_share if self.feedback_this_set else 0.0
        s.last_unseen_miss = 0.0 if self.feedback_this_set else missed_share
        # How the set went against what was predicted of it, carried into the decision
        # to start another one. A set walked out of was never judged.
        s.last_set_delta = 0.0 if abandoned else float(sum(self.episode_deltas))
        if abandoned:
            s.abandoned += 1
        else:
            s.episodes += 1
        self.store.append({"node": "set_end", "session": self.session, "set": self.set_id,
                           "abandoned": abandoned, "passed": s.last_episode_passed,
                           "achievement": s.last_achievement, "review_load": s.last_review_load,
                           "skips": self.skips, "seen_miss": s.last_seen_miss,
                           "unseen_miss": s.last_unseen_miss, "feedback": s.last_feedback,
                           "set_delta": s.last_set_delta})
        self.persist()
        self.ask_initiate()

    # The start decision: the next set, or stop. Stopping closes the app.
    def ask_initiate(self):
        s = self.state
        s.reentry_cost = round(self.v_cold.get(), 3)
        value = value_at_choice_point(s)
        base, effort = cost_components(s)
        self.screen = "start"
        self.refresh_stats()

        def done(go, rt):
            self.screen = None
            rec = {"node": "start", "session": self.session, "set": self.set_id,
                   "value": snap_value(value), "cost": snap_cost(base + effort),
                   "GO": go, "RT": rt, "last_set_delta": s.last_set_delta,
                   "seen_miss": s.last_seen_miss, "unseen_miss": s.last_unseen_miss,
                   "feedback": s.last_feedback,
                   "in_session": s.in_session, "cost_base": base, "cost_effort": effort,
                   "last_episode_passed": s.last_episode_passed,
                   "last_achievement": s.last_achievement,
                   "last_review_load": s.last_review_load,
                   "missed_count": len(s.missed), "timestep": s.t}
            s.initiation_passages.append(rec)
            self.store.append(rec)
            s.t += 1
            if go != 1:
                s.stuck += 1
                self.quit()
            else:
                self.start_episode()

        self.ask_decision("start the next set?", "start the next set", "I'm done", done)

    # Save everything and close. `abandon`: the window was closed without choosing.
    def quit(self, abandon=False):
        if self.closed:
            return
        self.closed = True
        atexit.unregister(self._record_unclean_exit)
        if abandon:
            self.store.append(self.exit_record())
        self.state.in_session = False  # coming back is a cold start
        self.persist()
        if self.controller is not None:
            self.store.save_controller(self.controller)
        self.shutdown()
        self.root.destroy()

    def on_close(self):
        self.quit(abandon=True)

    # ------------------------------------------------------------ controller
    def current_environment(self):
        t, fb = self.v_target.get(), self.v_feedback.get()
        r = min(REVIEW_LEVELS, key=lambda x: abs(x - self.v_review.get()))
        # With review at 0 nothing comes back, so "skipped come back" means nothing.
        rs = self.v_return_skipped.get() and r > 0
        return min(CANDIDATES, key=lambda e: abs(e.target_p - t) + (e.review_rate != r) * 10
                   + (e.feedback != fb) * 10 + (e.return_skipped != rs) * 10)

    def maybe_plan(self):
        if not self.v_controller.get() or self._plan_future is not None:
            return
        if self.controller is None:
            self.controller = Controller(start=self.current_environment(), min_spacing=PLAN_EVERY)
        if not self.controller.due(self.state):
            return
        # The settings may have been moved by hand since the last plan.
        self.controller.env = self.current_environment()
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1)
        self.controller_status = "planning..."
        # Copy now: the worker pickles later, while this thread keeps logging.
        self._plan_future = self._pool.submit(
            plan_in_worker, self.controller, copy.deepcopy(self.state), self.params)
        self.root.after(200, self.check_plan)

    def check_plan(self):
        fut = self._plan_future
        if fut is None or self.closed:
            return
        if not fut.done():
            self.root.after(200, self.check_plan)
            return
        self._plan_future = None
        try:
            self.controller, update = fut.result()
        except Exception as exc:
            self.controller_status = f"planning failed: {exc}"
            self.refresh_stats()
            return
        if not self.v_controller.get():
            self.controller_status = "off (last plan discarded)"
            self.refresh_stats()
            return

        env, est = update.chosen, update.estimate
        self.v_target.set(env.target_p)
        self.v_review.set(env.review_rate)
        self.v_feedback.set(env.feedback)
        self.v_return_skipped.set(env.return_skipped)
        self.controller_status = (
            f"f {est.f:.2f}, k {est.k:.2f}, patience {est.patience:.1f}s, T0 {est.t0:.2f}s "
            f"from {est.rows} rows ({est.note}). Next: {env} -- {update.reason}.")
        self.store.save_controller(self.controller)
        self.store.append({"node": "controller", "session": self.session, "set": self.set_id,
                           "rows": est.rows, "f": est.f, "k": est.k,
                           "patience": est.patience, "t0": est.t0, "note": est.note,
                           "target": env.target_p, "review_rate": env.review_rate,
                           "feedback_on": env.feedback, "return_skipped": env.return_skipped,
                           "switched": env != update.previous,
                           "reason": update.reason})
        self.refresh_stats()

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ inference
    def frame(self):
        return passages_to_frame(self.state.initiation_passages + self.state.first_passage)

    def run_fit(self):
        df = self.frame()
        if len(df) < MIN_FIT_ROWS:
            self.fit_label.config(text=f"only {len(df)} decided rows\nkeep going")
            return

        attempts = attempts_to_frame(self.state.persistence)
        self.fit_button.config(state="disabled")
        self.fit_label.config(text=f"fitting {len(df)} rows, {len(attempts)} questions...")

        def work():
            try:
                f_hat, k_hat, t0_hat = fit(df, t0=True)
                gave_up = int(attempts["skipped"].sum()) if len(attempts) else 0
                patience = f"{fit_patience(f_hat, k_hat, attempts):5.1f}s" if gave_up else "  n/a"
                text = "\n".join([
                    f"f (reward pull)  {f_hat:5.2f}",
                    f"k (effort push)  {k_hat:5.2f}",
                    f"patience         {patience}",
                    f"non-decision     {t0_hat:5.2f}s",
                    f"f, k: {len(df)} decisions",
                    f"patience: {len(attempts)} questions, {gave_up} given up",
                    "rough under ~700 decisions",
                ])
            except Exception as exc:
                text = f"fit failed: {exc}"

            def show():
                self.fit_label.config(text=text)
                self.fit_button.config(state="normal")
            self.root.after(0, show)

        threading.Thread(target=work, daemon=True).start()

    def export(self):
        events = self.store.events()
        path = self.store.root / "decisions.csv"
        pd.json_normalize(events).to_csv(path, index=False)
        self.fit_label.config(text=f"wrote data/{path.name}\n{len(events)} events")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Ctrl-C, a closed terminal or `kill` close the app the same way as the window's
    # close button, so the exit is logged with what was on screen.
    def stop(*_):
        app.on_close()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)

    def poll():
        # Python only runs a signal handler when it gets control back from Tk.
        if not app.closed:
            root.after(250, poll)

    poll()
    root.mainloop()
