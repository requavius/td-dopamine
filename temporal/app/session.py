"""One person playing the task: the rules, the bookkeeping and the record.

A frontend (desktop.py, web.py) subclasses `Session` and draws the screens; every rule
lives here, so both frontends write the same record. Screen code goes through the hooks
`make_var`, `build`, `clear`, `refresh_stats`, `render_problem`, `ask_decision`, `flash`
and `show_summary`, so a change to the flow needs no frontend change unless it adds a new
kind of screen.
"""
import atexit
import copy
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..controller import (APP_CANDIDATES, REVIEW_LEVELS, STAGE_LEVELS, AppController,
                          plan_in_worker)
from ..inference import decision_arrays, fit_closed_form, passages_to_frame
from ..model.accumulator import snap_cost, snap_value
from ..model.competence import settle_skipped, update_ability
from ..model.conditions import cost_components, stage_conditions, value_at_choice_point
from ..model.params import ModelState, UserParams
from ..model.persistence import fit_patience
from ..model.questions import TYPES, Ratings, choose_type
from ..model.value import ACHIEVEMENT_SCALE, late_reward, slot_reward, value_of_stage
from .store import DATA_DIR, Store, closed_cleanly, state_from_dict, state_to_dict

PLAN_EVERY = 5         # sets (start decisions) between controller plans
MIN_FIT_ROWS = 20
MAX_EPISODE_STEPS = 40 # questions shown in one set before it is abandoned
RETIRE_AFTER = 3       # failed returns before a question stops coming back; redos do not count
SLIP_SECONDS = 2.0     # a miss put right on the redo faster than this is a typo, not a miss

# Settings and their defaults; the kind of variable follows the type of the default.
DEFAULTS = {
    "target": 0.75,          # chance of getting a question right
    "stages": 4,
    "cold": 0.6,             # cold-start cost profile
    "review": 0.25,          # chance a slot brings back an owed question
    "return_skipped": False, # skipped questions join what review brings back
    "controller": False,
    "show_count": True,
    "show_score": True,
    "feedback": True,
    "show_choice": True,
    "end_redos": False,      # with feedback hidden, offer the set's misses after its score
    "countdown": 0.0,        # seconds on every redo prompt; 0 for none
    "forced": "continue",
}
SETTINGS = tuple(DEFAULTS)

# Version of the rules for rating and retiring. Older saves are replayed forward.
# 3: the reaction terms went; what a set does to the next start decision is carried by
#    the prediction error in `value` instead, so rows recorded under 1-2 hold a `value`
#    that predates the carry and are not comparable with later ones.
RULES = 3


# (ratings, failed returns per question), replayed from the record under current rules.
def rebuild_from_record(events):
    ratings = Ratings()
    failed = {}
    miss = None  # (question, ratings before that miss)
    after_repeat = False
    for e in events:
        node = e.get("node")
        if node == "stage":
            after_repeat = e.get("GO") == 1  # older records do not mark redos; this infers them
            continue
        if node in ("set_start", "score"):
            miss, after_repeat = None, False  # a redo after the score is never a typo fix
        elif node == "retired":
            failed.pop(e.get("question"), None)
        if node not in ("answer", "skip"):
            continue
        redo = e.get("redo", after_repeat)  # newer records mark it outright
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


# f, k, T0 and λ from decisions, patience from answer and skip times, as text. Redo
# prompts under a countdown are read at their own bound and deadline (fit_closed_form).
def fit_summary(passages, attempts):
    f_hat, k_hat, t0_hat, lam_hat = fit_closed_form(passages)
    go, *_, deadline = decision_arrays(passages)
    timed = int((~np.isnan(deadline)).sum())
    gave_up = int(attempts["skipped"].sum()) if len(attempts) else 0
    patience = f"{fit_patience(f_hat, k_hat, attempts):5.1f}s" if gave_up else "  n/a"
    return "\n".join([
        f"f (reward pull)  {f_hat:5.2f}",
        f"k (effort push)  {k_hat:5.2f}",
        f"patience         {patience}",
        f"non-decision     {t0_hat:5.2f}s",
        f"λ (countdown)    {lam_hat:5.2f}" if timed else "λ (countdown)      n/a",
        f"f, k: {int((go >= 0).sum())} decisions, {timed} under a countdown",
        f"patience: {len(attempts)} questions, {gave_up} given up",
        "rough under ~700 decisions",
    ])


class Session:
    # `root` schedules callbacks: `after(ms, fn)`, `after_cancel(handle)`, `destroy()`.
    def __init__(self, root, data_dir=DATA_DIR):
        self.root = root
        # f and k are what the fit recovers; only g and a are used here.
        self.params = UserParams(f=0.0, k=0.0)
        self.store = Store(data_dir)

        for name, value in DEFAULTS.items():
            setattr(self, "v_" + name, self.make_var(value))

        self._timer = None
        self.problem = None  # the question on screen: (text, answer)
        self.item = 0        # its type; for a review, the type it was first missed as
        self.p_now = 0.0
        self.shown_at = time.perf_counter()
        self.reviewing = False
        self.source = None   # for one that came back: "missed" or "skipped"; None when new
        self.redo = False    # the question on screen is an immediate redo of a miss
        self.at_end = False  # past the set's last slot: its score and end-of-set redos
        self.unseen = []     # (type, question, number) missed new this set, feedback hidden
        self.set_log = []    # (line, outcome) for each question of the set, for its summary
        self.end_number = 0  # which question of the set a redo after the score is
        self.end_points = 0.0  # what that redo earned, for its flash
        self.end_misses = [] # those still to be offered on the end-of-set screen
        self.score = 0.0     # TD reward landed since the app was opened: the score on screen
        self.first_pending = True  # no question shown yet since the app was opened
        self.first_of_launch = False
        self.target_now = 0.75
        self.shown_this_set = set()
        self.last_miss = None
        self.retire_pending = None
        self.screen = None   # "question", "redo", "score", "start", or None between screens
        self.prompt_at = time.perf_counter()
        self.launched_at = time.time()
        self.closed = False
        self.fit_result = "not fitted yet"
        self.decision_now = None  # the last start or redo prompt, for the live model view

        self.controller = None
        self.controller_note = ""  # what the last plan this launch found, or why it failed
        self._pool = None
        self._plan_future = None

        self._load()
        self.build()
        self.start_episode()
        atexit.register(self._record_unclean_exit)

    # ------------------------------------------------------------ frontend hooks

    # A settings variable with .get() and .set().
    def make_var(self, value):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def refresh_stats(self):
        raise NotImplementedError

    def render_problem(self, text, header):
        raise NotImplementedError

    # A two-button decision. `on_done(go, rt)`; RT is time to click. With no `deadline` it
    # waits, and a choice slower than T_DUR is kept and read as undecided when fitting.
    # With one, it shows a countdown of that many seconds and closes when it runs out,
    # calling `on_done(None, None)`.
    def ask_decision(self, prompt, yes_label, no_label, on_done, deadline=None):
        raise NotImplementedError

    # Show `text` briefly, then call `then`.
    def flash(self, text, then):
        raise NotImplementedError

    # The summary that ends a set played with feedback hidden: `rows`, one (line, outcome)
    # per question, outcome "correct", "incorrect" or "skipped"; then `total`. It stays
    # until they continue, then calls `then`.
    def show_summary(self, rows, total, then):
        raise NotImplementedError

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
            self._resume(saved, events)

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
        self._load_controller()

    def _resume(self, saved, events):
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
            # The last session ended with nothing logged: record it as left from the
            # screen the snapshot says was showing.
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
                except Exception:  # a saved value this variable will not take
                    pass
        if saved.get("rules", 0) < RULES:
            before = self.ratings.ability
            self.ratings, self.failed_returns = rebuild_from_record(events)
            self.store.append({"node": "rebuilt", "session": self.session, "rules": RULES,
                               "ability_before": before, "ability_after": self.ratings.ability,
                               "failed_returns": self.failed_returns})
            self._retire_due()

    def _load_controller(self):
        controller, failed = self.store.load_controller()
        if failed:
            self.store.append({"node": "controller_reset", "session": self.session})
        if controller is not None and not isinstance(controller, AppController):
            # A saved one from before 2026-10-29 (a reward controller, or InfoController
            # over 96 settings): start the app's. Nothing is lost, its estimate is refit
            # from the record.
            self.store.append({"node": "controller_reset", "session": self.session})
            controller = None
        if controller is not None:
            controller.candidates = APP_CANDIDATES  # a saved one may predate a new setting
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

    # ------------------------------------------------------------ what the panel shows

    def stats_text(self):
        s = self.state
        starts = s.initiation_passages
        yes = sum(p["GO"] == 1 for p in starts)
        no = sum(p["GO"] == 0 for p in starts)
        base, effort = cost_components(s)
        return "\n".join([
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
            f"score (session)  {self.score:6.2f}",
        ])

    def now_text(self):
        return f"now: {TYPES[self.item].name}, your chance {self.p_now:.0%}"

    # ------------------------------------------------------------ a set

    def start_episode(self):
        s = self.state  # settings that would break mid-set are only picked up here
        s.stage_amt = self.v_stages.get()
        s.rpe = {r: 0 for r in range(s.stage_amt)}
        s.in_session = True  # playing a set is being in the task
        self.feedback_this_set = self.v_feedback.get()
        self.score_this_set = self.v_show_score.get()  # and with it, what the set pays
        self.set_points = 0.0  # TD reward landed in this set
        self.reviewing = False
        self.source = None
        self.redo = False
        self.at_end = False
        self.unseen = []
        self.set_log = []
        self.last_miss = None
        self.shown_this_set = set()
        self.review_showings = 0
        self.set_id += 1
        self.s = self.steps = self.passed = self.wrong = self.skips = 0
        self.achievement = self.credited = 0.0  # earned this set, and how much has landed
        self.episode_deltas = []
        s.slot_delta = 0.0  # no slot precedes the first one
        self.store.append({"node": "set_start", "session": self.session, "set": self.set_id,
                           "target": self.v_target.get(), "review_rate": self.v_review.get(),
                           "feedback": self.feedback_this_set,
                           "controller": self.v_controller.get(), "stages": s.stage_amt,
                           "show_score": self.score_this_set,
                           "redo_prompts": self.v_show_choice.get(),
                           "end_redos": self.v_end_redos.get(),
                           "countdown": self.v_countdown.get(), "forced": self.v_forced.get()})
        self.maybe_plan()
        self.next_stage()

    def next_stage(self):
        self.state.reentry_cost = round(self.v_cold.get(), 3)
        if self.steps >= MAX_EPISODE_STEPS:
            self.end_episode(abandoned=True)
        elif self.s >= self.state.stage_amt:
            self.end_of_set()
        else:
            self.show_problem()

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
        if self.at_end:
            header.append(f"end of set: question {self.end_number}")  # the summary's number
        elif self.v_show_count.get():
            header.append(f"stage {self.s + 1} of {self.state.stage_amt}")
        if self.score_this_set:
            # Only what has landed. With feedback hidden that is nothing until the end of
            # the set, so the score gives no miss away.
            header.append(f"score {self.score:.2f}")
        if self.reviewing and self.feedback_this_set:
            header.append("review")  # flagged only with feedback on; it gives away a past miss
        self.render_problem(text, header)
        self.screen = "question"
        self.persist()  # so an exit nothing sees still leaves a record of what was on screen
        self.shown_at = time.perf_counter()

    # `rt` is measured here unless the frontend timed it (web.py times it in the browser).
    def on_answer(self, raw, rt=None):
        if rt is None:
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
            p = self.ratings.p(self.item)  # the same question again is not a fresh sample
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
                    if not self.feedback_this_set:
                        self.unseen.append((self.item, self.problem, self.s + 1))
                elif was_review and correct:
                    self._take(self.owed, self.item, self.problem)
            self.state.challenge += 1.0 - p
        if was_review and not redo and not correct:
            self._failed_return()
        self.sync_measurement()
        if self.at_end:
            pass  # a redo after the score is not part of the set's score, which has been paid
        elif correct:
            self.passed += 1
            self.achievement += 1.0 - p
        else:
            self.wrong += 1
        if not self.at_end:
            points = (f"  +{ACHIEVEMENT_SCALE * (1.0 - p) / self.state.stage_amt:.2f}"
                      if correct and self.score_this_set else "")
            self.set_log.append((f"question {self.s + 1}: {self.problem[0]} = {raw}  "
                                 f"{'correct' if correct else 'incorrect'}{points}",
                                 "correct" if correct else "incorrect"))
        rec = {"node": "answer", **self.question_record(), "given": raw, "correct": correct,
               "p": p, "rt": rt, "skipped": False, "slip": slip}
        self.state.persistence.append(rec)
        self.store.append(rec)
        if self.at_end:
            # Its outcome is shown, so what it earns lands now (Simulation(end_feedback=True)),
            # after the record, which keeps the error it was worked under.
            self.end_points = self.land_late(p) if correct else 0.0
        else:
            self.land()
        self.persist()
        self.reviewing = False
        self.source = None
        self.redo = False

        if correct or not self.feedback_this_set:
            # No redo prompt with feedback hidden: it would give the miss away. Owed
            # questions come back through review, or after the score (`end_of_set`).
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

    # They gave up on the question on screen. How long they worked on it first is the
    # measurement; a skip is neither right nor wrong, so the ratings do not move.
    def on_skip(self, rt=None):
        if self.closed or self.screen != "question":
            return
        if rt is None:
            rt = time.perf_counter() - self.shown_at
        self.screen = None
        self.steps += 1
        self.skips += 1
        was_review = self.reviewing
        self.review_showings += was_review
        self.state.skips += 1
        if not self.at_end:
            self.set_log.append((f"question {self.s + 1}: {self.problem[0]}  skipped",
                                 "skipped"))
        if not was_review:
            # A new question joins the skipped ones; one given up on again stays put.
            self.state.skipped.append(self.item)
            self.skipped.setdefault(self.item, []).append(self.problem)
        elif not self.redo:
            self._failed_return()
        rec = {"node": "skip", **self.question_record(), "rt": rt, "skipped": True}
        self.state.persistence.append(rec)
        self.store.append(rec)
        self.land()  # the slot is spent and earned nothing, which is itself an error
        self.persist()
        self.reviewing = False
        self.source = None
        self.redo = False
        self.last_miss = None
        self.advance()

    # The outcome just seen lands as reward and becomes a prediction error. Run per
    # outcome, not per slot, and before the redo prompt: they see how the question went,
    # that lands, and then they decide whether to take it again. With feedback hidden
    # nothing lands until the end-of-set score, which pays the set in one lump. What lands
    # is what the score shows: the (1 - p)-weighted right answers, or with the score hidden
    # the fraction right (Simulation.show_score). It is added to the score on screen.
    def land(self):
        if self.at_end:
            return  # the score has paid the set: a redo after it lands through land_late
        last = self.state.stage_amt - 1
        earned = self.achievement if self.score_this_set else self.passed / ACHIEVEMENT_SCALE
        reward, self.credited = slot_reward(self.state, earned, self.credited,
                                            self.feedback_this_set, self.s == last)
        self.score += reward
        self.set_points += reward
        delta = value_of_stage(self.state, self.params, self.s, reward)
        self.state.rpe[self.s] = delta
        self.state.slot_delta = delta
        self.episode_deltas.append(delta)

    # A right redo after the score: what it earns lands as more reward on the set's last
    # step (value.late_reward), and the error carries into the next prompt and the next
    # start decision. With the score hidden it earns the fraction right's share. Returns the
    # reward, which the score shows.
    def land_late(self, p):
        n = self.state.stage_amt
        earned = 1.0 - p if self.score_this_set else 1.0 / ACHIEVEMENT_SCALE
        reward = ACHIEVEMENT_SCALE * earned / n
        delta = late_reward(self.state, self.params, n - 1, reward)
        self.episode_deltas.append(delta)
        self.state.slot_delta += delta
        self.state.total_reward += 1.0 - p
        self.score += reward
        return reward

    def after_feedback(self, correct, then):
        if self.at_end:
            # A redo after the score always shows how it went, and what it earned.
            points = f" +{self.end_points:.2f}" if correct and self.score_this_set else ""
            self.flash(("correct" if correct else "wrong") + points, then)
        elif self.feedback_this_set:
            self.flash("correct" if correct else "wrong", then)
        else:
            then()

    # The redo node (conditions.stage_conditions): after a miss shown with feedback on, or
    # after the score of a set played with it hidden. There it is the same node at the
    # set's last slot (Simulation._redoes_at_end), and the question is named, since its
    # miss was not shown when it happened.
    def ask_repeat(self):
        end = self.at_end
        slot = self.state.stage_amt - 1 if end else self.s
        value, cost = stage_conditions(self.state, self.params, slot, self.item)
        # Under a countdown it closes undecided at the deadline, and they move on
        # (model/pressure.py): fitted as censored at bound λ·BOUND and that deadline.
        deadline = float(self.v_countdown.get()) or None
        self.screen = "redo"

        def done(go, rt):
            self.screen = None
            rec = {"node": "stage", "session": self.session, "set": self.set_id,
                   "value": snap_value(value), "cost": snap_cost(cost), "GO": go, "RT": rt,
                   "stage": self.s, "item": self.item, "timestep": self.state.t,
                   "slot_delta": self.state.slot_delta}
            if deadline is not None:
                rec["deadline"] = deadline
            if end:
                rec.update(end=True, question=self.problem[0])
            self.state.first_passage.append(rec)
            self.store.append(rec)
            self.persist()
            if go == 1:
                self.reviewing = True
                self.source = "missed"
                self.redo = True
                self.show_problem() if end else self.next_stage()
            else:
                self.advance()

        prompt = (f"redo question {self.end_number}: {self.problem[0]}?" if end
                  else "try this one again?")
        self.decide_on("stage", value, cost, prompt, "repeat it", "move on", done, deadline)

    # Move to the next slot, or after the score to the next miss.
    def advance(self):
        if self.retire_pending is not None:
            self._retire(*self.retire_pending)
            self.retire_pending = None
        if self.at_end:
            self.next_end_redo()
            return
        self.s += 1
        self.next_stage()

    # Every slot played. With feedback hidden the set ends on its score, which is when its
    # reward lands (`land` paid it at the last slot). With end-of-set redos on, each new
    # question missed in it is then offered again, one prompt at a time, as with
    # Simulation(end_redos=True).
    def end_of_set(self):
        if self.feedback_this_set:
            self.end_episode(abandoned=False)
            return
        self.at_end = True
        # The error the score just produced is the whole set's, and it is what the first
        # redo prompt carries. A right redo adds what it earned (`land_late`).
        self.state.slot_delta = float(sum(self.episode_deltas))
        offer = self.v_end_redos.get() and self.v_show_choice.get()
        self.end_misses = list(self.unseen) if offer else []
        # Other answers have moved the ratings since those misses, so a quick right answer
        # to one cannot be undone as a typo.
        self.last_miss = None
        n = self.state.stage_amt
        self.store.append({"node": "score", "session": self.session, "set": self.set_id,
                           "passed": self.passed, "stage_amt": n, "points": self.set_points,
                           "score": self.score, "show_score": self.score_this_set,
                           "set_delta": self.state.slot_delta, "offered": len(self.end_misses)})
        self.screen = "score"
        self.persist()
        # How each question went (points only with the score shown), then the set's points
        # and the score, or with the score hidden how many were right.
        total = (f"+{self.set_points:.2f} this set, score {self.score:.2f}"
                 if self.score_this_set else f"{self.passed} of {n} right")
        self.show_summary(list(self.set_log), total, self.next_end_redo)

    # The next miss to offer after the score, or with none left the start prompt.
    def next_end_redo(self):
        self.screen = None
        if not self.end_misses:
            self.end_episode(abandoned=False)
            return
        self.item, self.problem, self.end_number = self.end_misses.pop(0)
        self.ask_repeat()

    def end_episode(self, abandoned):
        self.at_end = False
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

        self.decide_on("start", value, base + effort, "start the next set?",
                       "start the next set", "I'm done", done)

    # Put a decision on screen, keeping its conditions and outcome for the live model view.
    def decide_on(self, node, value, cost, prompt, yes_label, no_label, on_done, deadline=None):
        now = self.decision_now = {"node": node, "value": snap_value(value),
                                   "cost": snap_cost(cost), "prompt": prompt,
                                   "yes": yes_label, "no": no_label, "go": None, "rt": None,
                                   "deadline": deadline, "decided": False}

        def done(go, rt):
            if deadline is not None and (go is None or rt is None or rt >= deadline):
                go, rt = None, None  # the countdown ran out first
            now.update(go=go, rt=rt, decided=True)
            on_done(go, rt)

        self.ask_decision(prompt, yes_label, no_label, done, deadline)

    # ------------------------------------------------------------ owed questions

    @staticmethod
    def _take(pool, t, question):
        pool[t].remove(question)
        if not pool[t]:
            del pool[t]

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

    # ------------------------------------------------------------ records

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

    # ------------------------------------------------------------ closing

    # Python is exiting and the app never closed. File writes only: the frontend may be gone.
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

    # The candidate setting nearest what the settings are now.
    def current_environment(self):
        t, fb = self.v_target.get(), self.v_feedback.get()
        redo = self.v_show_choice.get()
        r = min(REVIEW_LEVELS, key=lambda x: abs(x - self.v_review.get()))
        # With review at 0 nothing comes back, so "skipped come back" means nothing.
        rs = self.v_return_skipped.get() and r > 0
        # End-of-set redos mean something only with redo prompts on and feedback hidden,
        # a forced repeat only with them off and feedback on, a countdown only with them on.
        end = self.v_end_redos.get() and redo and not fb
        forced = not redo and fb and self.v_forced.get() == "repeat"
        countdown = self.v_countdown.get() if redo else 0.0
        n = min(STAGE_LEVELS, key=lambda x: abs(x - self.v_stages.get()))
        shape = (redo, end, forced, self.v_show_score.get(), n)
        # What kind of setting it is decides the match first, then feedback and target.
        # Settings with redo prompts or a forced repeat have no review.
        return min(APP_CANDIDATES, key=lambda e: 100 * sum(
            a != b for a, b in zip((e.redo_prompts, e.end_redos, e.forced, e.show_score,
                                    e.stage_amt), shape))
            + (e.feedback != fb) * 10 + abs(e.target_p - t)
            + (abs((e.countdown if e.pressure else 0.0) - countdown) if redo else 0)
            + ((e.review_rate != r) * 10 + (e.return_skipped != rs) * 10
               if not (redo or forced) else 0))

    # What the controller is doing, for the panel: off, planning, or on and waiting for its
    # next plan, which comes at the start of a set once PLAN_EVERY start decisions have
    # passed since the last one (a saved controller keeps counting across launches).
    def controller_text(self):
        on = self.v_controller.get()
        if self._plan_future is not None:
            return "planning..." if on else "off (the plan under way will be discarded)"
        if not on:
            return "off"
        ctl = self.controller
        if ctl is None or ctl.last_plan is None:
            head = "on, first plan when the next set starts"
        else:
            left = ctl.min_spacing - (len(self.state.initiation_passages) - ctl.last_plan)
            head = ("on, plans when the next set starts" if left <= 0 else
                    f"on, next plan in {left} set{'s' if left > 1 else ''}")
        return head + (f". Last plan: {self.controller_note}" if self.controller_note else "")

    def maybe_plan(self):
        if not self.v_controller.get() or self._plan_future is not None:
            return
        if self.controller is None:
            self.controller = AppController(start=self.current_environment(),
                                            min_spacing=PLAN_EVERY)
        if not self.controller.due(self.state):
            return
        # The settings may have been moved by hand since the last plan.
        self.controller.env = self.current_environment()
        # Copy now: the worker pickles later, while this thread keeps logging.
        self._plan_future = self.executor().submit(
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
            self.controller_note = f"failed: {exc}"
            self.refresh_stats()
            return
        if not self.v_controller.get():
            self.controller_note = "discarded: the controller was turned off while it ran"
            self.refresh_stats()
            return

        env, est = update.chosen, update.estimate
        self.v_target.set(env.target_p)
        self.v_review.set(env.review_rate)
        self.v_feedback.set(env.feedback)
        self.v_return_skipped.set(env.return_skipped)
        self.v_show_choice.set(env.redo_prompts)
        self.v_end_redos.set(env.end_redos)
        self.v_countdown.set(env.countdown if env.pressure else 0.0)
        self.v_forced.set("repeat" if env.forced else "continue")
        self.v_stages.set(env.stage_amt)  # these two from the next set
        self.v_show_score.set(env.show_score)
        self.controller_note = (
            f"f {est.f:.2f}, k {est.k:.2f}, λ {est.lam:.2f}, patience {est.patience:.1f}s, "
            f"T0 {est.t0:.2f}s "
            f"from {est.rows} rows ({est.note}). Next: {env} -- {update.reason}.")
        self.store.save_controller(self.controller)
        self.store.append({"node": "controller", "session": self.session, "set": self.set_id,
                           "rows": est.rows, "f": est.f, "k": est.k,
                           "patience": est.patience, "t0": est.t0, "note": est.note,
                           "target": env.target_p, "review_rate": env.review_rate,
                           "feedback_on": env.feedback, "return_skipped": env.return_skipped,
                           "redo_prompts": env.redo_prompts, "end_redos": env.end_redos,
                           "countdown": env.countdown if env.pressure else 0.0,
                           "forced": env.forced, "stages": env.stage_amt,
                           "show_score": env.show_score, "lam": est.lam,
                           "switched": env != update.previous,
                           "reason": update.reason})
        self.refresh_stats()

    def executor(self):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1)
        return self._pool

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    # Every decision so far: the records, and as pyddm's fit reads them.
    def decisions(self):
        return self.state.initiation_passages + self.state.first_passage

    def frame(self):
        return passages_to_frame(self.decisions())
