"""The simulated learner: plays sets, decides whether to start and redo, gives up.

It follows the app's rules for what can be recorded. A session holds at most one "stop":
saying it ends the session, and the time away until the person comes back (cold) is not
recorded, because nothing in the app sees it. Time is counted in ticks: one start
decision in a session, or one chance to come back while away.
"""
import random

import numpy as np

from .accumulator import initiation_decision, stage_decision
from .competence import draw_outcome, settle_skipped, sigmoid, success_prob, update_ability
from .conditions import cost_components, stage_conditions, value_at_choice_point
from .params import ModelState, UserParams
from .persistence import conditions, draw_quit_time, draw_solve_seconds, quit_drift
from .questions import choose_type
from .value import ACHIEVEMENT_SCALE, slot_reward, value_of_stage

MAX_TICKS = 1_000_000  # a safety cap; the lapse mixture keeps every return chance above ~1%

# Who decides whether a missed question comes back: the controller at review_rate per
# slot (scheduled), the learner at the repeat node (choice), or nobody (quiet).
REVIEW_MODES = ("scheduled", "choice", "quiet")

# What a set is worth: right answers weighted by (1 - p) (achievement), or the plain
# fraction right (accuracy, the earlier model).
REWARDS = ("achievement", "accuracy")


class Simulation:

    def __init__(self, params: UserParams, ability=None, reentry_cost=None,
                 review_mode="scheduled", review_rate=0.25, diff_spread=0.0,
                 target_p=0.75, feedback=True, ratings=None, reward="achievement",
                 skips=False, return_skipped=False):
        if review_mode not in REVIEW_MODES:
            raise ValueError(f"review_mode must be one of {REVIEW_MODES}, got {review_mode!r}")
        if not 0.0 <= review_rate <= 1.0:
            raise ValueError(f"review_rate must be in [0, 1], got {review_rate}")
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {reward!r}")

        self.params = params
        self.review_mode = review_mode
        self.review_rate = review_rate
        self.target_p = target_p # serve the question type whose chance is nearest this
        self.feedback = feedback # whether right/wrong is shown during a set
        self.ratings = ratings   # what the environment knows; None: use the true difficulties
        self.reward = reward
        self.skips = skips       # whether the learner can give up on a question (persistence.py)
        self.return_skipped = return_skipped # whether given-up questions come back through review
        self.cold_start = False # whether the set about to be played was started from outside a session
        self.state = ModelState(t=1)
        if ability is not None:
            self.state.ability = ability
        if reentry_cost is not None:
            self.state.reentry_cost = reentry_cost
        if diff_spread > 0:
            # Per-stage difficulty, drawn once per learner around state.diff.
            st = self.state
            st.stage_diffs = np.clip(
                [st.diff + random.uniform(-diff_spread, diff_spread)
                 for _ in range(st.stage_amt)],
                0.0, None,
            )

    def step_stage(self, s, reward=0.0):
        delta = value_of_stage(self.state, self.params, s, reward)
        self.state.rpe[s] = delta
        return delta

    # (item, "missed" or "skipped") for a revisit taking this slot, or (None, None).
    def _review_item(self, carry):
        state = self.state
        if carry is not None:
            return carry, "missed" # a redo the learner already asked for
        pool = [(i, "missed") for i in state.missed]
        if self.return_skipped:
            pool += [(i, "skipped") for i in state.skipped]
        if self.review_mode == "scheduled" and pool and random.random() < self.review_rate:
            return random.choice(pool)
        return None, None

    # Work on a question until they answer or give up. True if they gave up.
    def _gives_up(self, item, p_true, review):
        state, params = self.state, self.params
        value, cost = conditions(p_true, state.stage_amt, state.slot_delta)
        solve = draw_solve_seconds(p_true)
        # A race in seconds: time needed to solve against time taken to give up.
        quit_at = (draw_quit_time(quit_drift(params.f, params.k, float(value), float(cost)))
                   * params.patience)
        gave_up = quit_at < solve
        # The record keeps the chance as the environment knew it, not the truth.
        known = (self.ratings.p(item) if self.ratings is not None and state.type_diffs is not None
                 else p_true)
        state.persistence.append({
            "p": known, "rt": min(quit_at, solve), "skipped": gave_up,
            "stage_amt": state.stage_amt, "item": item, "review": review,
            "delta": state.slot_delta,
            "episode_id": state.episodes + state.abandoned + 1,
        })
        return gave_up

    # The item for a fresh slot: slot `s` itself, or a question type near target_p.
    def _new_item(self, s):
        state = self.state
        if state.type_diffs is None:
            return s
        if self.ratings is not None:
            probs = self.ratings.probs()  # what the environment believes about them
        else:
            probs = [sigmoid(state.ability - float(d)) for d in state.type_diffs]
        return choose_type(probs, self.target_p)

    # The redo prompt after a miss at slot `s`. True if they take the question again.
    def _redoes(self, s, item):
        state, p = self.state, self.params
        value, cost = stage_conditions(state, p, s, item)
        choice = stage_decision(value, cost, p.f, p.k, t0=p.t0)
        choice["stage"] = s
        choice["item"] = item
        choice["episode_id"] = state.episodes + state.abandoned + 1
        choice["timestep"] = state.t
        choice["slot_delta"] = state.slot_delta  # the error the miss just produced, as the app logs
        state.first_passage.append(choice)
        return choice["chose"] == 1  # a slow choice is still made

    # One episode: exactly `stage_amt` slots, one question each.
    def run_episode(self):
        state = self.state
        last = state.stage_amt - 1
        cold, self.cold_start = self.cold_start, False

        deltas = []
        passed = wrong = review_slots = 0
        achievement = credited = 0.0  # earned this set, and how much of it has landed
        state.slot_delta = 0.0        # no slot precedes the first one
        carry = None  # a missed question the learner chose to redo, owed the next slot

        for s in range(state.stage_amt):
            # A revisit displaces the new question that would have taken this slot.
            item, source = self._review_item(carry)
            carry = None
            review = source is not None
            if not review:
                item = self._new_item(s)
            review_slots += review

            p_item = success_prob(state, item)
            missed_new = False # a fresh question answered wrong, with the miss shown
            if self.skips and self._gives_up(item, p_item, review):
                if s == 0 and cold:
                    # Came back, looked at the first question and gave up: that is
                    # leaving. The set is abandoned and they are away again.
                    state.persistence[-1]["left"] = True
                    state.abandoned += 1
                    state.in_session = False
                    state.last_set_delta = 0.0 # they walked out; there was no set to judge
                    return deltas
                # The slot is spent and earns nothing. A skip is neither right nor
                # wrong, so the ratings do not move.
                state.skips += 1
                if not review:
                    state.skipped.append(item) # one given up on again stays where it was
            else:
                success = draw_outcome(state, item)
                # Only a question's first showing rates its type.
                if self.ratings is not None and state.type_diffs is not None and not review:
                    self.ratings.update(item, success)
                if source == "skipped":
                    settle_skipped(state, item, success)
                else:
                    update_ability(state, item, success, review)
                state.challenge += 1.0 - p_item

                if success:
                    passed += 1
                    achievement += 1.0 - p_item # a hard question right is worth more
                else:
                    wrong += 1
                    # No redo prompt with feedback hidden: offering one only after a
                    # miss would reveal the miss.
                    missed_new = self.feedback and not review and s < last

            # The outcome lands as reward the moment they are shown it, and at the
            # end-of-set score if they are not: a miss they see is a slot that paid
            # nothing against a value that expected something, which is the negative
            # error. A miss they never see cannot produce one.
            # `slot_reward` works on the achievement scale, so the accuracy reward --
            # the earlier model, kept for comparison -- is divided back out of it.
            earned = (achievement if self.reward == "achievement"
                      else passed / ACHIEVEMENT_SCALE)
            self.achievement_so_far = achievement  # for learners that assume the achievement reward
            reward, credited = slot_reward(state, earned, credited, self.feedback, s == last)
            delta = self.step_stage(s, reward=reward)
            state.slot_delta = delta
            deltas.append(delta)

            # The redo node fires after the error, not before it: they see the miss,
            # that lands, and then they decide whether to take the question again.
            if missed_new and self.review_mode == "choice" and self._redoes(s, item):
                carry = item

        # What the next start decision sees of the set just played.
        n = state.stage_amt
        state.last_review_load = review_slots / n
        state.last_episode_effort = 1.0 + state.last_review_load # effort is review load, not duration
        state.last_episode_passed = passed / n
        state.last_achievement = achievement / n
        state.total_reward += achievement
        state.last_feedback = int(self.feedback)
        state.last_seen_miss = wrong / n if self.feedback else 0.0
        state.last_unseen_miss = 0.0 if self.feedback else wrong / n
        # How the set went against what was predicted of it, carried into the decision
        # to start another one.
        state.last_set_delta = float(sum(deltas))
        state.episodes += 1
        return deltas

    def initiate(self):
        state, p = self.state, self.params

        base, effort = cost_components(state)
        record = initiation_decision(
            value=value_at_choice_point(state),
            cost=base + effort,
            f_val=p.f,
            k_val=p.k,
            t0=p.t0,
        )
        record["episodes_done"] = state.episodes
        record["episode_id"] = state.episodes + state.abandoned + 1
        record["in_session"] = state.in_session
        record["cost_base"] = base
        record["cost_effort"] = effort
        record["last_episode_effort"] = state.last_episode_effort
        record["last_episode_passed"] = state.last_episode_passed
        record["last_achievement"] = state.last_achievement
        record["last_review_load"] = state.last_review_load
        record["last_set_delta"] = state.last_set_delta
        record["feedback"] = state.last_feedback
        record["missed_count"] = len(state.missed)
        record["timestep"] = state.t
        state.initiation_passages.append(record)

        # What they do, even past the deadline; the fit reads a slow choice as undecided.
        if record["chose"] != 1:
            # "Stop" ends the session, as "I'm done" closes the app. What follows is
            # time away until they come back cold, and none of it is recorded.
            state.stuck += 1
            state.in_session = False
            return False
        return True

    # Away, a chance to come back, at cold cost. It is not recorded: in the app, not
    # opening the task leaves no trace, and opening it is a decision whose "RT" is hours.
    # It costs a tick and is not data. Coming back starts a set cold.
    def come_back(self):
        state, p = self.state, self.params
        base, effort = cost_components(state)
        draw = initiation_decision(value_at_choice_point(state), base + effort, p.f, p.k, t0=p.t0)
        state.away += 1
        if draw["chose"] != 1:
            return False
        state.in_session = True
        self.cold_start = True
        return True

    # One tick: a recorded start decision in a session, or a chance to come back when
    # away. Either way, a GO starts a set.
    def tick(self):
        went = self.initiate() if self.state.in_session else self.come_back()
        self.state.t += 1
        if went:
            self.run_episode()
        return went

    def run(self, max_decisions=None, max_episodes=None, max_ticks=MAX_TICKS):
        state = self.state

        # The learner is already in the task; the first episode is not gated.
        state.in_session = True
        self.run_episode()

        while True:
            self.tick()
            if max_decisions is not None and len(state.initiation_passages) >= max_decisions:
                break
            if max_episodes is not None and state.episodes >= max_episodes:
                break
            if state.t >= max_ticks:
                break

        return state
