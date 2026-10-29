# Modeling motivation through ordinary task behavior

A computational model of task initiation: whether a person starts the next piece of a
task, and how long they take to decide. The person plays short sets of arithmetic
questions. From their choices, how long each choice takes, and how long they work on a
question before giving up, the model estimates

- `f`, **reward sensitivity**: how strongly the expected payoff pulls them in;
- `k`, **effort aversion**: how strongly effort pushes them away;
- **patience**: how quickly they give up on a question.

How they react to the set they just played needs no parameter of its own. It reaches the
next decision as prediction error, scaled by `f`.

The goal is to use these estimates to reshape the task for each person (how hard the
questions are, how often mistakes come back, whether right/wrong is shown, whether skipped
questions come back) so that engaging becomes easier than stopping.

This is an active research project. It runs on simulated people, and on human sessions
played in the app (`temporal/app/`) and recorded across sessions. No estimate from a real
person has been validated yet.

This README is the project's lab notebook. It records every current measurement, the
conditions it was taken under, and why each design decision was made. Results from earlier
model versions are in `RETIRED.md`.

---

## Contents

- [In short](#in-short)
- [Running the app](#running-the-app)
- [Terms used here](#terms-used-here)
- [The model](#the-model)
- [The controller](#the-controller-controllerpy)
  - [The app's controller](#the-apps-controller-appcontroller-since-2026-10-29)
- [Human sessions](#human-sessions-temporalapp)
- [Current results](#current-results)
  - [Reporting conventions](#reporting-conventions)
  - [Recovering f and k](#recovering-f-and-k)
  - [Fitting the discount factor](#fitting-the-discount-factor-2026-10-28)
  - [The bound can be fitted: f, k, γ, BOUND and T0 together](#the-bound-can-be-fitted-f-k-γ-bound-and-t0-together)
  - [Measuring the learning rate and the reward](#measuring-the-learning-rate-and-the-reward-2026-10-28)
  - [Which difficulty and whether to show feedback](#which-difficulty-and-whether-to-show-feedback)
  - [Measuring against planning for reward](#measuring-against-planning-for-reward)
  - [Four controllers, all five parameters fitted](#four-controllers-all-five-parameters-fitted-2026-10-29)
  - [Planning with fitted γ and bound](#planning-with-fitted-γ-and-bound-2026-10-28)
  - [Feedback in blocks, with redos mid-set or at the end of the set](#feedback-in-blocks-with-redos-mid-set-or-at-the-end-of-the-set-2026-10-28)
  - [A countdown on the redo prompt](#a-countdown-on-the-redo-prompt-2026-10-28)
  - [Every parameter, every setting](#every-parameter-every-setting-2026-10-29)
- [Methodological notes](#methodological-notes)
- [Known limitations](#known-limitations)
- [Next steps](#next-steps)
- [Reproducing every result](#reproducing-every-result)
- [Files](#files)
- [Code layout, tests and speed: since 2026-10-11](#code-layout-tests-and-speed-since-2026-10-11)

---

## In short

What has been shown, in simulation:

- **`f` and `k` can be recovered from the decisions the app records**, and so can γ, the
  accumulator's bound and the non-decision time `T0`, all five together. Across 100
  simulated people, each with 300 start decisions plus their redo decisions, r is 0.946 to
  0.991 and γ is measured to about ±0.03 ("The bound can be fitted").
- **Start decisions alone barely measure `k`.** A person only ever decides to start the next
  set in the middle of a session, at the same low cost, so cost hardly varies. Redo
  decisions, whose cost depends on how hard the missed question was, carry most of what
  measures `k` ("How value and cost together decide what can be measured").
- **A constant held at the wrong value biases the estimates.** With γ held at 0.9, a person
  whose γ is 0.7 gets `f` 62% too low, and one whose γ is 0.95 gets it 57% too high.
  Fitting γ removes this. On the same data, the app's current fit (γ and the bound pinned)
  has `f` RMSE 0.469, against 0.062 with all five free.
- **Which reward a person values can be measured.** Comparing right answers weighted by
  difficulty against the fraction right picked the true one for 63 of 64 simulated people.
  The learning rate α is measured only loosely, to about a factor of 5.
- **Each person has a plateau of difficulties that keep them engaged.** Questions that are
  too easy lose people, because right answers pay almost nothing; questions that are too
  hard lose the effort-averse. Showing right/wrong raises engagement at most difficulties,
  and more for people who discount the future steeply.
- **Planning to measure finds `k`; keeping people in the task is what finds γ.** On 128
  people drawn over all five parameters, all five fitted, a controller that plans for
  information about `f`, `k` and γ measures `k` to RMSE 0.20 in 300 ticks, against 0.58 for
  one that maximizes time in the task and 0.87 at the fixed default. γ is measured only
  for people who give many start decisions, and the engagement maximizer gives twice as
  many, so across the 128 neither measures γ better ("Four controllers, all five
  parameters fitted"). When every person has γ 0.9 and `BOUND` 1.5, `InfoController`
  measures `k` to 0.156 against 0.508 under reward planning, at no detectable cost in
  reward.

Not yet shown: anything about a real person. The app's fit still holds γ at 0.9 and the
bound at 1.5, and collecting data from other people is not approved yet.

---

## Running the app

```bash
uv sync
```

Run every command from the repo root. `uv run` uses the project's environment; inside an
activated `.venv`, plain `python` works too.

```bash
uv run python -m temporal.app.desktop                                 # in a window; data in data/
TEMPORAL_RESEARCHERS=you@example.com uv run python -m temporal.app.web # in a browser at http://127.0.0.1:8000
TEMPORAL_PANEL=1 uv run python -m temporal.app.web                    # as a guest, with the dev panel
```

The web app keeps only researchers' records until `TEMPORAL_COLLECT=1` ("On a website").
With the dev panel open, "show the accumulator next to the task" draws the model's live
prediction beside each start and redo decision.

---

## Terms used here

| term | meaning |
|---|---|
| set, slot | a set is `stage_amt` questions (4 by default), one in each slot |
| start decision | after a set: start the next one (GO) or stop. In the app, stopping is "I'm done", which ends the session |
| redo decision | after a miss the person is shown: take the question again, or move on. Offered only when redo prompts are on; with feedback hidden, after the set's score, when end-of-set redos are on too |
| giving up | leaving a question while working on it: a skip, or closing the app on it |
| warm and cold starts | a start decision in the middle of a session (cost `WARM_COST`), and coming back after leaving (cost `reentry_cost`). Only warm starts are recorded |
| `f`, `k` | reward sensitivity and effort aversion. The drift of every decision is `f · value − k · cost` |
| value, cost | what an option is worth to the person, and the effort it takes |
| `δ` | prediction error: what a slot paid minus what was expected of it |
| target | the chance of a right answer each question is aimed at |
| `SIGMA`, `BOUND` | the accumulator's noise (0.7, always fixed) and its bound (1.5 unless fitted) |
| `T_DUR` | the deadline, 7 s. A decision not made by then is *undecided*, and it is kept as data |
| `T0` | non-decision time, added to every decision's time |
| γ, α | the learner's discount factor (0.9 unless fitted) and learning rate (0.05 unless fitted) |
| patience | seconds per unit of model time when giving up; how fast someone gives up |
| tick | one step of a simulation that measures engagement: a start decision in a session, or one chance to come back while away |
| question-type task, single-`diff` task | the app's task, with question types rated per person; and the older task with one difficulty, which the recovery pipelines use |
| shadow learners | TD learners that replay a person's rewards at other values of γ (or α), giving the value each decision would have had |
| `Controller`, `InfoController`, `AppController` | the controllers that plan for reward and plan to measure `f` and `k`, and the app's: `InfoController` over every setting the app has, since 2026-10-29 |
| engagement maximizer, information planning | the two planners of "Four controllers" (`experiments/controller_full.py`): the setting that keeps a person in the task for the most ticks, and the one that would most sharpen `f`, `k` and γ. Both fit all five parameters |
| ticks in the task | the share of ticks a person spends in a session rather than away: 1 − time away |
| λ | the share of `BOUND` a person keeps on a redo prompt under a countdown; fitted by the app |
| score | the TD reward landed since the app was opened, as the app shows it |
| V4 | the current model version. Earlier ones are defined in `RETIRED.md` |
| beyond 2 SE | the rule for every claim: a paired difference counts only if it exceeds twice its standard error |
| claims fixed before the run | what a run was meant to show, written down before any of its results were read |

---

## The model

Four layers: the task (the environment), the learner, the decisions, and the fit that
inverts all three. The decision comes first, since the rest feeds it.

### The main equation

After each set, the person decides whether to start the next one or stop. Their drive
toward starting is

```
Drive = f · Value − k · EffortCost
Value = V(0) + Σ δ                  Σ δ: the last set's prediction error, carried
```

- `Value` is what the person expects the next set to be worth, `V(0)`, plus how the set
  just played turned out against what was predicted of it.
- `EffortCost` is how much it takes to get going again.
- `f` sets how strongly value pulls, and `k` how strongly cost pushes away.

The drift has only these two parameters. What the last set did to the person has no term
of its own: it is prediction error, which reaches the decision through `Value` and is
scaled by the same `f`. Seeing that you got a question wrong is a slot that paid nothing
against a value that expected something, a negative `δ`. A miss you are not shown cannot
produce one. "Feedback is timing, not reward" covers what this gains and what it gives up.

Drive does not decide instantly. It is the drift of a noisy accumulator (a drift-diffusion
model) that moves toward one of two bounds, GO (start the next set) or STAY (stop), and
gives both a choice and a reaction time. The reaction times are what separate the
parameters: choices alone cannot tell a high `f` from a low `k`, but the shape of the RT
distribution can.

### The task

Each set has `stage_amt` slots (default 4), one question in each.

**Question types, rated per person** (`model/questions.py`). Each question type has a rough
prior difficulty in logits:

| type | example | prior |
|---|---|---|
| add, 1-digit | 7 + 5 | −2.0 |
| add, 2-digit | 47 + 38 | −0.5 |
| subtract, 2-digit | 83 − 47 | 0.0 |
| times tables | 8 × 7 | 0.0 |
| multiply, 2-digit × 1-digit | 64 × 7 | 1.0 |
| subtract, 3-digit | 702 − 358 | 1.0 |
| divide, 3-digit ÷ 1-digit | 952 ÷ 7 | 2.0 |
| multiply, 2-digit × 2-digit | 47 × 38 | 2.5 |
| multiply, 3-digit × 2-digit | 347 × 58 | 3.5 |

The prior is only a starting guess. After every answer an Elo-style update re-rates both
the person and the question type,

```
p = sigmoid(ability − difficulty[type])
ability         += K_a · (correct − p)
difficulty[type] −= K_d · (correct − p)        steps shrink as answers accumulate
```

so a type's rating comes to describe how hard it is *for this person*. Each slot serves a
type whose predicted chance is nearest the current **target chance**. Types within 0.1 of
the best match are all eligible, so near-equivalent types all get rated. Hard is relative
to the person by construction.

Only a question's first showing rates it. A review or redo of the same question is not a
fresh sample of its type; without this rule, one question shown many times could be all
the evidence behind a person's chance on its whole type.

**What a set is worth.** A set's reward is its right answers, each weighted by how hard it
was, times a scale:

```
reward = ACHIEVEMENT_SCALE · Σ_right (1 − p) / stage_amt        ACHIEVEMENT_SCALE = 4
```

A trivial right answer earns almost nothing, so a set of easy wins is worth little and a
person can stop out of boredom. The previous reward, the fraction right, could not
represent that: the model believed the easiest setting was always the most rewarding, so a
person served only single-digit questions was never offered anything harder (the V1
comparison is in `RETIRED.md`). When each part of the reward lands is under "The learner".

The scale is a calibration. The expected reward per question, `p(1 − p)`, peaks at 0.25;
scale 4 makes that peak 1.0, the value of a perfect set under the old reward, so `f` keeps
its earlier meaning. Since drive is `f · Value`, changing the scale is the same as
rescaling `f`.

**What a set costs.** Its review load, the share of slots spent going back over mistakes,
raises the cost of the next start decision.

**Review, redos and skips.**

- A slot normally holds a new question. It can instead bring back one missed earlier,
  which displaces the new question. Sets never get longer, so review trades against
  coverage. Who decides this is `review_mode` (under "The learner").
- When right/wrong is hidden for a set, there is no redo prompt either. It only ever
  appears after a miss, so offering it would reveal the miss. Owed questions still come
  back through review.
- A skipped question uses its slot and earns nothing. It is kept apart from the missed
  questions, and whether skipped questions join what review brings back is a setting
  (`return_skipped`, one of the controller's levers). One that comes back and is answered
  right counts as a correction; answered wrong, it is owed like any miss; skipped again,
  it stays skipped.
- When the person decides redos (`review_mode = "choice"`), effort aversion produces a
  spiral that nothing in the model spells out: high `k` declines the redos, so less is
  learned, so fewer questions are answered right, so `V` stays low, so re-engagement drops.
  It was measured under V1 (`RETIRED.md`) and has not been re-measured.

**Rules only the app applies.** The simulator's owed items are question types rather than
individual questions, so these rules have no simulator counterpart.

- *Typos are not misses.* A miss put right on the redo within 2 s (`SLIP_SECONDS`) is a
  slip: the ratings are restored to before the miss, the question is rated as answered
  right, it is not owed, and it does not count as a miss. A wrong answer fixed in under a
  second is a slip of the keyboard, not a gap in skill.
- *A question is never shown twice in one set*, except as a redo the person chose.
- *Questions that keep failing retire.* One that comes back and is missed or skipped 3 times
  (`RETIRE_AFTER`; redos do not count) stops coming back, and a `retired` event is logged.
  Without it, a question someone keeps skipping could come back again and again.

**Two versions of the task.** The recovery pipelines (`python -m temporal recovery`) use an
older task with a single difficulty, `diff` (with optional per-stage `diff_spread`). The
controller's simulated people and the app use the question types above.

### The learner

The learner keeps two quantities apart by design: what the task is worth to it, and its
own competence.

**Value** is learned by TD(0) with linear function approximation:

```
V(s) = θ · φ(s)                    φ(s) = [1, stage position]
δ    = r + γ·V(s+1) − V(s)         θ ← θ + α·δ·φ(s)
```

At a set's last slot, `V(s+1)` is `V(0)`, the start of the next set (the continuing
bootstrap; see "Value bootstrap", below). `φ` has no difficulty feature. It had one, which
did no work: `diff` never varies during learning, so `θ[1]` collapsed to exactly `θ[0]·d`,
perfectly collinear with the intercept, and contributed ~1% of `V(0)`.

**A slot's reward arrives when the person is shown it.** With feedback on, a question's
worth lands at its slot: `ACHIEVEMENT_SCALE · (1 − p) / stage_amt` if they got it right,
nothing if not. With feedback off nothing lands until the end-of-set score, which pays the
whole set at once: one lump of `ACHIEVEMENT_SCALE · achievement / stage_amt` at the last
slot, the schedule every set had before 2026-09-21. A set totals the same either way.
Feedback moves *when* the reward arrives, not how much (`value.slot_reward`), and that
timing is the whole of feedback's effect on motivation.

**Competence**, in the simulator, grows only from correcting what was missed:

```
P(success) = sigmoid(ability − difficulty)       ← Rasch / IRT
outcome    ~ Bernoulli(P(success))               ← per question
ability   += CORRECTION_GAIN   only when a question answered wrong (or skipped)
                               earlier is shown again and answered right
```

A first-time right answer teaches nothing, and neither does a failure; only a correction
does (`competence.update_ability`, `competence.settle_skipped`). This is deliberate
practice rather than error correction: working a question out between sessions, without
being shown it again, teaches nothing here.

For a real person nothing is simulated. The app *measures* competence with the Elo ratings
above, and the controller's rollouts start from that measurement. The correction rule is
how a simulated person's true ability changes; the ratings are what the environment can
know about it. The controller only ever sees the ratings, never a simulated person's
truth.

Competence is kept off `δ` because they are different learning systems. Prediction error
teaches value; competence comes from repetition and error correction. Driving skill from
`max(0, δ)` would stop competence growing as soon as predictions became accurate, which is
the wrong direction: expertise keeps improving long after outcomes become predictable.

**Who decides whether a missed question comes back** is `review_mode` on `Simulation`:

| mode | who decides | learning gated by |
|---|---|---|
| `scheduled` (default) | the controller, at `review_rate` per slot | the schedule |
| `choice` | the learner, offered a redo after each miss | `k` |
| `quiet` | nobody; missed questions never return | nothing is learned |

This choice is what keeps `k` a motivational parameter. When the learner decides, `k` sets
the learning rate as well as the willingness to start, and effort cost and skill can no
longer be manipulated separately (V1 measurement in `RETIRED.md`).

### The three decisions

Starting the next set and redoing a miss use the **same accumulator** (same drift form,
noise, bound and deadline) fed different conditions. They are two observations of the same
person, and their draws stack into one fit. `GO = 1` means "took the effortful option" in
both. Giving up on a question shares `f` and `k` with them but is a separate process with
one bound.

**Starting the next set.**

- Value is `V(state, 0)` plus the last set's total prediction error, `Σ δ`, clipped to
  `±DELTA_CLIP` so the condition grid stays bounded. The error is carried at full weight;
  there is no fitted carry coefficient.
- Cost is `structural(in_session) + EFFORT_WEIGHT · last_episode_effort`. The structural
  part is `WARM_COST` (0.05) in a session, and `state.reentry_cost` after leaving (the
  cold-start profile: `low` 0.3, `mid` 0.6, `high` 1.0). The effort part is `EFFORT_WEIGHT`
  (0.1) times `1 + review load` of the last set. Every recorded start decision is in a
  session, so the cold-start cost only sets how long a person stays away.

**Redoing a miss.** This decision comes after a missed new question, when the learner
decides redos and feedback is shown. It comes *after* the miss has landed as reward: the
person sees how the question went, that produces the error, and then they decide whether to
take it again.

- Value is the slot's share of the terminal reward, discounted by the slots to come, plus
  that error. So missing an easy question, the bigger surprise, discourages the redo more
  than missing a hard one.
- Cost is how hard the missed question is for them: `REPEAT_COST_SCALE` (0.15) times the
  expected number of attempts, `1/P(success)`, capped at 8. Cost varies more here than
  anywhere else, so this decision is what measures `k`.

**The accumulator's constants.**

- The noise is fixed at `SIGMA = 0.7`. As in the standard DDM, it sets the unit everything
  else is measured in. It cannot be fitted together with the bound: scaling drift, bound
  and noise by one factor changes nothing observable.
- The bound is `BOUND = 1.5` in every fit except where it is fitted per person ("The bound
  can be fitted" and the results that build on it).
- The deadline `T_DUR = 7.0` s is a cutoff of the analysis, not a property of the person,
  so it is never fitted: a person who decides slowly is described by their bound and
  non-decision time. It was 3.0 s until 2026-09-13 ("The deadline must sit past human
  decision times").

#### Giving up on a question

A person can skip a question, but only after trying it; there is no separate "answer it or
skip it" choice. While they work on it, the same drive runs the other way:

```
quit drift = k · cost − f · value
value = ACHIEVEMENT_SCALE · p(1 − p) / stage_amt + δ  what getting it right would add,
                                                      plus the error at the slot just left
cost  = REPEAT_COST_SCALE · min(1/p, 8)               the effort it takes
```

with `p` their chance on that question. The quit drift feeds an accumulator with one bound,
the same `SIGMA` and `BOUND`, and no deadline. If it reaches the bound before they reach an
answer, they give up: a skip, or leaving if they close the app on that question. The time
to give up then has a closed form, the inverse Gaussian (Wald) distribution
(`model/persistence.py`).

- **Patience is per person.** It turns model time into seconds (seconds per unit of model
  time, default 10 s), and someone who skips at a glance has a small one. It was a fixed
  10 s at first, which made skips of a few seconds all but impossible.
- **Answered questions count too.** An answer after 25 s says "kept going at least 25 s", so
  it enters the likelihood through the survival function, as a right-censored time to give
  up. Every question is an observation, skipped or not.
- **2% of questions are lapses**: given up on, or kept at, for reasons the model does not
  describe, at a uniform time within 60 s. Then one odd skip, such as dropping an easy
  question after two seconds, cannot decide the estimate on its own.
- **Leaving from the first question after coming back** is giving up on it too. In the
  simulator, giving up on the first question of a set started cold abandons the set, and
  the person is away again.
- **A skip is neither right nor wrong**, so it does not move the ratings.
- **The last question carries into this one.** `δ` is the prediction error from the slot
  just left. Seeing that you missed the last one makes giving up on this one more likely,
  and seeing that you got a hard one right makes it less likely. With feedback hidden there
  is no error to carry, so `δ` is 0 and only `p` matters.
- **It is used for patience only.** `f` and `k` come from the other two decisions; answer and
  skip times add nothing to them ("Skips do not help estimate `f` and `k`").

Simulated people also need a time to solve each question. It is lognormal with a median of
`4 s / p^0.7`: about 4 s for a sure thing and 20 s at a 10% chance. Real people take the
time they take.

### Sessions, and what is recorded

Since V4 (2026-10-25) the simulated person follows the app's rules, so the simulator
records only what the app could:

- **One stop ends the session.** "I'm done" closes the app, so a session holds at most one
  stop. After it the person is away. Each tick away is a chance to come back, decided by
  the same accumulator at the cold-start cost, and none of it is recorded: not opening the
  app leaves no trace, and opening it is a decision whose "RT" is hours. Coming back starts
  a set cold, and the next recorded start decision is the warm one after it.
- **A slow choice is still made.** A draw that reaches no bound by `T_DUR` is recorded as
  undecided, which is how the fit reads a slow human click. The person keeps deliberating:
  they end up at the bound the same diffusion reaches after the deadline
  (`accumulator._upper_after_deadline`, exact for constant drift), and that is what they
  do. The same holds at the redo prompt.

Before V4 the simulator recorded every start draw: up to three stops in a row inside a
session, then every cold start until the next GO. No person can produce those rows, and
they carried most of the start decision's cost variation. Every earlier simulated result
was measured that way; those results are in `RETIRED.md`, with an account of the flaw ("The
recording flaw in V1 to V3"). `tests/test_sessions.py` checks that there is a set between
every two recorded start decisions, and that none of them is cold.

### Fitting

PyDDM's `Model.fit()` solves the Fokker-Planck equation for the first-passage time
distribution and fits `(f, k)`, bounded to `[0, 3]`, by differential evolution. Those two
are the whole drift. What the last set did to the person is already inside `value`, so it
needs no parameters of its own.

- **Non-decision time.** Where real people are fitted (the controller's estimate and the
  interface's fit), `T0` is fitted too, bounded to [0, 0.5] s. The recovery pipelines pin it
  at 0 ("Recovery demonstrates identifiability, not accuracy"). Since 2026-10-29 the app
  fits in closed form instead of with pyddm, with λ free for prompts under a countdown
  ("The app's controller").
- **Undecided trials are data.** A decision that reached no bound by `T_DUR` is undecided.
  That includes a *person* who took longer than 7 s to click, since pyddm refuses outright
  to fit an RT past the model's deadline. Undecided trials go back into the fit as counts
  instead of being dropped ("Censored trials are data").
- **Answer and skip times** have a closed-form likelihood (`persistence.neg_log_likelihood`,
  per second, with lapses). The controller uses it only for patience
  (`persistence.fit_patience`, with `f` and `k` held at the decisions' estimate). Passing the
  times to `fit` as `attempts` adds them to pyddm's likelihood through a custom loss, which
  fits decisions and persistence as one set of parameters. That joint fit is kept for
  comparison and is not used for planning.
- **γ (since 2026-10-28).** γ is not in the drift; it shapes `value` through learning. To fit
  it, the person's reward history is replayed through TD learners at candidate values of γ
  (at the fit's α, 0.05), which gives the value every recorded decision would have had
  under each. The γ whose values best explain the decisions is the estimate: a profile
  likelihood, with `(f, k)` fitted at each candidate. The app records everything the replay
  needs. In simulation the replay runs alongside the person, as shadow learners
  (`experiments/gamma_profile.py`, `GridShadow`).
- **A closed-form likelihood (since 2026-10-28, `temporal/fastlik.py`).** With a constant
  drift, the accumulator's first-passage density has a standard series (Navarro & Fuss,
  2009), and so does the chance of reaching neither bound by `T_DUR`. `fastlik` adds
  pyddm's `T0` shift and its uniform lapse mixture the same way pyddm does, and matches
  pyddm's solution (the equations and the check are under "The closed-form likelihood").
  One evaluation takes about 0.2 ms at 700 decisions, against seconds for a pyddm fit, and
  value needs no rounding to 0.05, so the likelihood is smooth in γ.
  `experiments/recovery_lhc.py` uses it to fit `f`, `k`, γ, `BOUND` and `T0` together, with
  standard errors from the observed and the Fisher information.

How well the fit recovers the parameters is under "Current results".

**What the app fits today.** The controller's estimate and the interface's "fit my
parameters" fit `(f, k, T0)` with pyddm, with γ held at 0.9 and `BOUND` at 1.5, and
`InfoController` plans with those values. Fitting γ and the bound in the app is not built
yet ("Next steps").

### Why the model is built this way

Two design decisions changed every result, and they explain parts of the model that would
otherwise look arbitrary.

#### Feedback is timing, not reward (since 2026-09-21)

Until 2026-09-21 the drift had three more terms,
`b_seen · seen_miss + b_unseen · unseen_miss + b_fb · feedback`. They patched a hole in the
learner. Every slot but the last paid zero reward, so outcomes in the middle of a set
produced no prediction error: a miss the person was shown moved `δ` by nothing, and
feedback never entered the TD system. The terms read, off to the side, what the value
system should have been computing.

They were replaced by the mechanism they stood in for: a slot's worth lands when the person
is shown it (under "The learner"). A seen miss is then a slot that paid nothing against a
value that expected something, a negative `δ`, and an unseen miss produces none. That error
reaches the next decision through `value`, at full weight, scaled by the same `f` as
everything else. Three fitted parameters went and none came in.

- **Whether to show feedback is derived, not fitted.** Showing an outcome delivers its
  `(1 − p)` early, so it is discounted less, and it delivers the misses instead of burying
  them in the end-of-set lump. Its effect therefore depends on difficulty. What that does
  is measured under "Which difficulty and whether to show feedback".
- **Boredom is a state the model can express**: sustained `|δ| ≈ 0`. A trained person on an
  easy question has `(1 − p) → 0`, so being shown they were right confirms an expectation
  and pays nothing.
- **What it gives up** is a person *spurred on* by a visible miss. `b_seen > 0` was the only
  way to express that sign; under pure RPE a seen miss is negative for everyone, in
  proportion to their `f` ("Known limitations").

The reaction terms' recovery, and the controller results that depended on them, are in
`RETIRED.md` (V1, V2).

#### Value bootstrap: continuing since 2026-09-13

Is a set its own episode, or is the task a cycle? The start decision re-enters the task, so
it is a cycle. A set's last slot bootstraps from the start of the next set,
`δ = r + γ·V(0) − V(last)`, which makes `V(0)` the fixed point of a recurrence over all
future sets (`value.CONTINUING = True`). The old terminal bootstrap, `V_next = 0`, is still
available for comparison. `value_at_choice_point` is still `V(0)`, the state being entered.

- **Why it matters for measurement.** `f` is identified from variation in value, and under
  the continuing bootstrap value is larger (mean 0.58 against 0.20) and varies about 1.6
  times as much (sd 0.31 against 0.20). Measured under V4 on 2026-10-25
  (`python -m experiments.value_checks`): f = k = 0.8, default setting, 8 seeds × 400 start
  decisions, the same runs with `value.CONTINUING` switched. `RETIRED.md` has the V3
  measurement and the V2 ones, which put the spread at about twice on the single-`diff`
  recovery task (sd 0.222 against 0.115), and whether the switch moved the reward
  controller closer to the best fixed setting (it did not).
- **Saved sessions.** Value weights saved under the terminal bootstrap adapt from the next
  set on.

---

## The controller (`controller.py`)

This is what the project is for: using the estimated parameters to adjust the task so the
person keeps engaging and keeps being challenged. There are two controllers. `Controller`
plans for reward. `InfoController` plans to measure `f` and `k`, and every simulated
controller result here is one of these two. The app runs `AppController`, which is
`InfoController` over every setting the app has ("The app's controller"), since
2026-10-29; until then it ran `InfoController`.

### What both controllers do

Every 5 start decisions in the app (every 25 in the simulations under "Current results"),
the controller plans:

1. **Re-fit.** If at least 15 new decided rows have come in since its last fit, it re-fits
   the person's `f` and `k` (shrunk toward a prior while data is thin), and their patience
   from time on questions.
2. **Copy the person** exactly as they are: measured ability, how hard each question type
   is for them, what they still owe, their current value.
3. **Simulate 96 candidate settings forward** with the *estimated* parameters. A setting
   combines
   - a **target chance of a right answer**, 90% down to 20%;
   - a **review rate**: how an unanswered question comes back;
   - **feedback shown or hidden**;
   - **whether skipped questions come back** through review, or are dropped (with review
     at 0 nothing comes back);
   - **redo prompts**: after each miss the person sees, ask whether to take it again (the
     app's "offer repeat-or-continue"). These 12 settings (6 targets × feedback) have
     review 0, so a miss comes back only when the person chooses it. The other 84 settings
     schedule review.

Each rollout follows its setting's review mode and includes giving up on questions, driven
by the same estimated `f` and `k` and the person's estimated patience, so a hard target is
worth less for someone who would skip most of it. Rollouts count ticks, not decisions, so
the time a person would spend away after stopping earns nothing in them.

The app plans in a separate process, so play continues meanwhile. One `InfoController`
plan takes about 0.9 s ("Code layout, tests and speed"), and one `AppController` plan
over 1104 settings 6 to 7 s ("The app's controller").

Two settings are not levers. `reentry_cost` is left out because cheaper re-entry always
wins, and since V4 it enters no recorded decision: it only sets how long a person stays
away, so it says nothing about `f` or `k` either. (Under V1, setting it from the person's
state coupled the recorded cost to value and collapsed the `k` estimate to zero.) Showing
the stage count is left out because nothing in the model represents it. The score is a
lever of `AppController` only: what it shows is what a set pays ("Playing").

### Planning for reward (`Controller`)

It screens all candidates with short rollouts (4 × 80 ticks), then re-tests the best four
and the current setting with long rollouts on fresh seeds (16 × 300). It switches to the
setting earning the most **total reward** only if that setting beats the current one in
the re-test by more than twice its standard error. Short rollouts alone were too noisy to
act on (the V1 case that showed it is in `RETIRED.md`).

About 15% of updates pick a setting at random instead. These probes do not depend on the
person. When the best setting keeps feedback on, they are the only source of sets with it
hidden, so they are what gets both ways of timing a set's reward into the same record.

**Total reward** is the right answers, each weighted by how hard it was for the person
(`1 − p`), summed over the sets they go on to play. It is engagement weighted by challenge,
not engagement itself. Under V1, sets played was rejected as the objective because it
always favoured the easiest setting (`RETIRED.md`). That reason no longer holds: under the
current model the easiest target is among the worst for engagement ("Which difficulty and
whether to show feedback"). Whether the reward controller should maximize engagement
instead is open ("Next steps").

### Planning to measure (`InfoController`, since 2026-10-11)

A controller that plans for reward settles on one setting. Once it does, cost stops
varying, and `k` stops being identified. `InfoController` instead picks the setting whose
rollouts would most sharpen the `(f, k)` estimate.

Drift is `f · value − k · cost`, so a decision depends on `(f, k)` only through its drift
`μ`, and its Fisher information about `(f, k)` is `I_μ(μ) · [[v², −vc], [−vc, c²]]`. `I_μ`
covers the choice, the RT, and being undecided past the deadline, and is computed once from
pyddm solutions on a grid of `μ`. A setting's score is the rise in
`log det(prior + data so far + its rollouts)`, which favours settings that move value and
cost in directions the data has not covered yet.

- **It plans for every person the data still allows, not one.** The score is averaged over
  6 `(f, k)` draws: from the prior (`PRIOR`, sd 0.5) until the fit is used, then from a
  Laplace approximation, `N(estimate, (prior + data information)⁻¹)`. Each draw is rolled
  out with its own parameters (25 ticks per rollout). Planning for the single current
  estimate collected the best data for the wrong person early on.
- **Safety floor.** A setting is eligible only if its rollouts start at least half as many
  sets as the best setting's, so measuring cannot drive the person away. This is its only
  rule about engagement.
- **No random probes.** It varies the setting because that is what it plans for.

### The app's controller (`AppController`, since 2026-10-29)

*`python -m experiments.app_checks` · results 2026-10-29 in `results/app_checks/`*

`AppController` (`controller.py`) is `InfoController` with every setting of the app whose
effect the model describes, and with redo prompts under a countdown read as such.

- **1104 settings** (`APP_CANDIDATES`): 6 targets × 4 set lengths (2, 4, 6, 8 questions)
  × score shown or hidden × 23. The 23 are `InfoController`'s 14 scheduled-review settings
  at that target; redo prompts mid-set with feedback shown, or after the score with it
  hidden (each redo's outcome then shown, as the app does), each untimed or under a 2, 3
  or 5 s countdown (8); and "repeats", every seen
  miss redone with no prompt (1). The stage count is not a setting: nothing in the model
  represents it, and it stays open until the literature on it is read.
- **The fit** is the closed-form likelihood (`inference.fit_closed_form`, `fastlik`) with
  `f`, `k`, `T0` and λ free, γ at 0.9 and `BOUND` at 1.5. A prompt under a countdown is
  read at bound λ · `BOUND` and its own deadline, one that ran out as censored there. λ
  has a prior N(0.7, 0.2) (`LAM_PRIOR`, as `experiments/controller_all.py`), so with no
  countdown in the record it stays at 0.7. `f` and `k` are shrunk toward the prior as
  `InfoController`'s are. The app's "fit my parameters" uses the same fit.
- **The score** is the rise in log det of the information about `f` and `k` with λ a
  nuisance (the Schur complement of the `(f, k, λ)` information; Ds-optimal design). With
  no decision timed it is `InfoController`'s score exactly (`tests/test_app_controller.py`).
  A timed decision informs its drift and its bound λ · `BOUND` at its own deadline, from
  the closed form. `(f, k, λ)` are drawn as `InfoController` draws `(f, k)`; a drawn λ is
  snapped to 0.05 so the countdown's draw tables are reused.
- **Two stages, same luck.** Every setting is rolled out on the first 2 of the 6 draws
  (25 ticks each), and the 24 best above the safety floor are scored on all 6. Every
  setting is rolled out on the same seeds, so settings are compared on the same luck. The
  safety floor is `InfoController`'s, at both stages.

Checks, without claims (conditions in the script's docstring; γ 0.9, `BOUND` 1.5,
`ACHIEVEMENT_SCALE` 4, continuing bootstrap, `T_DUR` 7.0 s):

- **The closed-form fit is pyddm's fit, within pyddm's discretization.** On 8 simulated
  records with no countdown (4 people, 2 seeds, 150 start decisions plus redos, `T0`
  0.2 s), the two estimates differ by at most 2.9% in `f`, 1.4% in `k` and 0.035 s in
  `T0`. Under the closed form, pyddm's estimate is worse than its own by a median 0.11 in
  −log L (largest 0.63). It fits in about 0.01 s, against about 14 s for pyddm's.
- **λ is recovered.** With a 3 s countdown on every redo prompt (f 0.9, k 0.7, λ 0.4 to
  1.0, 2 seeds each, about 200 timed prompts of about 350 decisions), λ's RMSE was 0.018
  and its bias +0.010; `f`'s RMSE 0.028 and `k`'s 0.116. The fit held γ and `BOUND` at the
  truth, as the app's does.
- **A plan takes 6 to 7 s**: 7.0 s with empty caches, then 6.8 and 6.4 s (Apple M5, after
  80 start decisions). `InfoController` over its 96 settings took 18 s with its pyddm fit,
  11.5 s of it the fit. In the app a fresh planning process returned the first plan in
  4.9 s.

**Not measured.** Nothing shows yet whether `AppController` measures `f` or `k` better
than `InfoController`, or keeps people engaged as well. The nearest test is
`AllInfoController` ("Every parameter, every setting": six fitted parameters, 132 settings
without set length or score, scored on all six). Against `InfoController` it kept people
engaged and measured λ, and measured `k` no better. Three of its levers are new to the model since 2026-10-29 and have no
recovery result of their own: set length, the score display (hidden, the reward is the
fraction right) and "repeats". Nor do the 2 and 5 s countdowns: `redo_pressure` measured λ
under 3 s.

---

## Human sessions (`temporal/app/`)

### Playing

You answer the questions, choose whether to redo one you got wrong, and choose whether to
start the next set. The start prompt waits for you, and the time to click is the reaction
time. The redo prompt waits too unless a countdown is set (below).

- **The score is the TD reward.** The header shows "score 2.35": every reward that has
  landed on the value learner since the app was opened, so it carries from set to set
  until you quit. A right answer adds `4 · (1 − p) / n` (`ACHIEVEMENT_SCALE`, your chance
  `p` on it, `n` questions per set) when you are shown it. With feedback hidden nothing
  lands until the end of the set, so the score in the header does not move during the set
  and gives no miss away. With "show score" off the header has no score, and what a set
  pays is what you can still see: the fraction right, `1 / n` per right answer (the
  simulator's `show_score=False`, its accuracy reward). That is a model change, decided on
  2026-10-29: what is shown is the reward.

- **"I'm done" closes the app** and is logged as the stop decision. Closing the window
  without choosing is logged as an abandon.
- **A redo is an extra question**, not the next slot, so a set can run past `stage_amt`
  questions and its review load can exceed 1. (In the simulator's `choice` mode a redo
  takes the next slot.)
- **A set played with feedback hidden ends on a summary**, which stays until you continue:
  each question with your answer and whether it was right ("question 1: 3 + 8 = 11
  correct +0.07", "question 2: 4 × 6 = 7 incorrect", or "skipped"), then the set's points
  and the score ("+0.27 this set, score 1.33"). The right answer to a miss is not shown.
  With the score hidden there are no points, and the total is how many were right ("3 of
  4 right"). The summary is when the model pays the set's reward, and a question's points
  are its share of it, so they add up to the set's. Tick "with feedback off, offer it
  after the score" as well as the redo prompts, and each new question missed in the set is
  then offered again, one prompt at a time ("redo question 2: 4 × 6?"). This is the
  simulator's `end_redos`: the same redo decision at the set's last slot, carrying the
  set's error. A redo taken there is an extra question that adds to the next start
  decision's review load, and does not count in the set's score. Its outcome is shown at
  once ("correct +0.61" or "wrong"), so what it earns lands then, as more reward on the
  set's last step (`value.late_reward`, the simulator's `end_feedback`), and the next
  prompt and the next start decision carry it. A quick right answer to it is never read
  as a typo, because other answers have moved the ratings since the miss.
- **Skip** (the button under the answer box, or Esc) gives up on the question on screen.
  It is logged with how long you worked on it and your predicted chance on it, and the set
  moves on to the next slot. Whether skipped questions come back is the "skipped questions
  come back too" checkbox, or the controller's call.
- **Leaving is recorded with what was on screen.** Closing the window, Ctrl-C in the
  terminal, closing the terminal, or `kill` all log an `abandon` with the screen you were
  on. On a question it also records the question, your chance on it, how long you had it,
  and whether it was the first question since opening the app. Looking at a question and
  leaving is giving up on it, so it joins the persistence data. An exit nothing sees (a
  crash, power loss) is found on the next launch and logged as `unlogged`, with the
  question that was on screen and no time.
- **A countdown on the redo prompt** ("countdown on redo prompts": off, 2, 3 or 5 s).
  The prompt shows the seconds left and closes when they run out, and you move on without
  the redo. That is `model/pressure.py`: under time pressure a person keeps only λ of their
  bound, λ being their own. A prompt that ran out is recorded with `GO` and `RT` empty and
  its `deadline`; a click that arrives after the deadline counts as running out. Every fit
  the app makes reads such a prompt at bound λ · `BOUND` and its own deadline ("The app's
  controller").
- **"Repeats"**, with redo prompts off, redoes every miss you are shown with no prompt
  (the simulator's `forced` review mode).
- **The controller.** Tick **"controller sets every setting but the stage count"** and
  `AppController` ("The app's controller") plans in a separate process every 5 start
  decisions and moves the settings panel itself, with a line saying why: target, review
  rate, feedback, skipped questions, redo prompts mid-set or after the score, the
  countdown, "repeats", questions per set and the score display. With it off, they are
  yours. Feedback, set length and the score display change from the next set. Showing the
  stage count stays yours: nothing in the model represents it yet.

### What is saved

Everything persists in `data/` (gitignored):

- `decisions.jsonl`: the record, append-only and synced to disk before the app moves on.
  It holds every start and redo decision, every answer and skip (question, type, your
  answer, your predicted chance, time on it), exits with what was on screen, retired
  questions, set starts and ends, end-of-set scores (`score`, with the set's points and
  the session's score), controller decisions and launches. A set start records the
  settings that shape the model's account of it: set length, score display, redo prompts,
  end-of-set redos, countdown and "repeats". A redo decision made after the score has `end`
  and the question's text, one under a countdown has `deadline`, and an answer or skip
  after the score has `slot` equal to the set's length.
- `state.json`: the snapshot to pick up from: value weights, ratings, owed and skipped
  questions with their exact text, how often each came back and failed, settings, and what
  is on screen right now. It is written atomically, so a crash never leaves half a file.
- `controller.pkl`: the controller, including its estimates.

Reopening the app logs a `return` with the time since you left, and every fit and plan uses
every session. "Export all decisions to csv" writes `data/decisions.csv`.

**Old saves are brought up to the current rules.** The first-showing, typo and retirement
rules apply to the whole record, not only to new answers. A state saved under older rules
gets its ratings and failed-return counts rebuilt from `decisions.jsonl` on the next
launch, logged as `rebuilt`, and questions that have already failed 3 returns are retired.
The record itself is never rewritten. Replaying a record written under the current rules
reproduces the live ratings exactly, so the rebuild is the same rule applied after the
fact, not an approximation.

One change cannot be replayed. `RULES` went to 3 on 2026-09-21, when the reaction terms
were removed. A start decision logged under rules 1 and 2 holds a `value` from before the
prediction error was carried. That `value` is right for the drift it was drawn under but
not comparable with a later row. The rows are kept and still fit, since the drift form
`f · value − k · cost` is unchanged, but they under-represent how much value varies.

### What human data gives the fit

- **Every stop is a session end.** Each session contributes at most one "no" at the start
  prompt, and those are what identify `k` there, so many sessions are needed. People also
  stop by closing the app without choosing, which is why leaving is recorded with what was
  on screen.
- **The fit never sees a cold start from a human.** Opening the app is the cold decision,
  and its "reaction time" is hours. Every start decision the fit sees is made mid-session,
  at warm cost; since V4 the simulator records the same ("Sessions, and what is
  recorded"). For humans, most of what identifies `k` comes from the redo prompts, whose
  cost does vary with how hard the missed question is. Time spent on questions sets
  patience only; it does not enter the `f` and `k` fit ("Skips do not help estimate `f`
  and `k`").
- **Every question is a persistence observation.** A set gives one start decision but four
  or more questions, each with a time and whether it was skipped. Earlier sessions' answers
  are used too: they were logged with a time and a predicted chance, so they enter as
  questions nobody gave up on.

### On a website (`temporal/app/web.py`)

`python -m temporal.app.web` serves the same task in a browser at `http://127.0.0.1:8000`.
The browser only draws screens and times them. Every rule is `session.Session`'s, run on
the server by a subclass that sends its screens to the page instead of Tk, so nothing about
the task, the record or the inference is re-implemented.

**Data collection is off until it is approved.** Research with other people needs approval
first, so by default (`TEMPORAL_COLLECT=0`) nobody else's data is kept:

- The start screen shows the consent information (a **draft**, in
  `temporal/app/static/index.html`, to be replaced with the approved wording) and offers two
  ways in: **as a guest** or **with an email**.
- **Guests are never saved.** Their record lives in a temporary folder that is deleted when
  they leave the page.
- **Email** is accepted only for the addresses in `TEMPORAL_RESEARCHERS` (comma-separated):
  the researchers, playing themselves. Their records are kept in `$TEMPORAL_DATA/<email>/`
  (default `data/web/<email>/`), with the same `decisions.jsonl` / `state.json` /
  `controller.pkl` as `data/`, and they see "dev stuff".
- With `TEMPORAL_COLLECT=1`, anyone may take part with an email after ticking the consent
  box. The first event of each session is a `consent` record with the email, whether they
  agreed, and `CONSENT_VERSION` (in `temporal/app/web.py`; bump it whenever the consent text
  changes).

The email is not verified: anyone who types an address plays as that person. That is
enough to keep one person's sessions together, but it is not a login.

How the web version behaves:

- **Reaction times are measured in the browser**, from the moment a screen is painted to
  the key press or click, so network latency never enters an RT. The 600 ms "correct" /
  "wrong" flash plays in the browser before the next screen, and its clock starts after the
  flash.
- **Starting from the start screen is launching the app**, so each start is a new session.
  Closing the tab, reloading or navigating away is closing the window: it is logged as an
  `abandon` with what was on screen and the browser's time on it. Starting again elsewhere
  (another tab or device, same email) closes the first one the same way. A page left silent
  for 2 hours is closed as well. A server that dies without closing is caught on the next
  launch, as `unlogged`, the same as a crash on the desktop.
- **Plans and fits run in one process pool** shared by all players.
- **Dev stuff** (settings, behind the scenes, fit, export, and the switch for the live
  accumulator) is shown to researchers only, or to everyone with `TEMPORAL_PANEL=1`.
- **The live accumulator** (`app/live.py`) shows what the model predicts for the start or
  redo decision on screen. On the web it sits beside the task once its box is ticked in dev
  stuff; on the desktop it opens from the "live model view" button. It draws the bounds,
  the mean drift `f·value − k·cost`, the predicted RT densities at each bound, and sample
  paths that run in real time on the person's clock, then marks their actual choice and RT.
  It uses the controller's estimate, or the prior `(0.8, 0.8)` until there is one. Its paths
  come from their own random generator rather than the global ones, so seeded sessions are
  unchanged, and nothing it shows is recorded.

Run it as **one worker process**: live sessions are held in memory.

**Hosting on Fly.io** (`fly.toml` is in the repo):

```bash
brew install flyctl && fly auth login
fly launch --copy-config --no-deploy        # pick an app name and region
fly volumes create temporal_data --size 1   # where records live; without it, a redeploy wipes them
fly secrets set TEMPORAL_RESEARCHERS=you@example.com
fly deploy
fly scale count 1                           # exactly one machine: sessions are in memory
fly ssh sftp get /data -r ./data-from-fly   # download the records
```

Anywhere else, build the `Dockerfile`, mount a persistent volume at `/data`, and set the
same environment variables. It listens on `$PORT` (default 8000).

---

## Current results

Every number here was measured under the current model (V4 in `RETIRED.md`): the drift
`f · value − k · cost` with prediction error carried in `value`, reward that lands when
shown, and a simulated person who records only what the app could. Older measurements, and
the ones these replace, are in `RETIRED.md` by model version.

### Reporting conventions

`f` is identified only up to `ACHIEVEMENT_SCALE` and the value bootstrap, and up to the
discount factor γ unless γ is fitted ("Fitting the discount factor"). Every result was
measured with `ACHIEVEMENT_SCALE = 4`, the continuing bootstrap and `T_DUR = 7.0`. Numbers
are not comparable across model versions.

Unless a result says otherwise:

- γ is 0.9 and `BOUND` 1.5, for the simulated person and for the fit;
- simulated people have no non-decision time (`T0` 0), and the recovery pipelines pin it at
  0;
- "the grid" is `f, k ∈ {0.2, 0.6, 1.0, 1.5}` (4×4). The five-parameter runs draw people
  from a Latin hypercube instead;
- "the app's task" is the question-type task with Elo ratings and skips on, steered by
  `InfoController` from the default setting, as the app ran it until 2026-10-29.

**Run length.** Recovery runs stop at a number of **recorded decisions**, since they ask how
much data identifies `f` and `k`. Runs about engagement, reward or a controller last a
number of **ticks**, so a person who stops more spends more ticks away and gives fewer
decisions ("Sessions, and what is recorded"). Until V4 the two were the same thing.

**Claims.** Most results state claims fixed before the run and say whether each held. The
decision rule, fixed on 2026-09-13 for controller comparisons, governs every claim here:

- A claim that one way of choosing settings beats another for a person (the controller
  against the fixed default, or against random switching) needs the mean over seeds of the
  paired difference to exceed twice its standard error. Paired means the same seed: the same
  simulated person and question difficulties. The controller applies this margin to itself
  before switching.
- A difference inside that margin is reported as undetermined, in either direction.
- Every cell reports the mean ± standard error over seeds.
- The best fixed setting is chosen on 8 seeds and scored on 8 others, so picking and scoring
  on the same runs does not flatter it.

### Recovering f and k

Does the fit get back the `f` and `k` a simulated person was given?

#### Recovery from one person

*2026-10-25 · `python -m experiments.recovery_sweep`*

One simulated person on the single-`diff` task (mid cold-start cost, scheduled review). Each
fit uses 1200 recorded decisions, split across four learners that start after 0, 200, 500
and 1000 warm-up decisions; the grid × 3 reps gives 48 fits, with 95% bootstrap intervals
over the fits. Pass condition, fixed on 2026-09-13: `f` and `k` RMSE ≤ 0.05, the step the
controller rounds its estimates to (`snap_value`), so an estimate about one step off plans
like the truth or its neighbour.

| f RMSE | k RMSE |
|---|---|
| **0.049** [0.040, 0.058] | **0.095** [0.078, 0.111] |

- **The pass condition is not met.** `k` misses it clearly. `f`'s point estimate is under
  0.05 but its upper bound is not. `k` is measured worst because a person's start decisions
  are all warm, so `k` comes mostly from cost's constant offset on the drift ("How value and
  cost together decide what can be measured").
- **The worst grid points**: `k` RMSE 0.152 at f 0.6, k 1.0, 0.133 at f 1.0, k 0.6 and 0.130
  at f 0.6, k 0.2; `f` RMSE 0.077 at f 1.0, k 0.2.

#### The recovery pipelines

*2026-10-25, seeded per grid point ·
`python -m temporal recovery --mode MODE --reps R --out results/pipelines`*

The grid, with three ways of generating the data:

| pipeline | f: r | f: RMSE | k: r | k: RMSE |
|---|---|---|---|---|
| **single**, one continuous learner, 1200 recorded decisions (3 reps) | 0.994 | 0.052 | 0.984 | 0.090 |
| **coupled**, two stacked learners at different cold-start costs, 700 recorded decisions each (3 reps) | 0.998 | 0.034 | 0.988 | 0.077 |
| **uncoupled**, value and cost set by design, 400 per condition (5 reps) | 0.999 | 0.026 | 0.999 | 0.023 |

![Coupled recovery](assets/initiation_recovery_coupled.png)

- **`single` measures `k` worst.** Its start decisions are all warm, so cost moves only with
  the last set's review load (sd about 0.01). Its worst grid points for `k` are 0.154 (f 1.5,
  k 0.6), 0.151 (f 0.2, k 1.5) and 0.130 (f 0.6, k 1.0); for `f`, 0.091 at f 1.5, k 0.6.
- **`coupled` has lost the contrast it was built for.** Its two learners differ in
  cold-start cost, which since V4 only sets how long each stays away. It is now two learners
  pooled, a little better than `single` because it has 1400 decisions against 1200.
- **`uncoupled` does not run the simulator**, so the recording flaw never touched it.
  Re-run under V4, it reproduced the V3 numbers exactly.
- **Which to use**: `single` (or the four-learner design above) for one real participant,
  and `uncoupled` when the experimenter controls value and cost. A single learner's value
  and cost are correlated and confined to what it happens to experience, so it recovers less
  well than a designed contrast.

#### Prediction error is a large part of the variation in value

*2026-10-25 · `python -m experiments.value_variation`*

On the question-type task the target keeps `p` away from 1, so `δ` stays alive: the carried
error's standard deviation is 50% of value's (44% to 57%). Eight fits of 700 decisions (the
four people of the persistence test × 2 seeds, settings switched by random probes) give `f`
RMSE 0.033, `k` RMSE 0.062, value sd 0.38 to 0.44 and `corr(value, cost)` +0.35 to +0.64. On
the single-`diff` task a learner masters the fixed difficulty, so `p → 1` and `δ` shrinks.

#### Skips do not help estimate f and k

*2026-10-25 · `python -m temporal recovery --mode persistence --reps 3`*

Twelve simulated people (4 profiles × 3 reps) who can give up on questions, 300 start
decisions each (about what a person gives), settings switched at random:

| fitted from | f RMSE | k RMSE |
|---|---|---|
| decisions only | **0.047** | **0.219** |
| answer and skip times only | 0.648 | 0.431 |
| both, jointly | 0.052 | 0.187 |

- **Adding answer and skip times changes neither parameter detectably.** Paired over the
  twelve, squared error with both minus decisions only is +0.0005 ± 0.0008 for `f` and
  −0.0130 ± 0.0194 for `k`. The fit factorial below finds the same on 32 people.
- **Times alone cannot place `f`.** A question's value, `p(1 − p)`, hardly varies across
  questions, while its cost, `1/p`, varies about 4×, so skip times mostly measure `k`, which
  the redo prompts already carry.
- So `f` and `k` are fitted from decisions, and time on questions sets only patience: times
  would add a likelihood and its assumptions for no detectable gain. (Under V3 they made
  both worse.)

#### What each part of the fit contributes, alone and together

*2026-10-25 · `python -m experiments.fit_factorial`*

The 32 people of the controller comparison (grid × 2 seeds), the app's task, 300 start
decisions, true non-decision time 0.3 s, true patience 10 s. Each dataset is fitted 8 ways:
timed-out trials kept or dropped, `T0` fitted or pinned at 0, and answer and skip times
added or not (at a fixed 10 s patience, their best case). Claims fixed before the run: a
choice matters alone if switching only it worsens squared error beyond 2 SE; two choices
interact if the bias from switching both differs from the sum of switching each, beyond 2
SE.

| timeouts | T0 | times | f RMSE | k RMSE | f bias | k bias |
|---|---|---|---|---|---|---|
| **kept** | **fitted** | **no** | **0.042** | **0.047** | −0.2% | −0.6% |
| dropped | fitted | no | 0.051 | 0.155 | +8.6% | +15.0% |
| kept | pinned | no | 0.275 | 0.144 | −20.5% | −15.8% |
| kept | fitted | yes | 0.035 | 0.083 | −1.3% | −3.9% |
| dropped | pinned | no | 0.267 | 0.115 | −13.5% | −3.2% |
| dropped | fitted | yes | 0.045 | 0.101 | +5.9% | +3.4% |
| kept | pinned | yes | 0.270 | 0.125 | −20.0% | −14.2% |
| dropped | pinned | yes | 0.264 | 0.098 | −14.0% | −7.2% |

- **Pinning `T0` costs the most**: about six times the `f` error (0.275 against 0.042), with
  `f` 13 to 21% low whatever else is done. It matters alone beyond 2 SE for both parameters.
  Dropping timed-out trials matters alone for `k` (beyond 2 SE), not for `f`. Adding times
  matters for neither alone (`f` −0.0006 ± 0.0004, `k` +0.0047 ± 0.0041 in squared error).
- **Every pair interacts beyond 2 SE**, by 1.5 to 8.3 percentage points of bias, mostly by
  cancelling: dropping timeouts pushes the estimates up, while pinning `T0` and adding times
  push them down. Dropping timeouts and adding times together leaves little mean bias
  (+5.9%, +3.4%), yet `k`'s error is 0.101 against 0.047. Report error, not only bias.
- **The pipeline holds with a real non-decision time**: `T0` recovered at 0.31 ± 0.03 s;
  8.9% of decisions timed out; 173 skips per person on average.

#### Patience can be recovered

*2026-10-25 · `python -m temporal recovery --mode patience --reps 3`*

Twelve simulated people (4 profiles × 3 reps), 300 start decisions each, settings switched
at random. `f` and `k` were fitted from decisions, then patience from time on questions,
once with the fitted `f` and `k` and once with the true ones:

| true patience | fitted with fitted `f`, `k` (3 reps) | give-ups per person |
|---|---|---|
| 2 s | 1.4, 1.7, 2.1 s | 289 to 784 |
| 5 s | 4.6, 4.9, 5.7 s | 40 to 64 |
| 10 s | 10.1, 11.4, 13.7 s | 17 to 55 |
| 20 s | 21.7, 29.6, 30.0 s | 2 to 58 |

- **Errors in `f` and `k` add little**: patience was off by a factor of 1.26 with the fitted
  `f` and `k`, against 1.21 with the true ones (root mean square of the log error).
- Someone who gives up at a glance (2 s) was never mistaken for a patient person (10 s). The
  most patient person was fitted too patient in two of three reps (about 30 s).
- With no give-ups patience is not identified, so the controller keeps the 10 s default
  until there is one.

#### The redo node matters on the question-type task

Its cost varies with how hard the missed question is, and a controller that seeks out redo
prompts brought `k` RMSE from 0.508 to 0.156 over 300 ticks ("Measuring costs no detectable
reward"). On the single-`diff` task redo cost barely varies, and stacking redo draws did
little (V1 measurement in `RETIRED.md`).

### Fitting the discount factor (2026-10-28)

The results above hold γ at 0.9, as the app's fit does. If a person's γ differs, the wrong
γ mostly rescales value and `f` absorbs it: on the app's task `f` comes out 62% too low at
γ 0.7 and 57% too high at γ 0.95 (table below). Fitting γ removes that bias.

*`python -m experiments.gamma_profile`*

- **People and task**: (f, k) = (0.8, 0.8), (1.2, 0.4), (0.6, 1.2) × true γ 0.7, 0.8, 0.9,
  0.95 × 5 seeds, 60 people, each on the app's task for 300 start decisions plus their redo
  decisions (about 420 per person).

<details>
<summary>How γ is fitted, and the statistics</summary>

- **How γ is fitted**: the person learns at their own γ. Shadow learners at each γ of a grid
  (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99) replay the same rewards, as the app could
  replay a person's record, and give each decision's value under that γ. `(f, k)` is fitted
  at every grid γ, and the γ with the lowest −log L is the estimate. α 0.05, `BOUND` 1.5,
  `SIGMA` 0.7 and `T0` 0 are shared by person and fit, so γ is the only constant that
  differs.
- **Statistics**: the grid γ values within 1.92 of the lowest −log L form a 95%
  likelihood-ratio interval (Wilks: twice the drop in log-likelihood is χ² with 1 df, whose
  95% point is 3.84). Each grid point's fit is a pyddm fit, so value is rounded to 0.05, as
  in every pyddm fit here. Bias is (estimate − true) / true, mean ± SE over a cell's 15 runs
  or all 60.

</details>

Claims fixed before the run:

- **P1**: the estimate is the true γ or its grid neighbour in at least 80% of runs.
- **P2**: every grid γ more than 0.1 from the estimate costs at least 1.92 in −log L (a 95%
  likelihood-ratio interval inside ±0.1) in at least 80% of runs. As first written, P2 had
  no run-level threshold; the 80% was added on 2026-10-28, after the run had started and
  before any result was read.

| true γ | estimate = truth | `f` bias, γ fitted | `k` bias, γ fitted | `f` bias, γ held at 0.9 | `k` bias, γ held at 0.9 |
|---|---|---|---|---|---|
| 0.7 | 15 of 15 | +0.7 ± 1.1% | +1.4 ± 1.4% | −62 ± 1% | +17 ± 2% |
| 0.8 | 15 of 15 | +0.4 ± 1.4% | −2.0 ± 1.6% | −46 ± 1% | +4 ± 2% |
| 0.9 | 15 of 15 | +0.7 ± 1.0% | +0.7 ± 1.1% | +1 ± 1% | +1 ± 1% |
| 0.95 | 15 of 15 | −0.4 ± 0.8% | −2.2 ± 1.3% | +57 ± 1% | −8 ± 4% |

- **P1 and P2 hold in all 60 runs.** The profile is steep: one grid step from the truth costs
  8 to 230 in −log L in most runs, and the smallest rise anywhere was 0.4 (γ 0.6 for a
  person at 0.7, inside ±0.1).
- **With γ fitted, `f` and `k` are unbiased**: +0.3 ± 0.5% and −0.5 ± 0.7% pooled, and no
  person is off by more than 12%.
- **What it does not show.** Every true γ was a grid point, so "the estimate is the truth"
  means correct to the grid's resolution (0.05 to 0.1), not a measured precision. Fitted
  continuously, with the bound and `T0` free too, γ is measured to about ±0.03 ("The bound
  can be fitted"). Every person shared α, `BOUND`, `SIGMA` and the reward with the fit, the
  best case, which the next study relaxes. The simulated person is paid the end-of-set
  reward with feedback hidden, as the app's score has shown it since 2026-10-28. Only three
  combinations of `f` and `k` were run.

#### When other constants are wrong

*`python -m experiments.gamma_robustness`*

The same task, people and grid, but each person differs from the fit in one more constant,
and the fit is not told:

- a person's own `BOUND` applies to their start, redo and give-up draws only, while the
  controller's rollouts and every fit keep 1.5;
- a person who values the fraction right learns from that reward, while the shadow learners
  that give the fit its values are paid `1 − p`, as the app would compute;
- a person's own α drives their learning, while the shadow learners keep 0.05.

True γ 0.8 and 0.95, 2 seeds per person (6 runs per cell), on seeds distinct from the
profile runs (the profile's seed plus 1000 · (1 + level index + 10 · factor index)). The
last column holds γ at 0.9 as well, so there both γ and the other constant are wrong.
Claim fixed before the run, **R1**: the estimate is within one grid step of the truth in at
least 80% of runs at every level.

| person differs in | true γ | estimate | `f` bias at the estimate | `k` bias at the estimate | `f` bias, γ held at 0.9 |
|---|---|---|---|---|---|
| `BOUND` 1.0 | 0.8 | 6 of 6 exact | +58 ± 7% | +57 ± 5% | −11 ± 4% |
| `BOUND` 1.0 | 0.95 | 6 of 6 exact | +52 ± 2% | +47 ± 6% | +145 ± 4% |
| `BOUND` 2.0 | 0.8 | 6 of 6 exact | −28 ± 1% | −27 ± 1% | −61 ± 1% |
| `BOUND` 2.0 | 0.95 | 6 of 6 exact | −26 ± 1% | −26 ± 1% | +17 ± 1% |
| values the fraction right, not `1 − p` | 0.8 | 4 exact, 2 one step off | −52 ± 4% | −22 ± 8% | −73 ± 2% |
| values the fraction right, not `1 − p` | 0.95 | 6 of 6 exact | −54 ± 3% | −22 ± 8% | −25 ± 5% |
| α 0.01 | 0.8 | 5 exact, 1 one step off | −10 ± 7% | −26 ± 7% | −54 ± 1% |
| α 0.01 | 0.95 | 3 exact, 3 one step off | −36 ± 7% | −36 ± 10% | −20 ± 1% |
| α 0.025 | 0.8 | 6 of 6 exact | −5 ± 3% | −3 ± 1% | −49 ± 2% |
| α 0.025 | 0.95 | 6 of 6 exact | −24 ± 1% | −20 ± 6% | +21 ± 1% |
| α 0.1 | 0.8 | 6 of 6 exact | +3 ± 1% | +5 ± 2% | −45 ± 1% |
| α 0.1 | 0.95 | 6 of 6 exact | +16 ± 2% | +13 ± 4% | +85 ± 2% |
| α 0.2 | 0.8 | 6 of 6 exact | +4 ± 1% | +7 ± 1% | −44 ± 1% |
| α 0.2 | 0.95 | 4 exact, 2 one step off | +51 ± 16% | +21 ± 8% | +104 ± 2% |

- **R1 holds at every level** (84 runs: the true γ in 76, one grid step off in 8, never
  further). γ survives a wrong bound, reward or learning rate; `f` and `k` do not.
- A wrong bound scales `f` and `k` by about the same factor, which fitting the bound removes
  ("The bound can be fitted").
- A person who values the fraction right gets `f` about half its true value even with γ
  fitted, and no fitted parameter absorbs that. Choosing the reward by likelihood removes it
  ("Measuring the learning rate and the reward").
- A wrong learning rate costs little at γ 0.8 (`f` within 10%) and a lot at γ 0.95 (`f`
  −36% at α 0.01 to +51% at α 0.2): a patient learner's value depends on many sets, so how
  fast it was learned matters more.

#### Where the information about γ comes from

*2026-10-28 · `python -m experiments.gamma_sources`*

The 12 runs of seed 0 of the profile design (3 people × 4 true γ, the same data), measured
two ways:

1. Profiles from start decisions alone, and from redo decisions alone.
2. The information about γ left once `f` and `k` are profiled out. Since drift is
   `f·v − k·c`, the fit can absorb any change in value along `v` or `c`; only the rest of
   `∂v/∂γ` identifies γ. That rest is the residual `r` of `∂v/∂γ` after a projection onto
   `(v, c)` weighted by `I_μ`, one decision's information about its drift
   (`controller.i_mu_table`), and the profiled information is `J = f² · Σ I_μ · r²`. It is
   computed for `∂v/∂γ` and for each part of value alone (`V(0)`, the carried error, a
   redo's slot share), with `∂v/∂γ` a central difference between the true γ's grid
   neighbours.

Claim fixed before the run, **S1**: the carried error alone carries more of `J` than `V(0)`
alone in at least 80% of runs.

| true γ | share of `J`: `V(0)` | carried error | slot share |
|---|---|---|---|
| 0.7 | 0.92 | 0.07 | 0.01 |
| 0.8 | 0.94 | 0.06 | 0.01 |
| 0.9 | 0.96 | 0.03 | 0.00 |
| 0.95 | 0.97 | 0.03 | 0.00 |

- **S1 does not hold in any run.** γ is identified through `V(0)`, the learned value of
  starting a set, which holds 92 to 97% of the profiled information. A wrong γ does not only
  rescale `V(0)`; it changes how `V(0)` moves over the run, and `f` cannot absorb that. The
  carried error adds 3 to 7%, and a redo's slot share at most 1%.
- In line with that, start decisions alone identify γ (the profile's P1 in 12 of 12 runs, P2
  in 11), and redo decisions alone barely do (P1 10, P2 5).
- As a check of the approximation, `½·J·h²` predicts a rise of 29.8 ± 6.4 in −log L one grid
  step from the truth, against 22.4 ± 3.9 observed: the same order, a third high.

#### Feedback switched in blocks

*2026-10-28 · `python -m experiments.gamma_blocks`*

Hiding feedback moves a set's reward to its end without changing the total, so how much
less a person starts after hidden sets depends on γ, not on `f`. Two arms, paired on person
and seed, no controller, one fixed setting (aim 60% right, no scheduled review, redo
prompts): feedback on for 25 start decisions and off for 25, alternating, against on
throughout. The profile design's people and true γ, 2 seeds each (24 pairs). As in the
profile, the simulated person is paid the end-of-set reward with feedback hidden, as the
app's score has shown it since 2026-10-28. Sharpness is the rise in −log L from the
true γ to the best grid γ more than 0.1 away.

Claims fixed before the run: **B1**, the profile's P1 and P2 hold in at least 80% of the
blocks arm's runs; **B2**, sharpness is higher with blocks than without, paired, beyond 2
SE.

| true γ | sharpness, feedback always on | in blocks |
|---|---|---|
| 0.7 | 11.0 | 16.0 |
| 0.8 | 24.8 | 48.4 |
| 0.9 | 88.5 | 137.4 |
| 0.95 | 146.7 | 182.2 |

- **B1 and B2 hold.** P1 and P2 held in all 24 runs of both arms. Blocks raised sharpness by
  28.3 ± 4.9 (5.8 SE), although those sets had no redo prompts while feedback was off, and
  the blocks arm had 138 redo decisions per run against 247.
- `f` at the estimated γ was unbiased in both arms (blocks +1 ± 2%, always on −1 ± 1%).
- So switching feedback on a schedule is a cheap way to measure γ better. The app shows the
  end-of-set score this relies on since 2026-10-28 ("Playing").

### The bound can be fitted: f, k, γ, BOUND and T0 together

*`python -m experiments.recovery_lhc` · results 2026-10-28; histories in
`results/recovery_lhc/histories/`, fits in `results/recovery_lhc/fits.csv`*

All five parameters fitted at once, by maximum likelihood with the closed-form likelihood,
from the data the app records.

- **People**: 100 from a seeded Latin hypercube over `f` and `k` in 0.2 to 1.5, γ in 0.6 to
  0.97, `BOUND` in 1.0 to 2.0 and `T0` in 0 to 0.4 s
  (`scipy.stats.qmc.LatinHypercube(d=5, seed=20261028)`, scaled to those ranges). Person `i`
  is simulated with seed `20261028 + i`, so every machine draws the same people and
  `--people` only says which to run.
- **Task**: the app's (its `InfoController` plans with γ 0.9 and `BOUND` 1.5, as the app
  does), for 300 start decisions plus their redo decisions. α is 0.05 for person and fit.

<details>
<summary>How the fit and its standard errors were computed</summary>

- **Value at any γ**: each history is saved once with each decision's value under every γ
  from 0.50 to 0.99 in steps of 0.01, and a cubic spline joins those 50 values, so the
  likelihood is smooth in γ. The fit uses value before rounding, while the person's own
  draws used value rounded to 0.05: a small misspecification the fit carries.
- **The fit**: `fastlik` over `f`, `k` in [0, 3], γ in [0.5, 0.99], `BOUND` in [0.5, 3] and
  `T0` in [0, 0.5] s, by differential evolution (seeded by the person's index), then
  L-BFGS-B from its best point. The same data are also fitted the current way: γ 0.9 and
  `BOUND` 1.5 pinned, the same search over `f`, `k` and `T0`.
- **Standard errors, two ways.** *Observed*: the Hessian of −log L at the estimate by
  central differences (steps 0.01 for `f`, `k` and `BOUND`, 0.005 for γ and `T0`),
  recomputed at half the steps to check that the step does not matter. A fit within two
  steps of a parameter bound has no valid Hessian there; it is counted and left out of L2
  and L3. *Fisher at the truth*: `I = Σ_decisions Σ_outcomes (∂p/∂θ)(∂p/∂θ)ᵀ / p`. The
  outcomes are each choice in 0.01 s bins up to `T_DUR`, plus undecided. `p` comes from
  `fastlik` (lapses and `T0` included), and the derivatives from central differences with
  the same steps. It is conditional on the values and costs the decisions actually had.
  Either way, the SEs are the square roots of the diagonal of the inverse (`I⁻¹` for
  Fisher), and the correlation of γ with `f` is `(I⁻¹)_fγ / √((I⁻¹)_ff (I⁻¹)_γγ)`.
- **Calibration**: for each parameter, `z = (estimate − true) / SE`. Honest SEs make `z`
  close to standard normal: SD near 1, and about 95% of `|z|` under 1.96.

</details>

Claims fixed before any fit:

- **L1**: report RMSE, bias and r for each parameter (no threshold; the numbers are the
  result).
- **L2**: over fits away from a parameter bound, `z` with SE from the observed information
  has SD within 0.8 to 1.25 for every parameter, and the truth is within ±1.96 SE in 90 to
  98% of fits.
- **L3**: the same, with SE from the Fisher information at the truth.
- **L4**: report the correlation of γ with `f` from the inverse Fisher information.

100 people, 722 recorded decisions each on average (389 to 878; 422 of them redo
decisions):

| | RMSE | bias (mean of recovered − true) | r | SD of z, observed | coverage, observed | SD of z, Fisher | coverage, Fisher |
|---|---|---|---|---|---|---|---|
| `f` | 0.062 | −0.005 ± 0.006 | 0.987 | 0.97 | 89/93 (95.7%) | 0.98 | 91/93 (97.8%) |
| `k` | 0.054 | +0.013 ± 0.005 | 0.991 | 0.96 | 88/93 (94.6%) | 0.97 | 88/93 (94.6%) |
| γ | 0.028 | +0.001 ± 0.003 | 0.969 | 1.09 | 87/93 (93.5%) | 1.08 | 88/93 (94.6%) |
| `BOUND` | 0.045 | +0.002 ± 0.005 | 0.988 | 1.18 | 86/93 (92.5%) | 1.18 | 85/93 (91.4%) |
| `T0` (s) | 0.040 | +0.003 ± 0.004 | 0.946 | 1.14 | 83/93 (89.2%) | 1.10 | 84/93 (90.3%) |

- **L1: all five are recovered together.** r is 0.946 to 0.991, and four of the five biases
  are within 2 SE of zero. `k` is the exception: +0.013 ± 0.005, 2.45 SE, small against its
  range (0.2 to 1.5) but detectable. A likely source, not tested, is the rounding the fit
  does not see.
- **γ is measured to about ±0.03** (RMSE 0.028), continuously rather than on a grid.
- **L2 does not hold, on one number.** With observed-information SEs, every SD of z is
  inside 0.8 to 1.25 and four coverages are inside 90 to 98%, but `T0`'s is 89.2% (83 of 93):
  its error bars are slightly too narrow.
- **L3 holds.** With Fisher-information SEs, every SD of z is 0.97 to 1.18 and every coverage
  90.3 to 97.8%.
- **L4.** The correlation of γ with `f` in the inverse Fisher information has a median of
  −0.73 (5th to 95th percentile −0.88 to −0.47; −0.75 from the observed information): a
  higher γ and a lower `f` partly trade off, which the standard errors already carry.
- **Seven fits sat at a parameter bound** and were left out of L2 and L3, as fixed before the
  run (they count in L1). Six had a true `T0` of 0.06 s or less and were fitted at 0. One had
  γ fitted at 0.5 against a true 0.61: a person with `f` 0.26, whose decisions carry little
  value signal.
- **The current way of fitting does far worse on the same data.** With γ held at 0.9 and
  `BOUND` at 1.5, `f` has RMSE 0.469 and bias −0.274 ± 0.038 (all five free: 0.062), and `k`
  RMSE 0.262 and bias +0.098 ± 0.024 (all free: 0.054).
- **The finite-difference step does not matter**: the median ratio of γ's SE at half the
  step to its SE at the step is 0.999.

#### Profile-likelihood intervals

*2026-10-28 · `python -m experiments.lhc_profile`*

The standard errors above give symmetric intervals, and none at all for the 7 fits at a
parameter bound. A profile-likelihood interval needs neither: for each parameter it is every
value whose profile (−log L with the other four re-fitted) is within 1.92 of the minimum,
clipped to the parameter's range, so a fit at `T0` = 0 gets [0, upper]. The data, likelihood
and estimates are the ones above. The profile was walked outward in steps of 0.4 observed
SEs, the other four re-fitted by L-BFGS-B at each step, and the 1.92 crossing placed by
linear interpolation.

Claims fixed before the run:

- **Q1**: for every parameter, the profile intervals contain the truth in 90 to 98% of all
  100 fits, those at a bound included.
- **Q2**: `T0`'s contain it in at least 90% of the 93 fits away from a bound.

| | coverage, all 100 | coverage, 93 away from a bound | observed-information (Wald), the same 93 | median width | Wald width |
|---|---|---|---|---|---|
| `f` | 97.0% | 96.8% | 95.7% | 0.233 | 0.237 |
| `k` | 95.0% | 94.6% | 94.6% | 0.184 | 0.184 |
| γ | 93.0% | 92.5% | 93.5% | 0.059 | 0.058 |
| `BOUND` | 93.0% | 92.5% | 92.5% | 0.141 | 0.142 |
| `T0` (s) | 91.0% | 90.3% | 89.2% | 0.114 | 0.125 |

- **Q1 and Q2 hold.** Every coverage over all 100 fits is 91 to 97%, and `T0`'s over the 93
  is 90.3%, against 89.2% for its Wald intervals. No profile point fell below an estimate's
  −log L, so every fit was at its maximum.
- **All 7 fits at a bound are covered**, with intervals that run from the bound, as they
  should (the 6 at `T0` = 0 and the one at γ = 0.5).
- **Away from a bound the two kinds of interval are almost the same**, in coverage and in
  width: the median widths (profile over all 100, Wald over the 93) are within 2% of each
  other, except `T0`'s, 9% narrower by profile. The likelihood is close to quadratic there,
  so the observed-information SEs were not the problem; leaving out the fits at a bound was.

#### From less data, alone and pooled

*2026-10-28 · `python -m experiments.lhc_size`*

A real person may give far fewer than 722 decisions. The same 100 histories were refitted
from their first N start decisions, N = 50, 100, 200 and 300, each with the same share of
its redo decisions. The histories store starts and redos in order but not their times, so
the first round(N / 300 × redos) redos stand in for those made before the N-th start. Each
cut is fitted two ways:

- **Alone**: maximum likelihood, as above.
- **Pooled**: empirical Bayes. The 100 people are taken as normal on an unbounded scale
  (log `f`, log `k`, logit of γ in (0.5, 0.99), log `BOUND`, logit of `T0` in (0, 0.5)).
  The population's mean and covariance are fitted by expectation-maximization with a
  Laplace approximation (each person's MAP and inverse Hessian, then the population's
  moments; Huys et al., 2011), run until the mean moves less than 0.001 (5 to 8 rounds).
  The pooled 95% interval is the MAP ± 1.96 posterior SD on that scale, mapped back. The
  people were drawn uniformly from a box, so the normal population is an approximation, as
  it would be for real people.

Claims fixed before the run, at N = 100, paired over people:

- **H1**: pooled squared error in `f` is below alone beyond 2 SE.
- **H2**: the same for `k`.
- **H3**: the pooled intervals contain the truth in 90 to 98% of people for every
  parameter.

RMSE:

| N starts (decisions) | `f` alone / pooled | `k` alone / pooled | γ alone / pooled | `BOUND` alone / pooled | `T0` alone / pooled |
|---|---|---|---|---|---|
| 50 (120) | 0.159 / 0.147 | 0.166 / 0.150 | 0.080 / 0.058 | 0.105 / 0.093 | 0.098 / 0.075 |
| 100 (241) | 0.097 / 0.091 | 0.100 / 0.095 | 0.054 / 0.040 | 0.078 / 0.071 | 0.067 / 0.059 |
| 200 (482) | 0.071 / 0.068 | 0.066 / 0.065 | 0.035 / 0.026 | 0.052 / 0.050 | 0.046 / 0.043 |
| 300 (722) | 0.062 / 0.060 | 0.054 / 0.054 | 0.028 / 0.022 | 0.045 / 0.044 | 0.040 / 0.038 |

Pooled coverage:

| N starts | `f` | `k` | γ | `BOUND` | `T0` |
|---|---|---|---|---|---|
| 50 | 87% | 88% | 90% | 90% | 86% |
| 100 | 93% | 98% | 92% | 88% | 87% |
| 200 | 96% | 96% | 95% | 94% | 86% |
| 300 | 96% | 90% | 95% | 93% | 85% |

- **Alone, the fit degrades gracefully.** From 100 start decisions (241 with their redos),
  `f` and `k` are measured to RMSE 0.10 and γ to 0.05. At 300 the fits reproduce the
  full-data recovery exactly (0.062, 0.054, 0.028, 0.045, 0.040).
- **H1 and H2 do not hold.** At N = 100, pooling lowers squared error in `f` by
  0.0012 ± 0.0009 (1.3 SE) and in `k` by 0.0011 ± 0.0006 (1.8 SE). It helps where a parameter
  is measured worst: γ beyond 2 SE at every N up to 200 (−0.0013 ± 0.0005 at 100), the bound
  and `T0` beyond 2 SE at 50 and 100, and `k` at 50 (−0.0051 ± 0.0025, 2.0 SE).
- **H3 does not hold.** At N = 100 the pooled intervals for the bound (88%) and `T0` (87%) run
  narrow, and `T0`'s stay at 85 to 87% at every N: on the logit scale near `T0` = 0 the
  Laplace approximation is poor. `f`, `k` and γ are inside 90 to 98% from N = 100 on.
- So pooling is worth having for γ, the bound and `T0` when data are few, but not for `f` and
  `k`. Its `T0` intervals should not be used; the profile intervals above are honest.

### Measuring the learning rate and the reward (2026-10-28)

With γ and the bound fitted, two things are still assumed: the learning rate α (0.05), and
the reward, right answers worth `1 − p` ("achievement") rather than the fraction right
("accuracy"). Both can be replayed the way γ is.

*`python -m experiments.alpha_reward`*

- **Shadow learners** run at every γ from 0.50 to 0.99 (steps of 0.01), every α in 0.01,
  0.015, 0.025, 0.035, 0.05, 0.07, 0.1, 0.14 and 0.2 (about √2 apart), and under both
  rewards, each paid what its reward pays for the same answers.
- **People**: 64 from a seeded Latin hypercube over `f` and `k` (0.2 to 1.5), γ (0.6 to
  0.97), `BOUND` (1.0 to 2.0), `T0` (0 to 0.4 s) and log α (0.01 to 0.2). Even-numbered
  people value achievement and odd-numbered ones accuracy. Each plays the app's task (whose
  `InfoController` assumes γ 0.9, α 0.05 and achievement) for 300 start decisions plus
  their redo decisions.
- **The fit**: for each grid α and each reward, all five parameters are fitted by maximum
  likelihood (γ continuous). That gives a profile of −log L over α for each reward. α̂ is
  the vertex of a parabola in log α through the lowest grid point and its neighbours. The
  95% interval is where the profile is within 1.92 of its minimum. The chosen reward is the
  one with the lower minimum.

Claims fixed before the run:

- **A1**: under the true reward, α̂ is within a factor of 1.5 of the truth in at least 80% of
  runs.
- **A2**: its 95% interval contains the truth in at least 90%.
- **R1**: the reward with the lower minimum is the true one in at least 90%.

- **A1 does not hold: α is measured, but loosely.** α̂ was within a factor of 1.5 of the
  truth for 53% of people. The median error was 39%, and r(log α̂, log α) was 0.66.
- **A2 holds, with wide intervals.** They covered the truth for 95% of people and span a
  factor of 5.1 end to end at the median: from 300 start decisions plus redos, the data say
  roughly which decade α is in, and not much more.
- **R1 holds: which reward a person values can be measured.** The true reward was chosen for
  63 of 64 people (all 32 who value accuracy, 31 of 32 who value achievement). The median
  margin in −log L was 7.6 for accuracy and 22.5 for achievement.
- **Choosing the reward removes the bias that no fitted parameter absorbed.** Bias in `f` for
  people who value accuracy:
  - the current fit (α 0.05, achievement assumed): −52.9 ± 1.9% (RMSE 0.509)
  - α fitted, achievement still assumed: −50.4 ± 1.9% (RMSE 0.474)
  - α and the reward both chosen by the profile: +3.7 ± 2.5% (RMSE 0.109)

  Their `k` moves from −19.7 ± 3.2% to +1.7 ± 1.3%. For people who value achievement,
  choosing costs little: `f` +2.1 ± 2.7% (RMSE 0.138) the current way, +6.6 ± 3.7% (RMSE
  0.110) with both chosen. Fitting α alone cut their `f` RMSE from 0.138 to 0.084. γ's mean
  bias was within 1% under every fit.

### Which difficulty and whether to show feedback

Which target keeps people engaged and earns the most reward, and does showing right/wrong
help? These runs use no controller: one setting is held for each whole run.

*2026-10-25 · `python -m experiments.task_choice`*

Question-type task, skips on, one setting held for each run of 300 ticks, review 0.25,
skipped questions dropped. The target is the predicted chance of a right answer each
question is aimed at. It is a prediction from the ratings; the chance actually achieved was
not recorded. γ = 0.9, `ACHIEVEMENT_SCALE = 4`, continuing bootstrap, `T_DUR = 7.0`.

Claims fixed before the run (paired on seed):

- a peak is determined if it beats both neighbouring targets beyond 2 SE;
- the feedback crossover is established if shown − hidden is positive beyond 2 SE at one
  target and negative beyond 2 SE at a harder one.

#### Engagement by difficulty

Engagement has two parts: whether a person starts the next set when asked, and how long
they stay away after they stop. P(start next set) is the share of recorded start decisions
on which they chose to start. Time away is the share of the 300 ticks spent out of the
task. Both by target, feedback shown, 16 seeds (descriptive; no claim was fixed in advance
for either):

| person | 90% | 75% | 60% | 45% | 30% | 20% | targets within 2 SE of the best |
|---|---|---|---|---|---|---|---|
| f 0.6, k 1.2 | 0.55 | 0.74 | **0.87** | 0.81 | 0.55 | 0.56 | 60 |
| f 1.0, k 1.0 | 0.74 | 0.92 | **0.94** | 0.89 | 0.80 | 0.61 | 60, 45 |
| f 0.8, k 0.8 | 0.72 | 0.91 | **0.92** | 0.90 | 0.78 | 0.62 | 75, 60, 45 |
| f 0.4, k 0.4 | 0.65 | 0.78 | 0.84 | 0.86 | **0.87** | 0.76 | 45, 30 |
| f 1.2, k 0.4 | 0.83 | 0.95 | 0.94 | 0.94 | **0.96** | 0.89 | 75, 30 |

| share of ticks away | 90% | 75% | 60% | 45% | 30% | 20% |
|---|---|---|---|---|---|---|
| f 0.6, k 1.2 | 0.90 | 0.82 | **0.63** | 0.74 | 0.96 | 0.97 |
| f 1.0, k 1.0 | 0.65 | 0.36 | **0.28** | 0.30 | 0.67 | 0.96 |
| f 0.8, k 0.8 | 0.63 | 0.28 | **0.25** | 0.30 | 0.62 | 0.92 |
| f 0.4, k 0.4 | 0.53 | 0.34 | 0.27 | **0.25** | 0.27 | 0.51 |
| f 1.2, k 0.4 | 0.25 | **0.07** | 0.09 | 0.11 | 0.10 | 0.29 |

- **Engagement is a plateau, and effort aversion sets its width.** Each person engages about
  equally well over one to three targets. For three of the five, P(start) and time away put
  the best target in the same place. For the two with `k = 0.4` they differ by one or more
  targets inside a flat plateau. Low `k` keeps the plateau wide into hard questions. The
  most effort-averse person's plateau is a single target (60%), and even there they are
  away 63% of the time.
- **Easy questions lose people.** At a 90% target P(start) falls to 0.55 to 0.83, and the
  three people with `k ≥ 0.8` are away 63 to 90% of the time. Near-certain right answers pay
  almost nothing, which is the model's boredom. Hard questions lose the effort-averse
  faster: at 20% those three are away 92 to 97% of the time.

#### Reward by difficulty

Total reward by target, the same runs:

| person | 90% | 75% | 60% | 45% | 30% | 20% | peak |
|---|---|---|---|---|---|---|---|
| f 0.6, k 1.2 | 10.9 | 27.6 | **65.3** | 57.2 | 7.2 | 3.2 | 60%, not determined |
| f 1.0, k 1.0 | 24.1 | 67.5 | 94.3 | **114.3** | 59.7 | 4.4 | 45%, determined |
| f 0.8, k 0.8 | 25.1 | 75.0 | 98.6 | **112.4** | 68.0 | 8.7 | 45%, not determined |
| f 0.4, k 0.4 | 30.9 | 70.4 | 96.1 | 120.0 | **130.8** | 78.8 | 30%, not determined |
| f 1.2, k 0.4 | 36.4 | 83.7 | 98.4 | 119.3 | **140.7** | 93.5 | 30%, determined |

- **Reward peaks at or past the engagement peak.** Within the plateau each right answer is
  worth more at a harder target, so reward keeps rising until engagement falls. For f 1.0,
  k 1.0 and f 0.8, k 0.8, reward picks 45% where engagement picks 60%; for the other three
  the two agree. The two people with `k = 0.4` peak at 30%, which is why the target range
  reaches down to 20%.

#### Showing feedback, one person

Feedback shown − hidden, `f = k = 0.8`, 96 paired seeds:

| target | 90% | 75% | 60% | 45% | 30% | 20% |
|---|---|---|---|---|---|---|
| P(start) | **+0.027 ± 0.008** | **+0.021 ± 0.006** | **+0.015 ± 0.003** | **+0.019 ± 0.007** | **+0.069 ± 0.016** | +0.009 ± 0.014 |
| share of ticks away | **−0.056 ± 0.009** | **−0.071 ± 0.014** | **−0.065 ± 0.008** | **−0.079 ± 0.018** | **−0.119 ± 0.024** | −0.029 ± 0.015 |
| total reward | **+2.3 ± 0.8** | **+4.1 ± 2.0** | +1.7 ± 1.3 | +5.9 ± 3.0 | **+19.7 ± 5.0** | +5.4 ± 2.9 |

- **Showing feedback raises engagement at every target but the hardest**, by 0.015 to 0.069
  in P(start) and 5.6 to 11.9 points less time away, each beyond 2 SE. At 20% both are
  within 2 SE. A seen right answer lands as reward at once, instead of being discounted to
  the end of the set.
- **On reward it helps at three targets and is undetermined at the other three.** It is
  ahead beyond 2 SE at 90%, 75% (2.05 SE, the narrowest) and 30% (3.9 SE), and never behind.
  At hard targets a run's total depends heavily on runs of luck, so the paired difference
  varies several times more than at 90%.
- **No crossover.** Hiding feedback never won on reward beyond 2 SE, so by the rule above the
  crossover is absent. The earlier result that pointed to a flip at 30% is in `RETIRED.md`.

**The five people of the first table, per person** (added 2026-10-28, after the results were
known; `python -m experiments.intervals`). Shown − hidden in P(start), each person's mean
over six targets and 16 paired seeds. Each seed is first averaged over the six targets, so
a person's interval is a t interval over 16 values (15 df):

| person | shown − hidden, P(start) [95% interval] |
|---|---|
| f 0.4, k 0.4 | +0.034 [+0.025, +0.042] |
| f 0.6, k 1.2 | +0.020 [−0.015, +0.054] |
| f 0.8, k 0.8 | +0.016 [+0.002, +0.030] |
| f 1.0, k 1.0 | +0.028 [+0.009, +0.047] |
| f 1.2, k 0.4 | +0.021 [+0.005, +0.036] |
| **across the five** (t, 4 df) | **+0.024 [+0.015, +0.033]** |

All five are above zero. A one-sample test on the 30 person × target means gives the same
mean with p = 6e-5, but it treats the six cells of one person as independent, so its p is
too small.

#### Across people

*2026-10-25 · `python -m experiments.feedback_grid`*

The 16 people of the grid, six targets, shown and hidden, 16 paired seeds each. Claims
fixed before the run:

- showing feedback never lowers engagement if at most 4 of the 96 person × target cells fall
  below zero beyond 2 SE, and none beyond 3 SE;
- for a person, it raises engagement if it helps beyond 2 SE at every target where that
  person starts at least 10% of the time with feedback shown.

Findings:

- **The first claim fails, on one cell**: f 0.6, k 1.0 at a 20% target, where P(start) was
  0.49 shown and 0.55 hidden (−0.060 ± 0.018, 3.2 SE). No other cell fell below zero beyond
  2 SE, and 73 of 96 were positive. That cell is one of 96 tests and should not carry weight
  alone, but the claim as fixed does not hold.
- **The per-person claim held for none of 16**; 16 seeds per cell are too few. Pooled over
  targets (descriptive), showing feedback raised P(start) beyond 2 SE for 8 of 16 people, by
  0.015 to 0.032: exactly the eight with `k ≤ 0.6`.
- **By difficulty, over all 16 people** (256 paired runs per target), shown − hidden; P(start)
  is beyond 2 SE at every target but 20%, and reward where bold:

  | target | 90% | 75% | 60% | 45% | 30% | 20% |
  |---|---|---|---|---|---|---|
  | P(start) | +0.012 | +0.017 | +0.019 | +0.020 | +0.037 | +0.001 |
  | total reward | +0.7 ± 0.5 | **+2.4 ± 1.1** | **+3.0 ± 1.3** | +4.3 ± 2.4 | **+5.8 ± 2.0** | +3.4 ± 1.8 |

- **f and k set the two ends of the engagement curve** (feedback shown). At a 90% target
  P(start) is 0.20 to 0.57 for `f = 0.2` and 0.78 to 0.91 for `f = 1.5`: easy questions lose
  low-`f` people most. At 20% it is 0.67 to 0.94 for `k = 0.2` and 0.07 to 0.36 for
  `k = 1.5`. The best target ranges from 30% to 75% across the 16 people. Single peaks are
  noisy at 16 seeds; the two ends are not.

#### Feedback by γ

*2026-10-28 · `python -m experiments.gamma_feedback`*

Every result above is at γ 0.9. Feedback only moves reward in time, and only discounting
makes that matter. Discounted to a set's first slot, one lump at the last of its 4 slots is
worth 0.27 of the same achievement paid slot by slot at γ 0.5, 0.85 at 0.9, and 0.99 at
0.99.

The task and the five people of the first table, at γ 0.5, 0.7, 0.8, 0.9, 0.95 and 0.99,
all six targets, 32 seeds paired across γ, target and feedback (`ACHIEVEMENT_SCALE` 4,
continuing bootstrap, `T_DUR` 7.0 s). Claim fixed before the run: the effect depends on γ
for a person if P(start) shown − hidden, averaged over targets, differs between γ 0.5 and
0.99 by more than 2 SE.

- **It holds for three of five**: f, k = 0.8/0.8 (+0.036 at 0.5, −0.002 at 0.99), 0.4/0.4
  (+0.053, −0.014) and 1.2/0.4 (+0.071, −0.002), all three about zero from γ 0.95 on. For
  0.6/1.2 (+0.029, +0.014) and 1.0/1.0 (+0.024, +0.010) the drop is within 2 SE.
- **Reward and time away follow the same course** (no claim). Showing feedback cut the share
  of ticks away by up to 0.19 (1.2/0.4 at γ 0.8), and by at most 0.04 at γ 0.95 and above.
- So the effect of feedback on starting is a measure of discounting, which
  `experiments/gamma_blocks.py` uses ("Feedback switched in blocks", under "Fitting the
  discount factor").

#### Lockout

*2026-10-25 · `python -m experiments.lockout`*

Lockout is a run in which the person almost never starts again, defined before the run as a
start rate below 0.10 over the start decisions the run records. Four effort-averse people
(f, k = 0.6/1.0, 0.6/1.2, 1.0/1.2, 1.0/1.5), targets 75, 60 and 45%, 64 paired seeds each,
300 ticks. Claims fixed before the run: lockout is more common with feedback hidden, pooled
and for each person.

- **Neither holds.** Pooled, 1.8% against 1.7% of 768 runs (14 and 13), +0.001 ± 0.001. Per
  person, it holds for none.
- **The definition no longer finds what it was written for.** A session now ends at its first
  stop, so a recorded start rate below 0.10 is rare; a person who withdraws shows it as time
  away instead. Measured that way (descriptive), feedback hidden kept the four away 9.0 ±
  0.8 points more of the time (0.61 against 0.52, 0.76 against 0.72, 0.54 against 0.41 and
  0.71 against 0.62 per person). In runs that did not lock out, showing feedback raised the
  start rate by 0.010 to 0.034.

### Measuring against planning for reward

The reward controller's evaluation under the decision rule (8 seeds, five people, two of
them defined by the retired reaction terms) is in `RETIRED.md` (V2). It has not been
repeated under the current model. The comparison that has been is below.

#### Measuring costs no detectable reward

*2026-10-25 · `python -m experiments.controllers`, which prints this table, the paired
tests, "Over time" and "What measuring costs, block by block"; it fits every arm at 100,
200 and 300 ticks*

Six ways of choosing settings were run on the same 32 simulated people: the grid × 2 seeds.
Each run lasted 300 ticks, on the question-type task with skips on, and the controllers
never saw the true `f` and `k`. γ = 0.9, `ACHIEVEMENT_SCALE = 4`, continuing bootstrap,
`T_DUR = 7.0`. The six:

- **fixed default**: the default setting throughout;
- **random**: a random setting every 25 start decisions, from the 84 scheduled-review
  settings, and **random over 96**: the same over all 96, redo-prompt settings included;
- **for reward, 84 settings**: reward planning over the 84 scheduled-review settings (the
  app's controller until 2026-10-11);
- **for reward, 96 settings**: reward planning over all 96;
- **for information**: `InfoController` over all 96 (the app's controller until 2026-10-29).

Pass conditions, fixed before the run and paired on person and seed at the end of the run:

- **P1**: squared error in `k`, measuring − reward planning (96 settings), is below zero
  beyond 2 SE, and squared error in `f` is not above zero beyond 2 SE;
- **P2**: reward, measuring − fixed default, is above zero beyond 2 SE.

The four arms that do not choose at random:

| how settings were chosen | f RMSE | k RMSE | total reward | P(start next set) | start decisions | ticks away | redo settings chosen |
|---|---|---|---|---|---|---|---|
| fixed default | 0.149 | 0.724 | 55.3 | 0.80 | 182 | 39% | 0% |
| for reward, 84 scheduled-review settings (the app's controller until 2026-10-11) | 0.102 | 0.663 | 136.2 | 0.85 | 216 | 28% | 0% |
| for reward, all 96 settings | **0.074** | 0.508 | 132.8 | 0.85 | 212 | 29% | 31% |
| for information, all 96 (`InfoController`) | 0.091 | **0.156** | **136.7** | 0.85 | 209 | 30% | 99% |

Paired differences over the 32 people:

- **P1 does not pass.** Squared error in `k`, measuring − reward planning (96 settings), was
  −0.234 ± 0.140: in measuring's favour, but 1.7 SE, short of the 2 SE fixed before the run.
  For `f` it was +0.003 ± 0.004, undetermined, so `f` is not worse. Against the app's
  previous controller (84 settings) the `k` gain is beyond 2 SE: −0.415 ± 0.154 (2.7 SE).
- **P2 passes.** Reward, measuring − fixed default: +81.4 ± 9.4.
- **Over the whole run, measuring cost no detectable reward.** Measuring − reward planning:
  +3.9 ± 10.7, and measuring − previous controller: +0.5 ± 11.5, both undetermined. Block by
  block it costs reward early and repays it late (below).
- **Giving the reward controller redo prompts measured `k` better.** 96 − 84 settings: `k`
  squared error −0.181 ± 0.087 (2.1 SE); reward −3.4 ± 4.2, undetermined. It used redo
  prompts on 31% of updates (32 redo decisions per person, against 302 for measuring).
- **Reward planning leaves `k` poorly identified.** At 0.51 to 0.66 RMSE, `k` is not much
  better than with the fixed default (0.72). `k` came out below 0.05 for 3 of 32 people
  under the previous controller and 2 under the 96-setting one, with true `k` 0.2 or 0.6.
  None did under measuring. That fits the reason the measuring controller exists: reward
  planning settles on a setting, so cost stops varying. The spread of cost in "Over time"
  bears it out: 0.023 and 0.049 under reward planning at 300 ticks, against 0.282 under
  measuring.

#### It found the redo prompts on its own

The redo decision is where cost varies most (`0.15 · min(1/p, 8)`, about 4× across
difficulties), and `InfoController` chose a redo-prompt setting on 99% of its updates.
Random choice over the same 96 settings picked one 14% of the time and saw about 2 redo
prompts per person.

#### Over time

The same people and seeds, fitted at 100, 200 and 300 ticks on everything recorded by then.
The spread of cost is the standard deviation of cost across those decisions.

| how settings were chosen | k RMSE at 100 / 200 / 300 | f RMSE at 100 / 200 / 300 | sd of cost at 300 | decisions in the fit at 100 / 200 / 300 |
|---|---|---|---|---|
| fixed default | 1.035 / 0.956 / 0.724 | 0.196 / 0.171 / 0.149 | 0.012 | 68 / 132 / 182 |
| random every 25 start decisions, 84 settings | 0.888 / 0.841 / 0.807 | 0.203 / 0.188 / 0.185 | 0.029 | 54 / 106 / 143 |
| for reward, 84 settings | 0.975 / 0.795 / 0.663 | 0.148 / 0.117 / 0.102 | 0.023 | 70 / 148 / 216 |
| for reward, 96 settings | 0.525 / 0.584 / 0.508 | 0.101 / 0.089 / 0.074 | 0.049 | 88 / 172 / 244 |
| for information (`InfoController`) | **0.193 / 0.180 / 0.156** | 0.103 / 0.091 / 0.091 | **0.282** | 149 / 333 / 512 |

Choosing for information measures `k` better at 100 ticks than every other way does at 300,
and it collects data in which cost varies 6 to 12 times as much as under reward planning.
It does not measure `f` better: reward planning over 96 settings is ahead at every
checkpoint. Random settings measure `k` worst by the end, and they also collect the fewest
decisions (143 in 300 ticks): people under them were away 52% of the time, against 28 to
39% under the other ways.

**The RMSEs are dominated by people who give few decisions.** In 300 ticks, 4 to 14 of the
32 people per arm gave fewer than 100 decisions in the fit, and their `k` estimates run to
the bounds (2 to 5 per arm reached 3.0, none under measuring). Among the rest (described,
not claimed), `k` RMSE is 0.046 under measuring, 0.258 and 0.341 under reward planning (96
and 84 settings), 0.327 under random settings and 0.632 at the fixed default. In the paired
tests above, the three worst people carry 80% of reward planning's squared error in `k` and
84% of measuring's, which is why those tests are noisy.

#### What measuring costs, block by block

The same runs (the five arms below × 32 people, 160 runs), with reward and start decisions
recorded at 100, 200 and 300 ticks. No claims were fixed before this run, so these are
descriptive.

| how settings were chosen | reward, ticks 1-100 / 101-200 / 201-300 | P(start), the same blocks |
|---|---|---|
| fixed default | 32.5 / 14.5 / 8.3 | 0.80 / 0.83 / 0.76 |
| random every 25 start decisions, 84 settings | 35.6 / 13.2 / 5.4 | 0.76 / 0.74 / 0.66 |
| for reward, 84 settings | 56.8 / 50.7 / 28.8 | 0.83 / 0.87 / 0.83 |
| for reward, 96 settings | 55.8 / 47.0 / 30.0 | 0.82 / 0.88 / 0.82 |
| for information (`InfoController`) | 43.4 / 47.1 / 46.2 | 0.83 / 0.85 / 0.87 |

Paired differences, information − reward planning (84 / 96 settings):

- **Ticks 1-100:** reward −13.3 ± 3.4 / −12.4 ± 3.4, both beyond 2 SE. P(start)
  −0.004 ± 0.026 / +0.004 ± 0.026, both undetermined.
- **101-200:** reward −3.6 ± 6.1 / +0.1 ± 5.5; P(start) −0.023 ± 0.024 / −0.016 ± 0.024. All
  undetermined.
- **201-300:** reward +17.4 ± 6.8 / +16.3 ± 7.2, both beyond 2 SE. P(start)
  +0.024 ± 0.029 / +0.028 ± 0.035, both undetermined.

So measuring costs about 13 reward in the first 100 ticks and earns about 17 more in the
last 100, where the reward planners' reward falls off and measuring's holds. The totals over
300 ticks hide this by netting the two. P(start) differs detectably in no block.

#### Intervals

*Added 2026-10-28, after the results were known; descriptive, beside the 2 SE rule ·
`python -m experiments.intervals`*

A bootstrap over the 32 people: people drawn with replacement, each person's arms kept
together so the pairing holds, and each arm's RMSE, mean reward and mean P(start)
recomputed on the resample. 10,000 resamples from a generator seeded at 0, 95% percentile
intervals of the difference. Information minus each other way:

| | default | random | reward, 84 | reward, 96 |
|---|---|---|---|---|
| k RMSE, 100 ticks | −0.84 [−1.07, −0.61] | −0.70 [−0.84, −0.54] | −0.78 [−0.97, −0.58] | −0.33 [−0.50, −0.15] |
| k RMSE, 300 ticks | −0.57 [−0.72, −0.42] | −0.65 [−0.86, −0.44] | −0.51 [−0.72, −0.29] | −0.35 [−0.62, −0.07] |
| f RMSE, 300 ticks | −0.06 [−0.11, −0.02] | −0.09 [−0.18, −0.01] | −0.01 [−0.06, +0.02] | +0.02 [−0.03, +0.07] |
| total reward | +81 [+63, +99] | +83 [+63, +101] | +0.5 [−22, +23] | +3.9 [−16, +25] |
| P(start) | +0.045 [+0.004, +0.085] | +0.091 [+0.056, +0.128] | +0.001 [−0.037, +0.037] | +0.002 [−0.044, +0.041] |

Against reward planning, P(start) is within about ±0.04 and total reward within about ±23:
measuring costs no engagement that these 32 people can detect at that resolution.

### Four controllers, all five parameters fitted (2026-10-29)

*`python -m experiments.controller_full --arms random_return,engage,info_fkg` · paper
Figure 8 (`fig8lhc`)*

Four ways of choosing settings on the 128 people of "Planning with fitted γ and bound" (a
Latin hypercube over `f`, `k`, γ, `BOUND` and `T0`), each person's data fitted with all
five free.

- **Fixed default**: the default setting throughout (the runs of the baselines below,
  reused).
- **Random**: a random setting at the start, and again each time the person comes back
  after stopping, applied before the set they come back to.
- **Engagement maximizer** (`EngagementController`): the setting whose 50-tick rollouts
  keep the person in the task for the largest share of ticks. A stop costs every tick
  until they come back, so this rewards both starting the next set and returning.
- **Information planning** (`FkgInfoController`): the rise in log det of the information
  about `f`, `k` and γ, with `BOUND` and `T0` as nuisances (Ds-optimal), and
  `InfoController`'s safety floor.

Both planners fit, draw and roll out all five parameters as `FullInfoController` does, plan
every 25 start decisions, and roll every setting out on the same seed per draw. The three
that choose have the same 384 settings (`END_SETTINGS`, below). 300 ticks; α 0.05;
cold-start cost 0.6; `ACHIEVEMENT_SCALE = 4`, continuing bootstrap, `T_DUR = 7.0`. People 0
to 89 ran on the server and 90 to 127 on the Mac. Three Mac tasks that had started while
V5 code was on disk were re-run under V4 (`runs_mac_four_[bcd].csv`); 19 others re-run as
a check reproduced exactly.

Claims fixed before the run, paired at 300 ticks: **H1**, squared error in γ, information
− engagement, below zero beyond 2 SE; **H2**, the same for `k`; **H3**, squared error in
`f`, information − engagement, not above zero beyond 2 SE; **H4**, the share of ticks in the
task, engagement − fixed default, above zero beyond 2 SE.

| how settings were chosen | f RMSE at 100 / 200 / 300 | k RMSE at 100 / 200 / 300 | γ RMSE at 300 | ticks in the task | P(start) | start decisions | total reward | feedback shown | questions per set | target |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed default | 0.400 / 0.388 / 0.305 | 0.954 / 0.987 / 0.868 | 0.119 | 41% | 0.73 | 123 | 45.1 | 100% | 4 | 75% |
| random | 0.387 / 0.314 / 0.239 | 0.706 / 0.639 / 0.609 | 0.111 | 30% | 0.69 | 91 | 49.4 | 50% | 4.9 | 53% |
| engagement | 0.260 / 0.186 / 0.173 | 0.704 / 0.625 / 0.579 | 0.090 | **58%** | **0.84** | **173** | 52.7 | 61% | 2.5 | 68% |
| information | 0.325 / 0.242 / **0.160** | **0.368 / 0.318 / 0.201** | 0.094 | 28% | 0.71 | 83 | 82.4 | 10% | 6.3 | 42% |

- **H2, H3 and H4 hold.** Squared error in `k`, information − engagement: −0.225 ± 0.053;
  in `f`: −0.003 ± 0.009. Time in the task, engagement − fixed default: +16.8 ± 1.7 points.
  Among people with at least 100 decisions in the fit, `k` RMSE is 0.112 under information
  planning, against 0.34 to 0.78 under the others.
- **H1 does not hold.** Squared error in γ, information − engagement: −0.001 ± 0.002. No
  arm measures γ across the 128 people better than guessing the middle of its range
  (RMSE 0.107).
- **γ is measured where a person gives start decisions, and not otherwise.** γ RMSE at
  300 ticks by the start decisions a person gave (people in each group):

  | start decisions | fewer than 50 | 50 to 99 | 100 to 199 | 200 or more |
  |---|---|---|---|---|
  | fixed default | 0.177 (42) | 0.109 (19) | 0.105 (34) | 0.028 (33) |
  | random | 0.151 (65) | 0.104 (17) | 0.032 (23) | 0.009 (23) |
  | engagement | 0.163 (36) | 0.120 (5) | 0.051 (17) | 0.025 (70) |
  | information | 0.129 (70) | 0.048 (11) | 0.018 (26) | 0.006 (21) |

  At the same number of start decisions, information planning measures γ best, as it is
  built to. But it keeps people out of the task, so 70 of its 128 people gave fewer than
  50, where nothing measures γ. The engagement maximizer gave 70 people 200 or more. For γ,
  keeping a person in the task is most of measuring them.
- **What information planning costs** (95% t intervals over people). Against the
  engagement maximizer: 30.2 points less time in the task [25.1, 35.3], P(start) −0.133
  [−0.162, −0.103], 91 fewer start decisions [75, 106], and 29.6 more reward [16.1, 43.2],
  from harder questions (a 42% target against 68%). Against the fixed default: 13.4 points
  less time in the task [10.0, 16.8], with P(start) −0.022 [−0.052, +0.007].
- **What each chose.** Information planning hid feedback before 90% of start decisions and
  gave 180 end-of-set redo decisions per person, in sets of 6.3 questions. The engagement
  maximizer chose short sets (2.5 questions; 2.1 for people with `k` up to 0.6, 2.8 above
  1.0), easier questions, and feedback shown before 61% of start decisions.

#### The app's InfoController and reward planning, on the same people

*`python -m experiments.controller_full --arms info_end,random_end,reward_end`*

"Measuring against planning for reward" gave every simulated person γ 0.9 and `BOUND` 1.5,
the values both planners and their fits assume, and fitted only `f` and `k`. Here the same
two planners, `InfoController` and reward planning, choose for the 128 people above, and
every arm's data are fitted with all five free. Both still plan with γ at 0.9 and `BOUND`
at 1.5.

- **Settings.** Every arm that chooses has the same 384 (`END_SETTINGS`): the 84
  scheduled-review settings; redo prompts mid-set with feedback shown, or at the end of the
  set with feedback hidden, each redo's outcome then shown as in the app; each at 2, 4, 6
  or 8 questions per set. So every redo-prompt setting yields redo decisions. (The 96
  settings of `InfoController` include six with redo prompts and feedback hidden, which
  never offer a redo; `RETIRED.md`.)
- **Arms**, paired on person and seed: the fixed default (the default setting, 4 per set);
  a random setting every 25 start decisions; reward planning; and `InfoController`. Both
  planners hold γ at 0.9 and `BOUND` at 1.5.
- 300 ticks; α 0.05; cold-start cost 0.6; `ACHIEVEMENT_SCALE = 4`, continuing bootstrap,
  `T_DUR = 7.0`. All five parameters fitted by maximum likelihood at 100, 200 and 300 ticks.
  The fit's shadow learners take an end-of-set redo's reward as the person does
  (`GridShadow._extra_question`; at the person's own γ the shadow matches the person
  exactly).
- People 0 to 89 ran on the server and 90 to 127 on the Mac. The server rebooted after 200
  of its 270 runs; the 70 missing were re-run on both (`runs_*_end*.csv`; tasks are seeded
  per person, so a re-run is the same run).

Claims fixed before the run, paired at 300 ticks: **K1**, squared error in `k`,
information − reward planning, below zero beyond 2 SE; **K2**, the same against the fixed
default; **K3**, squared error in `f`, information − reward planning, not above zero
beyond 2 SE; **K4**, P(start), information − reward planning, not below zero beyond 2 SE.

| how settings were chosen | f RMSE at 100 / 200 / 300 | k RMSE at 100 / 200 / 300 | γ RMSE at 300 | P(start) | total reward | ticks away | start decisions |
|---|---|---|---|---|---|---|---|
| fixed default | 0.400 / 0.388 / 0.305 | 0.954 / 0.987 / 0.868 | 0.119 | 0.73 | 45.1 | 59% | 123 |
| random | 0.360 / 0.307 / 0.407 | 1.002 / 0.836 / 0.796 | 0.127 | 0.66 | 36.3 | 77% | 69 |
| for reward | 0.362 / 0.284 / 0.262 | 0.871 / 0.745 / 0.712 | 0.107 | 0.71 | 147.2 | 66% | 103 |
| for information (`InfoController`) | 0.323 / 0.358 / 0.194 | **0.482 / 0.405 / 0.336** | 0.107 | 0.70 | 71.4 | 75% | 75 |

- **All four claims hold.** Squared error in `k`, information − reward planning:
  −0.379 ± 0.092; − fixed default: −0.621 ± 0.101. `f`: −0.029 ± 0.012 against reward
  planning, so `f` is measured no worse (better, beyond 2 SE). P(start) −0.010
  [−0.033, +0.014].
- **Choosing for information measures `k` better at 100 ticks than any other way does at
  300.** The median |`k` error| at 300 ticks is 0.067, against 0.29 to 0.57 for the others,
  and among people with at least 100 decisions in the fit (39 to 70 per arm) `k` RMSE is
  0.098 against 0.39 to 0.78. γ is measured the same under every arm.
- **It hides feedback and redoes misses at the end of the set.** Only 13% of its start
  decisions followed a set with feedback shown (reward planning 90%), and it gave 160
  end-of-set redo decisions per person (5 to 6 under the other two). It aimed at a 36%
  chance of a right answer (reward planning 56%) and chose 6.0 questions per set on average
  (reward planning 7.4).
- **The cost is in time away, not in starting** (descriptive, 95% t intervals over people).
  Against reward planning, people were away 9.2 points more of the 300 ticks [+5.4, +13.0],
  made 28 fewer start decisions [16, 39] and earned 76 less reward [55, 97], although each
  start decision went the same way. Against the fixed default they started slightly less
  often (−0.028 [−0.055, −0.001]) and were away 16 points more; against random choice they
  started more often (+0.040 [+0.012, +0.069]).

### Planning with fitted γ and bound (2026-10-28)

Every comparison above gave the simulated people γ 0.9 and `BOUND` 1.5, the values
`InfoController` and its fit assume. `FullInfoController` (`experiments/controller_full.py`)
plans without that assumption: it fits all five parameters, rolls each posterior draw out
with its own γ and bound, and scores settings by the information about all five.

<details>
<summary>How <code>FullInfoController</code> works</summary>

- **Estimate.** It fits `f`, `k`, γ, `BOUND` and `T0` together by MAP with the closed-form
  likelihood. Value at any γ comes from shadow TD learners replaying the person's rewards
  (as in the five-parameter recovery). The prior is independent normals, `f` and `k`
  0.8 ± 0.5, γ 0.85 ± 0.1, `BOUND` 1.5 ± 0.3, `T0` 0.2 ± 0.15 s, and the posterior is
  Laplace.
- **Rollouts.** Each of its 6 posterior draws is rolled out with its own γ (the nearest
  shadow learner's value weights) and its own bound. Rollout decisions are drawn from the
  closed-form likelihood, so a new bound costs no Fokker-Planck solve.
- **Score.** The rise in log det of the Fisher information about all five parameters. A
  decision's information is Jᵀ I(μ, BOUND, T0) J, with J's rows (v, −c, f·∂v/∂γ, 0, 0),
  (0, 0, 0, 1, 0) and (0, 0, 0, 0, 1), and I(μ, BOUND, T0) from the closed-form likelihood by
  central differences.
- The candidates, the rollout length, the safety floor and the schedule are
  `InfoController`'s.

</details>

*`python -m experiments.controller_full`*

128 people from a seeded Latin hypercube over the ranges of the five-parameter recovery (a
different seed). Each plays the app's task for 300 ticks twice on the same seed, once under
each controller. α 0.05 for everyone; cold-start cost 0.6; `ACHIEVEMENT_SCALE = 4`,
continuing bootstrap, `T_DUR = 7.0`. The data each arm recorded by 100, 200 and 300 ticks
are fitted the same way, all five parameters by maximum likelihood. The number of people
was raised from 48 to 128 after a timing test of one person and before any result was read.

Claims fixed before the run, paired at 300 ticks:

- **F1**: squared error in `k`, full − current, is below zero beyond 2 SE;
- **F2**: the same for `f`; **F3**: the same for γ;
- **F4**: P(start), full − current, is not below zero beyond 2 SE.

| RMSE at 100 / 200 / 300 ticks | app's controller | all five parameters |
|---|---|---|
| `f` | 0.402 / 0.213 / 0.214 | 0.312 / 0.201 / 0.202 |
| `k` | 0.468 / 0.291 / 0.259 | 0.353 / 0.341 / 0.299 |
| γ | 0.142 / 0.120 / 0.115 | 0.125 / 0.112 / 0.118 |
| `BOUND` | 0.230 / 0.160 / 0.154 | 0.199 / 0.157 / 0.131 |
| `T0` (s) | 0.123 / 0.107 / 0.111 | 0.128 / 0.129 / 0.109 |

- **F1, F2 and F3 do not hold.** At 300 ticks, squared error full − current was
  +0.022 ± 0.020 for `k`, −0.005 ± 0.017 for `f` and +0.002 ± 0.003 for γ. The median
  absolute errors are nearly equal (app's controller / full: `k` 0.066 / 0.069, `f`
  0.074 / 0.084), and the paired median difference in squared error is 0 for every
  parameter. At 100 ticks the full controller was ahead for `f` (−0.072 ± 0.035) and `k`
  (−0.097 ± 0.044), but no claim was fixed there, and the lead is gone by 200.
- **F4 holds, and the full controller kept people engaged better** (descriptive): P(start)
  +0.015 ± 0.013, total reward +11.6 ± 3.3 (77.0 against 65.3), share of ticks away
  −4.3 ± 1.3 points, and +13.0 start decisions [+5.5, +20.5]. It aimed easier: a mean
  target of 43% right against 37%.
- **More data, no better estimates.** The full controller's people gave more decisions (193
  against 144 at the median by 300 ticks), yet its estimates were no better. Two likely
  reasons, both untested. Its score spreads the information it seeks over γ, `BOUND` and
  `T0` as well as `f` and `k`. And easier targets make the redo cost, `0.15 · min(1/p, 8)`,
  vary less.
- **It did not choose feedback by γ.** The controllers showed feedback before 99% (app's)
  and 98% (full) of start decisions, because the redo prompts they favour (97 and 95% of
  updates) are offered only with feedback on. The γ dependence of feedback ("Feedback by
  γ") never came into play.
- **The RMSEs here are far above the recovery's**, because 300 ticks give these people 105 to
  118 start decisions on average (they are away 61 to 65% of the time), not 300. As under
  "Measuring costs no detectable reward", a few people with very few decisions dominate.

### Feedback in blocks, with redos mid-set or at the end of the set (2026-10-28)

Both information controllers keep feedback on: redo prompts are what measure `k`, and with
feedback hidden a miss is never shown, so no redo is offered. Hiding feedback is also what
measures γ ("Feedback switched in blocks"). Two changes let a controller have both.
- **Blocks.** `BlockInfoController` (`experiments/feedback_blocks.py`) is `InfoController`
  with feedback on a schedule: shown for one plan's 25 start decisions, hidden for the
  next, alternating. Within a block it chooses among the settings with that block's
  feedback.
- **Redos at the end of the set** (`Simulation(end_redos=True)`, `controller.REDO_END`).
  With feedback hidden, the misses are shown with the score on the end-of-set screen, and
  each can be redone there. It is the same redo node at the set's last slot: value is the
  undiscounted slot share 1/4 plus the error the score just produced (the whole set's),
  and cost is `0.15 · min(1/p, 8)` as always. A redo taken there is an extra question: it
  can be given up on, a right answer is a correction and adds to total reward, it lands no
  reward on the value learner (the set's reward has been paid), and it adds to the next
  start's review load, as a redo does in the app. Off by default; no seeded output changes
  (`tests/test_end_redos.py`).

**Design** (`python -m experiments.feedback_blocks`). 128 people from a seeded Latin
hypercube over the five-parameter recovery's ranges, each playing the app's task for 300
ticks under three controllers, paired on seed:
- **current**: `InfoController` as the app ran it until 2026-10-29;
- **blocks**: `BlockInfoController` over the same 96 settings, so a hidden block's redo
  prompts ask nothing;
- **blocks_end**: the same, with the six hidden redo-prompt settings replaced by
  end-of-set ones.

The simulated person sees the score at the end of a hidden set, as a person in the app has
since 2026-10-28. α 0.05; `ACHIEVEMENT_SCALE = 4`, continuing bootstrap, `T_DUR = 7.0`. All five
parameters are fitted by maximum likelihood at 100, 200 and 300 ticks. Claims fixed before
the run, paired at 300 ticks:
- **E1**: squared error in `k`, blocks_end − blocks, below zero beyond 2 SE;
- **E2**: squared error in γ, blocks_end − current, below zero beyond 2 SE;
- **E3**: squared error in `k`, blocks_end − current, not above zero beyond 2 SE;
- **E4**: P(start), blocks_end − current, not below zero beyond 2 SE.

| RMSE at 100 / 200 / 300 ticks | current | blocks | blocks_end |
|---|---|---|---|
| `f` | 0.404 / 0.223 / 0.219 | 0.378 / 0.227 / 0.230 | 0.371 / 0.216 / 0.224 |
| `k` | 0.378 / 0.292 / 0.265 | 0.377 / 0.229 / 0.246 | 0.359 / 0.210 / 0.229 |
| γ | 0.116 / 0.115 / 0.097 | 0.121 / 0.094 / 0.104 | 0.117 / 0.092 / 0.102 |
| `BOUND` | 0.216 / 0.174 / 0.154 | 0.209 / 0.140 / 0.132 | 0.214 / 0.135 / 0.133 |
| `T0` (s) | 0.131 / 0.113 / 0.109 | 0.120 / 0.109 / 0.119 | 0.124 / 0.107 / 0.118 |

- **E1 does not hold, narrowly.** End-of-set redos brought the redo decisions back (151
  per person against 77 with blocks alone and 153 under current; 81 of the 151 at the end
  of the set), and squared error in `k` fell by 0.0080 ± 0.0044 (1.8 SE).
- **E2 does not hold.** γ was measured no better than under current (+0.0014 ± 0.0023).
  The schedule barely ran: in 300 ticks the median person made 54 to 62 start decisions, so
  about half never reached a second hidden block. The blocks experiment that sharpened γ
  had 300 start decisions and six hidden blocks per person.
- **E3 and E4 hold.** `k` was no worse than under current (−0.018 ± 0.024), and P(start)
  no lower (+0.011 ± 0.012).
- **Descriptive.** End-of-set redos cost some time in the task: 7.4 ± 4.2 fewer start
  decisions than current and 2.5 ± 1.4 points more time away. Blocks alone did not:
  +1.0 ± 4.0 start decisions, and +10.2 ± 3.1 reward. The correlation of value and cost
  stayed far from proportional (−0.43 to −0.51 over all decisions). End-of-set redos'
  value varies more than mid-set ones' (sd 0.40 against 0.07 in a check at a 45% target,
  where the carried error is the whole set's) and their cost a little less.

### A countdown on the redo prompt (2026-10-28)

Under time pressure people lower their response threshold. They wait for less evidence, so
they decide sooner and less consistently, and the striatum and pre-SMA carry the adjustment
(Forstmann et al., 2008). `temporal/model/pressure.py` adds that to the redo prompt:
- **The countdown.** A pressured prompt shows a countdown of 3 s (`REDO_DEADLINE`). Still
  open when it ends, it closes and the question is not redone.
- **The person's response.** Their bound on it is λ × `BOUND`, with λ
  (`UserParams.pressure`, 1 keeps the bound) a new parameter of the person: how much
  caution they give up when hurried.
- **What stays the same.** Their drift, `f · value − k · cost`, is unchanged: time pressure
  changes how much evidence they wait for, not what it says. So countdown decisions observe
  the same `f` and `k`, as in a speed-accuracy design.
- **Draws and the fit.** Draws come from the closed-form likelihood. A prompt that timed
  out is a censored decision in the fit, read at its own 3 s deadline
  (`fastlik.neg_log_likelihood` takes a bound and a deadline per decision). Off by
  default; no seeded output changes (`tests/test_pressure.py`).

**Design** (`python -m experiments.redo_pressure`). 64 people from a seeded Latin hypercube
over the five-parameter recovery's ranges and λ from 0.4 to 1.0. Each plays the app's task
under `InfoController` for 300 start decisions plus their redo decisions, twice on one
seed: with no countdown, and with the countdown on every redo prompt in alternate blocks
of 25 start decisions (the controller is not told). With no countdown, `f`, `k`, γ,
`BOUND` and `T0` are fitted; with it, λ as well. For comparison, the countdown data are
also fitted with λ held at 1: the countdown's deadline known, the person's response to it
not. Claims fixed before the run:
- **T1**: λ recovered with r ≥ 0.8 and RMSE ≤ 0.10;
- **T2**: squared error in `k`, countdown − none, not above zero beyond 2 SE;
- **T3**: squared error in `BOUND`, countdown − none, below zero beyond 2 SE.

| | RMSE, no countdown | RMSE, countdown in blocks | squared error, countdown − none | bias with λ ignored |
|---|---|---|---|---|
| `f` | 0.069 | 0.078 | +0.0014 ± 0.0015 | −8.4 ± 2.3% |
| `k` | 0.059 | 0.063 | +0.0005 ± 0.0012 | +1.8 ± 1.3% |
| γ | 0.030 | 0.024 | −0.0003 ± 0.0003 | +0.7 ± 0.5% |
| `BOUND` | 0.037 | 0.042 | +0.0003 ± 0.0004 | −8.6 ± 0.9% |
| `T0` (s) | 0.037 | 0.027 | **−0.0006 ± 0.0002** | −41.8 ± 5.5% |

- **T1 holds: how a person responds to time pressure can be measured.** λ was recovered
  with RMSE 0.036, bias +0.002 ± 0.005 and r 0.978. The largest error, 0.70 fitted as
  0.90, is a person who met one countdown prompt in the whole run; without them the RMSE is
  0.026. The median person met 228.
- **T2 holds**: the countdown cost `k` nothing detectable.
- **T3 does not hold**: the bound was measured no better (+0.0003 ± 0.0004). `T0` was
  measured better, beyond 2 SE (−0.0006 ± 0.0002), with no claim fixed for it.
- **Ignoring the person's response biases the fit.** With λ held at 1, `f` came out 8% low,
  the bound 9% low and `T0` 42% low, and −log L was worse by 41.7 at the median. A countdown
  in the app would need λ in the fit.
- **People who do not lower their bound lose their redos.** By true λ, the share of
  countdown prompts that timed out was 3% (λ 0.4 to 0.55), 13%, 23% and 37% (λ 0.85 to 1.0).
  The share that ended in a redo was 27%, 23%, 12% and 12%. A cautious person under a
  countdown corrects fewer misses, which in the simulator is how ability grows.

### Every parameter, every setting (2026-10-29)

*`python -m experiments.controller_all`*

`FullInfoController` ("Planning with fitted γ and bound") showed feedback before 98% of
start decisions. The redo prompts it favours need feedback on, and it had no countdown to
choose. `AllInfoController` (`experiments/controller_all.py`) plans with all six fitted
parameters and chooses feedback, where redos are offered and the countdown by the same
score, with nothing on a schedule.

<details>
<summary>How <code>AllInfoController</code> works</summary>

- **Parameters.** `f`, `k`, γ, `BOUND`, `T0` and λ, fitted together by MAP with the
  closed-form likelihood, drawn from the Laplace posterior for rollouts, and all six scored.
  λ's prior is 0.7 ± 0.2; the other five keep `FullInfoController`'s.
- **132 settings.** `InfoController`'s 84 scheduled-review settings, and redo prompts at
  each of the 6 targets, either mid-set with feedback shown or at the end of the set with
  it hidden. Each redo setting is untimed or has a 2, 3 or 5 s countdown
  (`Environment.countdown`).
- **Score.** The rise in log det of the information about all six parameters (D-optimal).
  A decision under a countdown is read at bound λ · `BOUND` and its own deadline, so it
  informs λ.
- The draws, the 25-tick rollouts, the safety floor and the schedule are
  `FullInfoController`'s. Unlike `AppController` ("The app's controller") it has no set
  length, score display or "repeats", and an end-of-set redo's outcome is not shown.

</details>

Setup:
- The 128 people and seeds of "Planning with fitted γ and bound", each also given a λ drawn
  uniformly from 0.4 to 1.0. Each plays the app's task for 300 ticks under
  `AllInfoController`.
- The comparison arms are that result's saved runs of the same people on the same seeds,
  under `InfoController` and `FullInfoController`. Neither offers a countdown, so λ plays
  no part in them. Two of those runs were re-run with today's code and matched.
- This arm's data are fitted at 100, 200 and 300 ticks, all six parameters by maximum
  likelihood.
- The simulated person sees an end-of-set score when feedback is hidden.
- α 0.05; `ACHIEVEMENT_SCALE = 4`, continuing bootstrap, `T_DUR = 7.0`.

Claims fixed before the run, paired at 300 ticks against `InfoController`:
- **A1**: squared error in γ is below zero beyond 2 SE;
- **A2**: squared error in `k` is not above zero beyond 2 SE;
- **A3**: P(start) is not below zero beyond 2 SE.

A second arm, Ds, was added after A1 to A3 were read. `DsInfoController` is
`AllInfoController` scored on `f`, `k` and γ only (Ds-optimal): it still fits, draws and
rolls out all six parameters, but scores a setting by the rise in log det of the Schur
complement of the information about `(f, k, γ)`, so measuring `BOUND`, `T0` or λ earns
nothing for its own sake. Same people, seeds and fits. Claims fixed before its run, paired
at 300 ticks:
- **D1**: squared error in `k`, Ds − all six, is below zero beyond 2 SE;
- **D2**: squared error in γ, Ds − `InfoController`, is below zero beyond 2 SE;
- **D3**: P(start), Ds − `InfoController`, is not below zero beyond 2 SE.

| RMSE at 100 / 200 / 300 ticks | `InfoController` | `FullInfoController` | `AllInfoController` | `DsInfoController` |
|---|---|---|---|---|
| `f` | 0.402 / 0.213 / 0.214 | 0.312 / 0.201 / 0.202 | 0.342 / 0.240 / 0.183 | 0.233 / 0.188 / 0.172 |
| `k` | 0.468 / 0.291 / 0.259 | 0.353 / 0.341 / 0.299 | 0.525 / 0.396 / 0.308 | 0.450 / 0.368 / 0.293 |
| γ | 0.142 / 0.120 / 0.115 | 0.125 / 0.112 / 0.118 | 0.119 / 0.107 / 0.095 | 0.133 / 0.113 / 0.099 |
| `BOUND` | 0.230 / 0.160 / 0.154 | 0.199 / 0.157 / 0.131 | 0.179 / 0.205 / 0.201 | 0.192 / 0.156 / 0.123 |
| `T0` (s) | 0.123 / 0.107 / 0.111 | 0.128 / 0.129 / 0.109 | 0.102 / 0.090 / 0.081 | 0.135 / 0.109 / 0.099 |
| λ | | | 0.123 / 0.114 / 0.085 | 0.377 / 0.328 / 0.294 |

- **A1 does not hold.** γ's RMSE fell from 0.115 to 0.095, but the paired difference in
  squared error, −0.0034 ± 0.0022, is 1.5 SE.
- **A2 and A3 hold**, A2 only by its margin: squared error in `k` was +0.0315 ± 0.0235
  higher, and P(start) changed by −0.007 ± 0.015.
- **It hides feedback.** Feedback was shown before 14% of start decisions, against 99%
  under `InfoController` and 98% under `FullInfoController`. It hid it more for people
  with a high γ: shown before 24% of starts at γ 0.6 to 0.7, and 5% at 0.9 to 0.97. Of a
  person's 171 redo decisions, 165 were at the end of the set and 115 under a countdown
  (76 at 5 s, 27 at 3 s, 12 at 2 s).
- **λ is measured** from those: r 0.90 and RMSE 0.085 over the 123 people with a fit (the
  true λ has SD 0.18), and r 0.95 and RMSE 0.059 for the 108 with 20 or more timed redos.
- **Not every parameter gains.** Against `InfoController`, `T0` was measured better
  (−0.0046 ± 0.0024, 1.9 SE) and `f` a little (−0.012 ± 0.013). `k` (above) and `BOUND`
  (+0.018 ± 0.014) were worse, neither beyond 2 SE. Two likely reasons, both untested. The
  score spreads its information over six parameters, so `k` gets a smaller share. And a
  timed redo measures λ · `BOUND`, so λ and the bound trade off.
- **Engagement held against `InfoController`, not against `FullInfoController`.** Against
  `InfoController`: time away +1.2 ± 1.0 points, start decisions −3.5 ± 2.9, total reward
  +17.8 ± 3.4. Against `FullInfoController`: 16.5 ± 3.4 fewer start decisions and 5.5 ± 1.1
  points more time away.

The Ds arm:
- **D1 and D2 do not hold; D3 holds.** Against all six, `k` changed by −0.011 ± 0.018. Against
  `InfoController`, γ by −0.0036 ± 0.0022 (1.6 SE) and P(start) by −0.010 ± 0.014.
- **So spreading the score over six parameters is not what cost `k`.** `k`'s RMSE was 0.293,
  still above `InfoController`'s 0.259, though no `k` difference between any two arms here
  reaches 2 SE. Every controller that plans with fitted γ and bound measured `k` a little
  worse than `InfoController`, with feedback shown (`FullInfoController`) or hidden.
- **It dropped most countdowns** (17 timed redos per person, against 115 under all six), so
  λ was measured only where it had them: RMSE 0.044 and r 0.97 for the 47 people with 20
  or more, 0.294 over all 121 fitted. `BOUND` was measured better than under all six
  (−0.017 ± 0.011) and is the best of the four arms (0.123).
- **It still hides feedback**, before 84% of start decisions (shown before 22% at γ 0.6 to
  0.7, 10% at 0.9 to 0.97), with 153 of 159 redos at the end of the set.
- **It cost some engagement.** Against `InfoController`: time away +3.2 ± 1.4 points and
  9.5 ± 4.2 fewer start decisions (both 2.3 SE, no claim fixed), total reward +5.3 ± 3.8.
  Against all six: total reward −12.5 ± 3.2.

---

## Methodological notes

### Censored trials are data

PyDDM's likelihood is not renormalized over the decided region. If only the trials that
reached a bound are fitted, the optimizer can inflate the drift at no likelihood cost, by
shrinking the predicted share of undecided trials. The fix is to feed the censored counts
back in as undecided trials.

On the recovery design (48 fits of 1200 decisions, 4×4 grid × 3 reps, four learners across
the learning curve, `T_DUR = 7.0`, continuing bootstrap; 2026-10-25,
`python -m experiments.recovery_sweep`), 9,357 of 57,600 decisions (16.2%) timed out. Kept as
undecided, the mean bias was +1.0% for `f` and +1.5% for `k` (RMSE 0.049, 0.095). Dropped,
it was +19.1% and +10.0% (RMSE 0.122, 0.326).

### Every model here carries a 2% lapse mixture

pyddm's `gddm` mixes a lapse into every model by default (`mixture_coef = 0.02`): 2% of
trials are drawn from a uniform distribution over the whole deadline instead of from the
accumulator. It is left at its default, and it matters in three ways.

- **Simulation and fitting stay consistent.** The same mixture is in the draws the simulator
  takes and in the likelihood the fit maximizes, so recovery is unaffected.
- **It explains RTs faster than the non-decision time.** A model with `T0 = 0.3` still
  produces a few decisions under 0.3 s; they come from the mixture, not the accumulator.
- **It floors the likelihood.** No observed RT has zero density, so a single odd human
  decision cannot make a parameter set impossible. The explicit lapse term for giving up on
  questions (`persistence.LAPSE`) does the same job; it had to be written by hand because
  that likelihood is ours rather than pyddm's.

### The deadline must sit past human decision times

A decision that reaches no bound by `T_DUR` is undecided, and so is a person's click that
comes later than `T_DUR`: the fit keeps that trial as "no decision" and loses the choice.
Slow choices to start the next set are the hesitations a model of reluctance is meant to
capture, so the deadline has to sit past the decision times people actually take. `T_DUR`
was 3.0 s until 2026-09-13 and is 7.0 s now. `python -m temporal audit` checks a record
against it and flags more than 2% of decisions past it. What the longer deadline did to
simulated people is a V2 result in `RETIRED.md`.

### The lockout state is a model prediction

At high re-entry cost a learner can stay out of the task almost all the time. Each stop
ends a session, and coming back is decided at the cold-start cost, so for an effort-averse
person most ticks are spent away (90% and more for f 0.6, k 1.2 at an easy or a hard
target; "Which difficulty and whether to show feedback"). The cycle: rarely engages →
rarely improves → `V` stays low → the drift to come back stays negative → still rarely
engages. None of the time away is recorded, so in data the cycle shows as few sessions, not
as many refusals. This matches the avoidance-maintained deficit cycle that behavioral
activation therapy targets in depression. Review mode decides what an escape is worth:
under `quiet` an escapee learns nothing, under `choice` a high-`k` escapee declines the
redos and learns little, and under `scheduled` escapes compound.

### The closed-form likelihood (`temporal/fastlik.py`)

The accumulator starts midway between bounds at ±`BOUND`, drifts at a constant `μ` with
noise `SIGMA`, and stops at `T_DUR`. Dividing by the noise gives unit noise, drift
`v = μ / SIGMA`, bound separation `a = 2 · BOUND / SIGMA` and relative start `w = 1/2`.

<details>
<summary>The formulas</summary>

- **Density at the lower bound at time `t`** (Navarro & Fuss, 2009, small-time series), with
  `u = t / a²`:
  `p_lower(t) = (1/a²) · exp(−v·a·w − v²·t/2) · (2π·u³)^(−1/2) · Σ_{k=−7..7} (w + 2k) · exp(−(w + 2k)² / (2u))`.
  At the upper bound (GO) it is the same with `−v`, since the start is midway. Over the
  deadline `u` stays under about 1, where 15 terms are far more than enough.
- **No bound reached by time `T`.** From the eigenfunction expansion of the diffusion
  absorbed at both bounds, with the start midway:
  `S(T) = (2/a) · 2·cosh(v·a/2) · exp(−v²·T/2) · Σ_{k=1,3,5..39} sin(kπ/2) · exp(−k²π²·T/(2a²)) · (kπ/a) / (v² + (kπ/a)²)`,
  computed in logs so a large drift cannot overflow.
- **What pyddm adds, added the same way.** `T0` shifts every passage. The lapse share
  `m = LAPSE_MIX = 0.02` is split evenly between the choices, so a decided trial at `t` has
  density `(1 − m) · p(t − T0) + (m/2) / T_DUR` for its choice, and an undecided trial has
  probability `(1 − m) · S(T_DUR − T0)`. pyddm shifts by whole time steps; `fastlik` shifts
  exactly. Its densities are per second and pyddm's per time bin, which changes −log L by a
  constant and no estimate.

</details>

- **The check** (`tests/test_fastlik.py`): against pyddm's Fokker-Planck solution at bounds
  1.0, 1.5 and 2.4 with `T0` 0, 0.23 and 0.1 s, at five drifts each from −2.5 to 3,
  densities agree within 0.2% of their peak and undecided probabilities within 0.001.

### Recovery demonstrates identifiability, not accuracy

The generating model and the fitted model are identical here, so successful recovery is
expected. Real data is never generated by the fitted model. The two studies below test
what happens when the person and the model differ.

#### Non-decision time

*2026-10-25, seeded · `python -m temporal recovery --mode t0 --reps 3`*

One learner per run, 1200 recorded decisions (start decisions only, all warm), 3 reps per
value, true `f = k = 0.8`. Each dataset is fitted twice, with `T0` pinned at 0 and with
`T0` free. Means over reps:

| true T0 | f pinned | k pinned | f free | k free | T0 recovered | free − pinned, f / k |
|---|---|---|---|---|---|---|
| 0.0 s | 0.84 | 0.86 | 0.86 | 0.88 | 0.05 | +0.02 / +0.02 |
| 0.2 s | 0.76 | 0.74 | 0.83 | 0.80 | 0.18 | +0.07 / +0.06 |
| 0.3 s | 0.71 | 0.72 | 0.81 | 0.81 | 0.29 | +0.10 / +0.09 |
| 0.5 s | 0.61 | 0.65 | 0.74 | 0.78 | 0.49 | +0.14 / +0.13 |

- **The paired difference is the measurement here.** Start decisions alone measure `k` to an
  SD of about 0.2 per 1200-decision dataset (the precision table, scaled), so a mean of 3
  reps can sit 0.1 from the truth by chance, as it does at a true 0 (`k` 0.86 and 0.88). The
  two fits share each dataset, and the difference between them has an SD of 0.02 or less
  across reps.
- **Pinning `T0` biases both parameters down, more as the true value grows.** Against the
  free fit on the same data, pinning takes 0.06 to 0.07 off `f` and `k` at 0.2 s, 0.09 to
  0.10 at 0.3 s, and 0.13 to 0.14 at 0.5 s.
- **Freeing it removes most of the bias.** `T0` itself lands within 0.05 of the truth at
  every value, and the free fits sit within 0.08 of the truth, inside the sampling spread
  above.
- **It costs almost nothing when there is none.** With a true 0, the free fit returns 0.05
  and moves `f` and `k` by 0.02.
- **It is free where real people are fitted**: the controller's estimate and the interface's
  "fit my parameters". The recovery pipelines pin it, since their simulated people have
  none.

#### The person may not share the model's constants

*2026-10-25 · `python -m experiments.misspecification`*

The fit holds the discount factor γ at 0.9, the TD learning rate α at 0.05, `SIGMA` at 0.7
and `BOUND` at 1.5. Here each simulated person differs from the model in one of the four,
and the fit isn't told.

- **γ and α never enter the fit directly.** They set the person's learned value, and the app
  computes value with the defaults. To match that, a second TD learner at the defaults runs
  on the same rewards as the person, and the *analyst* fit uses its value; this is what real
  data would get. The *oracle* fit uses the person's own value instead, so the difference
  between the two is the cost of a wrong γ or α.
- **`SIGMA` and `BOUND` change only the person's own start, redo and give-up draws.** The
  controller's rollouts and every fit keep the pinned values.
- **Setup**: the app's (question types with Elo ratings, skips on, `InfoController` steering
  from the default setting), 300 start decisions plus their redo decisions per run. People
  (f, k) = (0.8, 0.8), (1.2, 0.4) and (0.6, 1.2), 5 seeds each, with the same seeds at every
  level.

Bias of the analyst fit, (estimate − true) / true, mean ± SE of 15 fits:

| person | bias in f | bias in k | bias in f / k | undecided |
|---|---|---|---|---|
| all defaults | +1 ± 1% | −1 ± 1% | +2 ± 1% | 6.2% |
| γ = 0.5 | −73 ± 1% | +29 ± 4% | −79 ± 1% | 8.6% |
| γ = 0.7 | −62 ± 1% | +16 ± 2% | −67 ± 1% | 8.3% |
| γ = 0.8 | −47 ± 1% | +3 ± 1% | −49 ± 1% | 8.2% |
| γ = 0.95 | +58 ± 1% | −4 ± 1% | +66 ± 2% | 6.7% |
| γ = 0.99 | +161 ± 2% | −7 ± 4% | +190 ± 13% | 6.6% |
| α = 0.01 | −34 ± 1% | −26 ± 4% | −5 ± 8% | 9.0% |
| α = 0.025 | −14 ± 1% | −10 ± 1% | −3 ± 2% | 7.5% |
| α = 0.1 | +6 ± 1% | +4 ± 1% | +2 ± 2% | 5.7% |
| α = 0.2 | +9 ± 1% | +10 ± 2% | −0 ± 2% | 5.7% |
| `SIGMA` = 0.5 | −2 ± 0% | −3 ± 1% | +1 ± 1% | 10.0% |
| `SIGMA` = 0.6 | −1 ± 1% | −1 ± 1% | +0 ± 1% | 9.0% |
| `SIGMA` = 0.85 | +1 ± 1% | +3 ± 2% | −2 ± 2% | 3.1% |
| `SIGMA` = 1.0 | +5 ± 1% | +6 ± 2% | −0 ± 2% | 1.3% |
| `BOUND` = 1.0 | +57 ± 2% | +71 ± 5% | −8 ± 2% | 0.5% |
| `BOUND` = 1.25 | +21 ± 1% | +22 ± 2% | −1 ± 2% | 3.0% |
| `BOUND` = 1.75 | −15 ± 0% | −15 ± 1% | −0 ± 1% | 10.2% |
| `BOUND` = 2.0 | −27 ± 0% | −28 ± 1% | +2 ± 1% | 14.6% |

**The accumulator's constants keep `f / k` close.** `BOUND` and `SIGMA` are constants of the
accumulator, and a wrong value for either changes `f` and `k` by about the same proportion.
A bound of 1.0 instead of 1.5 makes `f` come out 57% and `k` 71% too high, and `f / k` 8%
too low; from 1.25 to 2.0 the ratio stays within 2% of the truth. The share of undecided
decisions also changes with the bound, from 0.5% at 1.0 to 14.6% at 2.0, so that share can
check the pinned bound against real data. `SIGMA` has little effect: from 0.5 to 1.0, `f`
and `k` stay within 6% and their ratio within 2%, although the share of undecided decisions
falls from 10.0% to 1.3%.

**γ, a constant of the learner, has the largest effect.** It changes the value that goes
into the drift, and the ratio does not stay close. `f / k` comes out 49% too low at γ = 0.8
and 66% too high at 0.95. The oracle fit is within 2% at every γ, so the whole error comes
from computing value with γ = 0.9. A person who discounts more steeply than 0.9 looks less
sensitive to reward and more averse to effort than they are; a person who discounts less
looks much more sensitive to reward. This puts a size on the confound of `f` with the
discount factor ("Known limitations"). `k` is least precise at low γ (+29 ± 4% at 0.5),
where such a person starts the fewest sets (31% of start decisions, against 49% at the
defaults).

**A wrong learning rate moves `f` and `k` together**: −34% and −26% at α = 0.01, +9% and
+10% at 0.2. The ratio stays within 5% at every rate (−5 ± 8% at 0.01).

**On the single-`diff` task** (the same command; one learner, 1200 decisions, no redo
decisions), two constants push `k` toward 0. One is a high γ: at 0.95 `k` comes out
57 ± 8% too low and is fitted as 0 in some runs, and at 0.99 it is fitted as 0 in every run
while `f` comes out 242 ± 20% too high. The other is a slow learning rate: at α = 0.01 `k`
comes out 51 ± 9% too low, fitted as 0 in some runs. A low γ biases `f` about as on the app's
task (−74 ± 1% at 0.5) and `k` less (+17 ± 3%). For the ratio, `BOUND` and `SIGMA` keep
`f / k` within 8%, γ moves it (−52 ± 2% at 0.8), and α moves it by 9 to 35% wherever `k`
stays above 0. The likely reason `k` holds up on the app's task is the redo decisions,
whose cost varies more than at any other decision. All of these fits used γ = 0.9,
`ACHIEVEMENT_SCALE = 4`, the continuing bootstrap and `T_DUR = 7.0`.

### What can be fixed, and what fixing costs

- **`SIGMA` has to be fixed.** Drift, bound and noise scaled by one factor give the same
  choices and RTs, so one of them sets the unit. As in the standard DDM, it is the noise.
- **`BOUND` does not have to be.** With the noise as the unit, the bound has its own
  signature: it widens the RT distributions and makes choices more consistent without
  changing which way they lean. It is pinned at 1.5 in every fit except where it is fitted
  per person, and there it is recovered to RMSE 0.045 (r 0.988) with `f`, `k`, γ and `T0`
  free ("The bound can be fitted"). Pinning it at the wrong value costs `f` and `k`
  together: a person whose bound is 1.0 gets `f` 57% and `k` 71% too high, though their
  ratio stays within 8% ("The person may not share the model's constants").
- **`T_DUR` is not a parameter.** It is where the analysis stops reading a decision; slow
  deciders are described by the bound and `T0`.
- **γ can be fitted** ("Fitting the discount factor"), and α can be profiled, loosely
  ("Measuring the learning rate and the reward").

### How value and cost together decide what can be measured

Drift is `f · value − k · cost`, so `f` is read from how decisions change with value, and
`k` from how they change with cost, or from cost's constant offset on the drift.

- **The only case in which `f` and `k` cannot be told apart** is cost exactly proportional to
  value (`cost = b · value`). Drift is then `(f − b·k) · value`, and only `f − b·k` is
  measured. No decision in this task is like that. Constant cost still identifies `k`
  through the offset, and cost correlated with value (`cost = a + b · value`) identifies
  both.
- **Correlation, or too little variation in cost, costs precision instead**: more decisions
  for the same accuracy.

*2026-10-25 · `python -m experiments.precision`*

The table gives the expected standard deviation of each estimate from 300 start decisions,
from the Fisher information of the decisions at the true `f`, `k`. 16 people × 2 seeds,
each run until 300 start decisions are recorded, at the default setting for starts and a
fixed redo-prompt setting for starts with redos; medians:

| decisions in the data | SD of f | SD of k | sd of cost |
|---|---|---|---|
| start decisions | 0.114 | **0.436** | 0.011 |
| starts and redos | 0.025 | **0.106** | 0.053 |

Every recorded start decision is warm, for a person and, since V4, in simulation. So start
decisions alone barely measure `k`: their cost varies only with the last set's review load.
Redo decisions, whose cost varies with how hard the missed question is, cut the SD of `k`
about fourfold. That is why `InfoController` leans on redo prompts ("Planning to measure").
Before V4 the simulator also recorded cold starts, and the same table put start decisions
alone at an SD of 0.140 for `k`; that row described data no person gives (`RETIRED.md`).

**With γ, the bound and `T0` free as well** (2026-10-28,
`python -m experiments.precision_full`). The same quantity, computed from the 100 saved
histories of the five-parameter recovery ("The bound can be fitted") with the Fisher
information at the truth from the closed-form likelihood. Each person has 300 start
decisions plus their redo decisions (434 median, 89 to 578), settings chosen by
`InfoController`, and γ, `BOUND` and `T0` vary across people, so this is not the design of
the table above. Medians, with the 5th to 95th percentiles in brackets, over the 100
people; no claim was fixed in advance:

| decisions in the data | free | SD of f | SD of k | SD of γ |
|---|---|---|---|---|
| start decisions | `f`, `k` | 0.055 [0.037, 0.078] | 0.285 [0.191, 0.576] | |
| start decisions | all five | 0.071 [0.044, 0.099] | **0.610** [0.438, 0.863] | 0.038 [0.004, 0.150] |
| starts and redos | `f`, `k` | 0.035 [0.018, 0.060] | 0.038 [0.021, 0.076] | |
| starts and redos | all five | 0.059 [0.041, 0.086] | **0.048** [0.024, 0.100] | 0.015 [0.004, 0.069] |

- With all five parameters free, start decisions alone hardly measure `k`. Freeing γ, the
  bound and `T0` doubles the SD of `k` from starts alone (×2.16, median), but raises it only
  18% once redos are in the data.
- With five parameters free, redos cut the SD of `k` about thirteenfold (0.610 to 0.048),
  not fourfold.
- The five-free SDs with redos agree with what the recovery measured. Their root mean square
  over the 100 people is 0.063 for `f`, 0.058 for `k` and 0.034 for γ, against RMSEs of
  0.062, 0.054 and 0.028.
- γ's SD is largest for people with a low γ (correlation −0.61 with the true γ).

The correlation of value and cost is still the number to re-check after any change to cost
or value. A lever that set recorded cost from the person's state, as `reentry_cost` once
did, can push it toward proportional (V1 cases in `RETIRED.md`).

**Carrying prediction error into value** (2026-09-21) changes the correlation only a
little. On the question-type task at the default setting, `f = k = 0.8`, 400 decisions,
means over 8 seeds (2026-10-25, `python -m experiments.value_variation`):

| | corr(V(0), cost) | corr(δ, cost) | corr(value, cost) | sd of value, V(0) → with δ |
|---|---|---|---|---|
| feedback on | +0.10 | +0.19 | +0.20 | 0.259 → 0.313 |
| feedback off | +0.12 | +0.18 | +0.22 | 0.247 → 0.316 |

At a held setting the correlations are small and positive: carrying the error adds about
0.1 to `corr(value, cost)`, and about 0.06 to the sd of value. Under V3 the same runs gave
negative correlations (−0.29 to −0.34, `RETIRED.md`); the sign changed with the session fix,
and the V3 explanation for it went with it. What matters for identification is the
magnitude. At about 0.2 here, and 0.35 to 0.64 under random probing (above), cost is far
from proportional to value, and `f` and `k` are recovered (RMSE 0.033 and 0.062 in the
probed runs).

---

## Known limitations

**Constants and forms the app's fit assumes**

- **With γ held fixed, `f` is confounded with the discount factor.** Under the continuing
  bootstrap `V(0) ≈ γ^(n−1)·E[r] / (1 − γ^n)`, so a wrong γ mostly rescales value and `f`
  absorbs it: a person whose γ is 0.8 gets `f` 47% too low and `k` 3% too high. Fitting γ
  removes this in simulation ("Fitting the discount factor"), but the app's fit still holds
  γ at 0.9, and so do most results here, the controller comparison included. `f` is also
  identified only up to `ACHIEVEMENT_SCALE`, which multiplies value. State γ (or that it was
  fitted), the scale and the bootstrap whenever an `f` value is reported.
- **The reward is assumed in every fit the app runs.** Right answers are worth `1 − p`
  with the score shown and the fraction right with it hidden (decided on 2026-10-29, not
  tested: that what is shown is what pays). A person who values the fraction right with
  the score shown gets `f` 46 to 54% too low even with γ
  fitted ("Fitting the discount factor", robustness). Comparing the two by likelihood picks
  the right one for 63 of 64 simulated people and removes the bias ("Measuring the learning
  rate and the reward"), but only these two reward forms have been compared.
- **The learning rate α is assumed (0.05)** in every fit the app runs. A person whose α is
  0.01 to 0.2 gets `f` within 10% at γ 0.8 but −36% to +51% at γ 0.95 ("Fitting the
  discount factor", robustness). Profiling α measures it only to about a factor of 5 (95%
  interval) from 300 start decisions plus redos.
- **Both decision nodes share one accumulator** and one set of parameters. That is an
  assumption: a real task might give the redo choice a different deadline.

**What the data can measure**

- **At the start node, effort cost is weak and retrospective.** Review load adds 0.008 to
  cost on average, with a standard deviation of 0.013 to 0.017 within a run (default
  setting, the same runs as the clip check below). That is all the cost variation a
  recorded start decision has, in human data and, since V4, in simulation. Before V4 the
  simulator's recorded cold starts supplied most of it (`RETIRED.md`, "The recording
  flaw"). The redo node's cost varies far more, which is why `InfoController` leans on it.
- **The value range is endogenous.** A stalled learner stops generating value variation, so
  there is less leverage on `f`. The recovery runs here have not shown it as a loss: `f` is
  recovered about as well at every `k`, and its worst grid point in "Recovery from one
  person" is at k 0.2.
- **`T0` near zero.** The non-decision time is bounded to [0, 0.5] s when fitted. Below that
  bound only the 2% lapse mixture gives a fast decision any density, so a person whose
  non-decision time exceeds their own fastest decision cannot be fitted, and the bound has
  to move with the data. With few decisions, a fitted `T0` of 0 cannot be told apart from a
  small positive value. In the five-parameter recovery, 6 of the 16 people whose true `T0`
  was 0.06 s or less were fitted at 0, and no one with a larger `T0` was. `T0`'s standard
  errors from the observed information run slightly narrow (89.2% coverage where 95% is
  nominal) and give no interval at all for a fit at 0. Profile-likelihood intervals cover
  the truth in 91% of all 100 fits, those at 0 included ("Profile-likelihood intervals",
  under "The bound can be fitted"), so they are the ones to report.
- **Give-up times cannot measure the bound.** In seconds, a give-up time is inverse Gaussian
  with mean `patience · BOUND / μ` and shape `patience · BOUND² / SIGMA²`, and scaling
  `BOUND` by λ, patience by 1/λ² and the drift by 1/λ leaves both unchanged. So where the
  bound is fitted, it has to come from the decisions, as `f` and `k` do, and patience then
  fitted with it held. `persistence.py` still uses the pinned `BOUND`, and the
  five-parameter recovery does not fit patience.
- **Patience is estimated with `f` and `k` held** at the decisions' estimate, so their errors
  pass into it, and it absorbs anything about giving up that the drift does not describe. A
  skip at a glance and giving up after a long try are one process on different time scales
  here, not two processes.
- **A type's rating only moves when that type is served.** Types far from every target the
  person ever gets keep their prior indefinitely.

**What the model cannot represent**

- **Someone spurred on by a visible miss.** Under pure RPE a miss they see is a negative
  error for everyone, scaled by their own `f`; only a free `b_seen > 0` could give it the
  other sign, and that is what was removed. If a real person turns out to work that way, it
  has to re-enter as a cost effect, not a value one.
- **`f` does two jobs**: how much the expected payoff pulls, and how hard the last set's
  surprise hits. The old `b` terms let those differ per person; folding them into value
  asserts they are one sensitivity. That is a substantive claim about people rather than a
  simplification, and it has not been tested against a human.
- **Nothing forgets.** Learning still saturates at a fixed difficulty, and under `choice` and
  `quiet` the missed list grows without bound.
- **Solve times.** Treating an answer as a censored time to give up assumes that how long a
  question takes to solve does not depend on motivation; someone who rushes when they want
  to stop breaks that. Simulated solve times are an assumption (median `4 s / p^0.7`,
  lognormal), not fitted to human answer times.
- **Time on a question includes anything else they did meanwhile.** Walking away
  mid-question reads as long persistence.

**The app and the human data**

- **The app's controller is not measured.** `AppController` chooses among 1104 settings,
  and nothing shows yet that it measures `f` and `k` as well as `InfoController` did or
  keeps people engaged as well ("The app's controller").
- **λ rests on its prior when countdowns are rare.** With few timed prompts the fit keeps
  λ near 0.7, so those prompts are read at a bound that may not be the person's. How much
  that moves `f` and `k` has not been measured.
- **The deadline is set from the human decision times seen so far.** Anyone slower than
  `T_DUR = 7.0` is censored again; `python -m temporal audit` flags a share past the
  deadline above 2%.
- **Interface rules with no simulator counterpart**: retiring questions, never repeating one
  within a set, and treating a quick fix as a typo (never after the end-of-set summary).
  `RETIRE_AFTER = 3` and `SLIP_SECONDS = 2` are judgment calls; retirement stops a
  question someone might have got right on a later showing.

**Numerical shortcuts**

- **The prediction error is clipped at `±DELTA_CLIP` (1.0)** so `value` stays on a bounded
  grid. At the default setting it binds on 0.2 to 0.8% of start decisions with feedback shown
  and 0.7 to 0.8% with it hidden. With redo prompts in place of review it binds on 0.3 to
  0.5% and 0.5 to 1.2% of start decisions, and on none of the redo decisions. Ranges are over
  four people, f, k = 0.8/0.8, 1.2/0.4, 0.6/1.2, 1.2/1.2, 2 seeds × 300 start decisions each
  (2026-10-25, `python -m experiments.value_checks`).
- **Value and cost are discretized** (`VALUE_STEP`, `COST_STEP`) to keep the number of
  Fokker-Planck solves bounded. The closed-form likelihood does not need it; the simulator
  and the app's pyddm fit still round.

---

## Next steps

### Near term: prepare the inference for real data

1. **Non-decision time**: fitted rather than assumed away since 2026-09-15, for the
   controller's estimate and the interface's fit ("Recovery demonstrates identifiability").
   What is left is to watch it per person as sessions accumulate, and to move the [0, 0.5] s
   bound if anyone's fastest decision falls near it.
2. **Keep the deadline past human decision times**: run `python -m temporal audit` after each
   batch of sessions. Done once, on 2026-09-13, which moved it from 3.0 to 7.0 s.
3. Add across-trial drift variability (Ratcliff's `sv`).
4. **Value for a real person has an observable proxy.** Under the continuing bootstrap it is
   `V(0) ≈ γ^(n−1) · E[set reward] / (1 − γ^n)`, and set reward is computable from the
   measured chance of each question, so no TD fit is needed. Confirm it tracks during
   learning, not only at the fixed point.

### Next: individual-level estimation

5. **Hierarchical estimation.** Tried on 2026-10-28 in simulation ("From less data, alone and
   pooled"): empirical-Bayes pooling helps γ, the bound and `T0` with little data and barely
   moves `f` and `k`, and its `T0` intervals run narrow. What is left: a population model that
   respects `T0`'s bound at 0 (or a full Bayesian fit), and testing it at the tens of
   decisions a real session gives.
6. Posterior predictive checks and held-out prediction on recorded sessions.
7. Seeding learners across the learning curve (`--mode reseeded`) is the recovery design
   ("Current results"). Under V2 it did nothing detectable for `f` against one continuous
   learner (`RETIRED.md`).

### Then: close the loop

8. **Improve the controller.** From 2026-10-11 to 2026-10-29 the app ran `InfoController`,
   and since then `AppController`, the same score over every setting the app has, which is
   not measured yet ("The app's controller"). `InfoController` measures `k` far better
   than reward planning (RMSE 0.156 against 0.508 over 300 ticks, though the paired test
   fixed in advance fell short of 2 SE) at no detectable cost in reward, and `f` no better
   ("Measuring costs no detectable reward"). What remains:
   - **Re-measure the reward controller under the current model.** On 2026-10-29 it was
     run against `InfoController`, the fixed default and random choice on the 128 hypercube
     people ("The app's InfoController and reward planning, on the same people"). Its only
     evaluation against the best fixed setting is still the V2 8-seed table in
     `RETIRED.md`, and two of its five people cannot be rebuilt. Open questions
     from it: the reluctant, effort-averse person (too few sets to decide), the gap to the
     best fixed setting given true parameters, and whether exploration probes cost reward.
   - **Which objective.** Total reward and engagement disagree on where to aim ("Which
     difficulty and whether to show feedback"). An engagement maximizer was built on
     2026-10-29 ("Four controllers"): it kept people in the task for 58% of ticks against
     41% at the fixed default, with short, easy sets. Information planning kept them there
     for 28%. What is left: what the app's controller should trade between the two, and
     whether engagement with a floor on challenge serves a learner better.
   - **γ needs start decisions.** Under every controller γ is measured to about 0.01 to
     0.05 for people who give 100 or more start decisions and not at all below 50 ("Four
     controllers"). Information planning, which measures γ best per decision, gives the
     fewest. The next thing to try is information planning with a floor on time in the
     task (its safety floor is only on sets started in a rollout), and a longer run.
   - **Feedback.** Under the current model, showing feedback raised engagement at every
     target but 20% for f 0.8, k 0.8, and never lost on reward beyond 2 SE. Across the grid
     it lowered P(start) beyond 2 SE in one cell of 96 (f 0.6, k 1.0 at 20%, 3.2 SE). Run
     that cell and the hard targets with more seeds before the controller relies on hiding
     feedback.
   - **Difficulty is one target per set**, from six levels, served from nine question types.
     There is no sequencing within a set (easy then hard), and at extreme targets some
     people cannot be served questions that match the label.
   - **The skipped-questions lever is unproven.** Nothing measured shows when returning
     skipped questions helps. A person whose ability grows from corrections would be the
     test.
   - **The stage count has no model channel.** Set length and the score display became
     levers on 2026-10-29 ("The app's controller"); whether showing the count matters, and
     how, waits on the literature.
   - **Measure `AppController`**: against `InfoController`, paired on person and seed, for
     `f` and `k` error, λ, and P(start), with recovery of the new levers first.
   - **Plan with fitted γ and bound.** Built and compared on 2026-10-28 ("Planning with
     fitted γ and bound"): it kept people engaged better but did not measure `f`, `k` or γ
     better at 300 ticks. With all six parameters and the countdown and feedback as
     settings ("Every parameter, every setting", 2026-10-29) it hid feedback before 86% of
     start decisions and measured λ, but γ gained short of 2 SE and `k` and the bound lost
     a little. Scored on `f`, `k` and γ only (Ds-optimal, same section) it dropped most
     countdowns, measured the bound better and `k` no better, and cost some engagement.
     With 384 settings, set length among them, the Ds-optimal planner measured `k` better
     than `InfoController` over the same settings, just beyond 2 SE (squared error
     −0.069 ± 0.034; RMSE 0.201 against 0.336, "Four controllers" and the section after
     it; no claim fixed), and `f` and γ no worse. Over these settings, then, planning with
     fitted γ and bound did not cost `k`. Why the earlier comparisons leaned the other way
     (never beyond 2 SE) is not settled: the two planners also differ in their score and
     in comparing settings on the same seeds.
   - **Feedback in blocks.** Tried on 2026-10-28 ("Feedback in blocks, with redos mid-set or
     at the end of the set"): with redos moved to the end of the set it kept the redo
     decisions and cost no `k`, but did not measure γ better in 300 ticks, where most people
     reach one or two hidden blocks. What is left: blocks over a longer run (or shorter
     blocks). The app has had the end-of-set screen and its redo prompts since 2026-10-28.
     Its controller, `AppController` since 2026-10-29, can choose end-of-set redos, but it
     does not switch feedback on a block schedule.
9. **Online re-estimation.** Parameters drift within a session with fatigue, mood and
   boredom, so the controller must track them rather than fit once.

### Deferred, but kept in view

- Put the reward comparison and the α profile into the app's fit. Both now pass recovery in
  simulation ("Measuring the learning rate and the reward"), the reward comparison clearly
  and α only loosely.
- Two-state fast/slow ability (Smith, Ghazizadeh & Shadmehr 2006), if the protocol ever gains
  washout and re-exposure blocks.
- Model a process that degrades: fatigue, forgetting, or the rising cost of continued
  exertion.

---

## Reproducing every result

Every number in this README comes from code in this repo. Simulations save their results
to `results/`, which is committed, and re-printing a table from the saved results
(`--replot`) takes seconds. Re-running a simulation from scratch takes minutes to hours and
gives the same results, because every run is seeded. Fits may differ slightly because of how
pyddm's optimizer works.

### Setup

Install as in "Running the app", then check the install:

```bash
uv run pytest
```

### Re-print or re-run each result

Each `python -m experiments.<name>` runs its simulation, saves `results/<name>/`, and prints
the README's table for it. Add `--replot` to print the table from the saved results without
running. Runs use all but one core; set `TEMPORAL_WORKERS` to use fewer, for example on a
machine with less memory per core. Most take minutes. The controller comparison took about
2 hours on 9 workers on 2026-10-25, and the misspecification sweep several.

| README result | command | results |
|---|---|---|
| Recovery from one person; Censored trials are data | `python -m experiments.recovery_sweep` | `results/recovery_sweep/` |
| The recovery pipelines (single, coupled, uncoupled) | `python -m temporal recovery --mode MODE --reps R --out results/pipelines` with R = 3, 3, 5 | `results/pipelines/MODE.csv` |
| The coupled recovery plot | `python -m temporal recovery --mode coupled --reps 3` | `assets/initiation_recovery_coupled.png` |
| Prediction error is a large part of the variation in value; the carry table | `python -m experiments.value_variation` | `results/value_variation/` |
| Skips do not help estimate `f` and `k` | `python -m temporal recovery --mode persistence --reps 3 --out results/pipelines` | `results/pipelines/persistence.csv` |
| What each part of the fit contributes | `python -m experiments.fit_factorial` | `results/fit_factorial/` |
| Patience can be recovered | `python -m temporal recovery --mode patience --reps 3 --out results/pipelines` | `results/pipelines/patience.csv` |
| Fitting the discount factor | `python -m experiments.gamma_profile` (split by `--gammas`, `--reps`, `--tag`) | `results/gamma_profile/runs_*.csv` |
| When other constants are wrong | `python -m experiments.gamma_robustness` | `results/gamma_robustness/runs_*.csv` |
| Where the information about γ comes from | `python -m experiments.gamma_sources` | `results/gamma_sources/runs_*.csv` |
| Feedback switched in blocks | `python -m experiments.gamma_blocks` | `results/gamma_blocks/runs_*.csv` |
| The bound can be fitted: f, k, γ, `BOUND` and `T0` together | `python -m experiments.recovery_lhc simulate`, then `fit` (split `simulate` by `--people` and `--tag`) | `results/recovery_lhc/histories/`, `results/recovery_lhc/fits.csv` |
| Profile-likelihood intervals | `python -m experiments.lhc_profile` | `results/lhc_profile/` |
| From less data, alone and pooled | `python -m experiments.lhc_size` | `results/lhc_size/` |
| Measuring the learning rate and the reward | `python -m experiments.alpha_reward` | `results/alpha_reward/runs_*.csv` |
| Which difficulty and whether to show feedback (engagement, reward, and feedback for one person) | `python -m experiments.task_choice` | `results/task_choice/` |
| Across people, with feedback by difficulty and the two ends of the engagement curve | `python -m experiments.feedback_grid` | `results/feedback_grid/` |
| Feedback by γ | `python -m experiments.gamma_feedback` | `results/gamma_feedback/` |
| Lockout | `python -m experiments.lockout` | `results/lockout/` |
| Measuring costs no detectable reward and its paired tests; It found the redo prompts on its own; Over time; What measuring costs, block by block | `python -m experiments.controllers` | `results/controllers/` |
| Intervals beside the controller comparison, and the five people's feedback intervals (no simulation) | `python -m experiments.intervals` | `results/intervals.txt` |
| Planning with fitted γ and bound | `python -m experiments.controller_full` | `results/controller_full/runs_*.csv` |
| Four controllers, all five parameters fitted | `python -m experiments.controller_full --arms fixed` and `--arms random_return,engage,info_fkg` (split by `--people` and `--tag`) | `results/controller_full/runs_*_base.csv` (fixed), `runs_*four*.csv` |
| The app's InfoController and reward planning, on the same people | `python -m experiments.controller_full --arms info_end,random_end,reward_end` | `results/controller_full/runs_*_end*.csv` |
| Feedback in blocks, with redos mid-set or at the end of the set | `python -m experiments.feedback_blocks` (split by `--people` and `--tag`) | `results/feedback_blocks/runs_*.csv` |
| A countdown on the redo prompt | `python -m experiments.redo_pressure` (split by `--people` and `--tag`) | `results/redo_pressure/runs_*.csv` |
| Every parameter, every setting | `python -m experiments.controller_all` (split by `--people` and `--tag`; reads `results/controller_full/` for the other two arms) | `results/controller_all/runs_*.csv` |
| Non-decision time (the T0 table) | `python -m temporal recovery --mode t0 --reps 3 --out results/pipelines` | `results/pipelines/t0.csv` |
| The person may not share the model's constants | `python -m experiments.misspecification` | `results/misspecification/` |
| How value and cost together decide what can be measured (the precision table) | `python -m experiments.precision` | `results/precision/` |
| The same with γ, the bound and `T0` free | `python -m experiments.precision_full` | `results/precision_full/` |
| How often the error hits `DELTA_CLIP`, the start node's effort cost, and what the continuing bootstrap does to value | `python -m experiments.value_checks` | `results/value_checks/` |
| The app's controller: the closed-form fit against pyddm's, λ recovery, plan time | `python -m experiments.app_checks` | `results/app_checks/` |

The `temporal recovery` sweeps print their tables as they finish. They have no `--replot`,
but their CSVs are saved. To re-run everything, run every command in the table.

### Paper figures

`python -m paper` draws every figure from the saved results into `paper/figures/`;
`python -m paper fig7 fig8` draws only those. It also writes `paper/figures/conditions.md`,
which says what each figure's data was generated under. That text comes from the
`conditions.json` each experiment saves beside its results, so it matches the results
drawn. `python -m paper --captions` prints it without drawing. The table below gives the
paper's conditions in short, and each experiment's docstring gives them in full.

| figure (`python -m paper` name) | shows | data in the paper | results read |
|---|---|---|---|
| 1 (`fig1`) | the model | schematic | none |
| 2 (`fig2`) | drift planes | exact drifts: one person, three decisions | none |
| 3 (`fig3`) | patience | a recorded human session | `data/decisions.jsonl` |
| 4 (`fig4lhc`) | f, k, γ, the bound and `T0` recovered together | 100 people from a Latin hypercube, 300 start decisions plus their redo decisions, `InfoController` | `recovery_lhc`, `lhc_profile` |
| `fig4` | f and k recovered alone | 48 fits: 4 × 4 grid × 3 seeds, 1200 start decisions, single-difficulty task | `recovery_sweep` |
| 5 (`fig5`) | bias once γ is fitted | 3 people × 5 seeds (γ held, γ fitted), 3 people × 2 seeds (one other constant wrong), the 100 people of Figure 4 | `misspecification`, `gamma_profile`, `gamma_robustness`, `recovery_lhc` |
| 6 (`fig6`) | what each shortcut in the fit costs | (a) as `fig4`; (b) f = k = 0.8, true `T0` 0 to 0.5 s × 3 seeds, 1200 start decisions; (c) 4 people × 3 seeds, 300 start decisions; (d) the 100 people of Figure 4 | `recovery_sweep`, `pipelines`, `precision_full` |
| 7 (`fig7`) | difficulty and feedback | 5 people, each setting held for 300 ticks, 16 seeds; (c) γ 0.5 to 0.99, 32 seeds | `task_choice`, `gamma_feedback` |
| 8 (`fig8lhc`) | four controllers, all five parameters fitted | 128 people from a Latin hypercube, 300 ticks under each: the fixed default, a random setting at each return, the engagement maximizer and information planning on f, k and γ, the last three over the same 384 settings | `controller_full` |
| `fig8` | how well each controller measures k when every person has γ 0.9 and bound 1.5 | 32 people (4 × 4 grid × 2 seeds), 300 ticks, f and k fitted | `controllers` |
| 9 (`fig9`) | planning with all five parameters | 128 people from a Latin hypercube, 300 ticks under each controller | `controller_full` |
| S2 (`figS2`) | timed-out trials kept or dropped | as `fig4` | `recovery_sweep` |
| S3 (`figS3`) | the fit factorial | 32 people (4 × 4 grid × 2 seeds), 300 start decisions, true `T0` 0.3 s | `fit_factorial` |
| S4 (`figS4`) | reward by difficulty, one-person feedback test | f = k = 0.8, 96 seeds; the 5 people of Figure 7 | `task_choice` |
| S5 (`figS5`) | controllers over time | as `fig8` | `controllers` |
| S6 (`figS6`) | feedback by γ | 5 people, γ 0.5 to 0.99, 32 seeds | `gamma_feedback` |
| S7 (`figS7`) | bias with γ held at 0.9 | 3 people × 5 seeds, with and without redo decisions | `misspecification`, `gamma_profile` |
| S8 (`figS8`) | the five-parameter fit from less data | the 100 people of Figure 4, cut at 50 to 300 start decisions | `lhc_size` |
| S9 (`figS9`) | measuring α and the reward | 64 people from a Latin hypercube, 300 start decisions plus their redo decisions | `alpha_reward` |
| S10 (`figS10`) | feedback in blocks, redos mid-set or at the end | 128 people from a Latin hypercube, 300 ticks under each of three controllers | `feedback_blocks` |
| S11 (`figS11`) | a countdown on the redo prompt | 64 people from a Latin hypercube with λ 0.4 to 1.0, 300 start decisions plus their redo decisions, with and without the countdown | `redo_pressure` |

### Running under other conditions

Every experiment behind a figure reads its conditions from the command line, and the
paper's are the defaults. They are set in the `design(...)` call near the top of each
experiment file.

- `--set name=value` changes one condition (repeat it for more), and `--out folder` names
  the results folder.
- If any condition differs from the paper's, the run won't start without `--out`, so
  `results/<experiment>/` can't be overwritten.
- An unknown name prints the conditions the experiment has.

```bash
python -m experiments.controllers --set reps=4 --set "checkpoints=(100, 200, 300, 500)" --out controllers_long
python -m paper fig8 figS5 --results controllers=controllers_long
```

The second command draws into `paper/figures/controllers_long/`, and the `conditions.md`
there lists what changed from the paper's conditions. Give `--results` once for each
experiment a figure reads (the last column of the table). `precision_full`, `lhc_profile`
and `lhc_size` read the saved histories of a `recovery_lhc` run and take its folder as a
condition, for example `--set histories=lhc_big`.

- The `conditions.json` files for runs made before 2026-10-28 were written afterwards from
  each experiment's defaults, after the saved results were checked against them. Their
  `date` and `commit` are empty.
- The `temporal recovery` sweeps behind Figure 6 (b, c) save no `conditions.json`, so their
  captions count the repetitions from the CSV.
- Figures take their people, seeds, γ values and checkpoints from the results. Some panels
  still place the paper's levels by hand, such as the α and bound axes of Figure 5, and may
  need adjusting for very different conditions.

### Checks that read a recorded session

No session record is in the repo (`data/` is gitignored). Check a record's decision times
against the deadline with `python -m temporal audit --path path/to/decisions.jsonl`.

### Other commands

```bash
python -m temporal.controller --values 0.8,0.8   # one simulated person: a fixed setting vs both controllers
python -m temporal run --values 0.8,0.8          # one simulated learner on the single-diff task
python -m temporal recovery --mode reseeded      # recovery from four learners across the learning curve
```

- `run`: `--values` takes `f,k`; `--cost` picks the cold-start profile; `--review-mode` and
  `--review-rate` set review; `--diff-spread` sets per-stage difficulty spread; `--decisions`
  caps the number of start decisions (default 200). Bare `python -m temporal` is shorthand
  for `python -m temporal run`.
- `recovery`: `--mode` picks the pipeline, `--reps` sets repetitions per grid point (default
  5), and `--out` saves each sweep's results.
- `audit`: `--path` points at a `decisions.jsonl` (default `data/decisions.jsonl`).
- `temporal.controller`: `--values`, `--decisions` (default 300), `--seed`.

### From a browser instead (optional)

Everything above runs from a terminal, and that stays the way to reproduce a result.
`taskboard/` is an optional dashboard over the same commands, for anyone who would rather
not type them, or who wants to see how far a long run has got. A run started from it is the
same `python -m` command, run from the repo root, and it saves `results/<name>/` as before.

```bash
uv run python -m taskboard        # then open http://127.0.0.1:8765
```

- **It finds the runs itself.** It uses only the standard library. Nothing in it lists the
  runs by hand: it reads every module with a `__main__` block in the repo's code folders,
  along with the usage lines in its docstring and its argparse options, from the source and
  without importing it. A new experiment shows up without being registered anywhere. A menu
  picks the checkout to run from, the main repo or any git worktree.
- **Each run gets a card** with a progress bar, the time elapsed and an estimate of the time
  left, read from the `[i/N]` lines that `run_pool` and the recovery sweeps print (pytest's
  percentage and tqdm bars also work). The full log is one click away. The card and the log
  also give the commit the run started from, and the card warns when the checkout had
  uncommitted code changes (changes under `results/` and `logs/` don't count). A run like
  that can't back a README number until its code is committed, by the rule at the top of
  this section.
- **Runs can be split across computers.** Any list option, such as `--people 0-99` or
  `--reps 0,1`, can be split into one piece per computer, sized by its cores, or into N
  pieces that whichever computer is free takes next. `--tag {host}` becomes each computer's
  tag, so every piece writes its own `runs_<tag>.csv`, and a follow-up such as `--replot`
  can run once all the pieces are done. A queued run can be moved to another computer; a
  running one is stopped and starts again there from the beginning. Runs started by hand in
  a checkout appear as well, with their progress when their output goes to a file.
- **It is careful with runs.** The board asks before adding a run that is already queued or
  running, and holds a run until its computer has the cores it asked for (the workers box
  sets `TEMPORAL_WORKERS`). Runs are detached, so closing the board doesn't stop them, and
  reopening it picks them up again.

With no configuration it uses only the computer it runs on, tagged `local`, the
experiments' own `--tag` default. To add computers reached over `ssh`, copy
`taskboard/hosts.example.json` to `taskboard/hosts.json` (gitignored) and edit it. Each
entry gives the ssh host, the folder that holds a copy of this repo, how many cores to use,
and the tag. Before a remote run the board copies the checkout's code to that folder with
rsync, only when the code differs and never while something else is running there. It
doesn't copy `results/`, `data/` or `logs/`. When the run finishes, the board copies the
result files it wrote back into the checkout. Logs and process ids are kept in
`~/.taskboard/runs/` on each computer, and the board's own state in `taskboard/state.json`
(gitignored).

---

## Files

```
temporal/
  cli.py            entry point (python -m temporal): single runs, recovery sweeps, audit
  model/            the person, as the environment models them
    params.py         UserParams (f, k, discount, learning rate, patience, T0, λ) and ModelState
    accumulator.py    the drift-diffusion process behind start and redo decisions; cached solves and draws
    conditions.py     the (value, cost) each decision node feeds it
    value.py          TD value of a set, and when a slot's reward lands
    competence.py     chance of a right answer, and how it grows
    persistence.py    giving up on a question: quit drift, patience, closed-form likelihood
    pressure.py       the redo prompt under a countdown: the person's bound drops to λ · BOUND
    questions.py      question types, their priors, and per-person Elo ratings
    simulation.py     the simulated learner: sets, review, redo prompts mid-set or at the end
                      of the set, countdowns, start decisions
  inference.py      sample construction, censoring, the two-parameter fit, the app's closed-form
                    fit with λ, sanity checks
  fastlik.py        the accumulator's likelihood in closed form, checked against pyddm
  recovery.py       data generation and recovery pipelines, including persistence and patience
  controller.py     Controller (plans for reward), InfoController (plans to measure f and k)
                    and AppController (the app's: InfoController's score over 1104 settings)
  plots.py          recovery figures
  app/              the playable task
    session.py        the rules of a session, shared by both frontends
    store.py          the record on disk: append-only log, snapshot, saved controller
    desktop.py        the Tk window
    web.py            the same session served to a browser; static/index.html draws it
    live.py           the live view of the accumulator for the decision on screen
experiments/        the runs behind the README's results; each saves results/<name>/
results/            their saved results (CSV), committed so every table can be checked
paper/              the paper's figures, drawn from results/ (python -m paper) into paper/figures/ (gitignored)
taskboard/          optional browser dashboard for starting, watching and splitting runs (python -m taskboard)
  hosts.example.json  the computers it can use; copy to hosts.json (gitignored) to add more
tests/              seeded regression tests (golden.json), the session rule (test_sessions.py),
                    the closed-form likelihood against pyddm (test_fastlik.py), end-of-set
                    redos (test_end_redos.py), the countdown (test_pressure.py), the app's
                    controller, end-of-set screen and settings (test_app_*.py), and the
                    taskboard's log and catalog reading (test_taskboard.py)
RETIRED.md          results from earlier model versions, and ones later measurements replaced
Dockerfile, fly.toml   hosting the web app; data in a volume at /data
data/               human session records (gitignored)
assets/             figures the README shows
```

---

## Code layout, tests and speed: since 2026-10-11

The model's equations did not change on 2026-10-11. What changed is which controller the
app runs, how the code is organised, how it is checked, and how fast it plans.

- **The app runs `InfoController`** ("Planning to measure"), over 96 settings: the 84
  scheduled-review settings, which were all the app's controller had until then, plus 12
  with redo prompts. Rollouts now follow each setting's review mode. Before, they always
  scheduled review while the app asked redos, so the planner was simulating a task the
  person never saw. A saved controller from before this change is replaced on the next
  launch; the replacement is logged as `controller_reset`, and the estimate is refit from
  the record.
- **One package, `temporal/`** ("Files"). `config.py` was split by what each part models:
  the person (`params`), TD value (`value`), competence, and the (value, cost) each node
  sees (`conditions`). `interface.py` was split into the session's rules
  (`app/session.py`), the record on disk (`app/store.py`) and the Tk window
  (`app/desktop.py`). The web frontend subclasses the same `Session`, so both still write
  the same record. Commands are now `python -m temporal ...` ("Reproducing every result").
- **Regression tests** (`uv run pytest`, about 5 s on 2026-10-25) pin seeded outputs in
  `tests/golden.json`. They cover the simulator under scheduled review and under redo
  prompts, people under a fixed setting and under both controllers, a fit to designed data
  and one to a learner, the give-up likelihood and patience, and a scripted app session (48
  events). They were recorded on the code before the move, and the restructured code reproduces them
  exactly. The exceptions are fits, compared at 1e-3, because pyddm's optimiser stops a few
  1e-5 apart between runs. Re-record (`--record`) only for a change meant to alter output,
  and say why in the commit. They were re-recorded on 2026-10-25 for V4, which changes every
  simulated output; the scripted app session did not change. `tests/test_sessions.py`
  checks the session rule itself. The regression tests pass unchanged on an x86 Linux
  machine as well as on the Apple silicon Mac they were recorded on.
- **Planning is about 3× faster.** One `InfoController` plan (96 settings × 6 draws × 25
  rollout steps, f 0.8, k 0.8) took 15.3 s before this change and 4.6 s after it
  (2026-10-11). Re-timed under V4 on 2026-10-25 (f and k given, after 100 decisions at the
  default setting), it takes about 0.9 s, and 1.2 s the first time as the caches fill.
  Rollouts now count ticks, and a tick spent away plays no set. The same speedup applies to
  every simulation and recovery sweep. Two changes did it:
  - A start or redo draw no longer goes through pyddm's `Solution.sample(1)`. That call
    rebuilt its sampling tables and ran its argument-checking decorators on every draw.
    `accumulator._draw` builds the tables once per condition and makes the same calls on
    numpy's global generator in the same order, so every seeded run is unchanged (the tests
    above confirm it). Since V4 it adds one more draw, only when a decision is undecided, for
    the bound the person ends up at. If pyddm is upgraded, check `_draw` against
    `Solution.sample` again.
  - `sigmoid` on a single number uses `math` instead of numpy.
- **Dependencies** are the eight packages the code imports, with pytest and ruff as dev
  tools. The other 22 of the 32 earlier pins were dependencies of dependencies, or optional
  pyddm backends this code never calls (`pybads`, `gpyreg`, `pathos`).
