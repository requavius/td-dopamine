# Retired results

`README.md` holds only results measured under the current model. Everything here was
measured under an earlier model, or was superseded by a later measurement under the same
one. It is kept so that nothing is lost and old claims can be checked, not to be quoted
as current. Each entry says which model produced it, what replaced it, and why it moved.

**The rule.** When a change to the model, the task or the measurement makes a README
result out of date, the new measurement goes in the README and the old one comes here,
under the model version that produced it. A result whose model version is uncertain is
re-measured or moved here; it does not stay in the README as if current. Implementations
that did not pan out are listed at the end, by the model they belonged to.

## Model versions

| version | dates | drift | reward | deadline `T_DUR` | value bootstrap | fitted | what the simulator records |
|---|---|---|---|---|---|---|---|
| V0 | 2026-03 to 2026-05 | engagement score with boredom `b` and effort cost, no accumulator | stages completed | none | none | particle filter | no decisions |
| V1 | 2026-05 to 2026-09-13 | `f·value − k·cost`, plus `b_seen·seen_miss + b_unseen·unseen_miss + b_fb·feedback` from mid-2026 | fraction right, then hard right answers `4·Σ(1 − p)/n` | 3.0 s | terminal (`V_next = 0`) | 5 parameters by pyddm, `T0` pinned at 0 | every start draw, at one cost (the recording flaw) |
| V2 | 2026-09-13 to 2026-09-21 | as V1 | hard right answers | 7.0 s | continuing | as V1; `T0` free for real people from 2026-09-15 | as V1 |
| V3 | 2026-09-21 to 2026-10-25 | `f·value − k·cost`; prediction error carried in `value`; reward lands when shown | hard right answers | 7.0 s | continuing | 2 parameters (`f`, `k`), `T0` free for real people | as V1 until 2026-09-27; then up to three stops in a row per session, and every cold start after them |
| **V4 (current)** | since 2026-10-25 | as V3 | hard right answers | 7.0 s | continuing | as V3 | what the app records: one stop ends the session, time away and the cold start on return are not recorded, and a choice slower than `T_DUR` is still carried out |

Changes under V3 that did not change the model: the information-seeking controller and
redo-prompt settings (2026-10-11), the package restructure (2026-10-11, regression tests
reproduce every seeded output), and the move of the experiments and figures into the
repo, with seeded recovery sweeps (2026-10-20). Only the sweeps' tables were re-run.

V4 changes no equation. It changes what a simulated person does and what is recorded of
it (see "The recording flaw in V1 to V3"), which changes every seeded output, so every
simulated result was re-measured. Changes alongside it that leave seeded output alone:
the lapse mixture is passed to pyddm explicitly at its default, simulated redo records
log the error they were decided after (as the app's do), and `experiments/value_checks.py`
replaces two hand measurements.

`f` values are on different scales across versions: with γ held fixed, `f` is identified
only up to γ, `ACHIEVEMENT_SCALE` and the bootstrap, all of which changed between V1 and
V2.

Changes under V4 that did not change the model (2026-10-28): γ, and in one experiment
`BOUND` and `T0` as well, can be fitted per person instead of held at 0.9 and 1.5 (README,
"Fitting the discount factor" and "The bound can be fitted"); a closed-form likelihood
(`temporal/fastlik.py`) that matches pyddm's; and the simulator records the achievement
reward as it accrues, which changes no draw (the regression tests pass unchanged). No
earlier result was re-measured: every result fitted with γ at 0.9 stands, under that
condition. What these changes retired is listed below, under "Claims withdrawn under
V4".

Two optional features were added under V4 on 2026-10-28, both off by default, so every
seeded output is unchanged (the regression tests pass unchanged).
- **Redos at the end of the set** (`Simulation(end_redos=True)`, `Environment.end_redos`,
  `controller.REDO_END`). With feedback hidden, the misses are offered for redo on the
  end-of-set screen instead of not at all.
- **A countdown on the redo prompt** (`temporal/model/pressure.py`,
  `Environment.pressure`, `UserParams.pressure`). The person's bound on the prompt drops
  to `pressure` × `BOUND`, and the prompt closes without a redo at 3 s.

Neither is in the app's controller or the app yet. `fastlik.neg_log_likelihood` also
takes a bound and a deadline per decision now, which changes no likelihood computed
before.

---

## The recording flaw in V1 to V3 (fixed 2026-10-25)

In the app, answering "I'm done" at the start prompt closes the app. A session therefore
holds at most one stop, and the decision to come back is opening the app, which nothing
records (its "reaction time" is hours). Every simulator before V4 recorded decisions a
person in the app cannot give:

- **Until 2026-09-27** (V1, V2 and the first days of V3) every start draw was recorded, at
  the one cost the learner had. A stop used up a tick, and the next draw was another
  recorded start decision.
- **From 2026-09-27** a stop ended the session only on the third in a row
  (`SESSION_EXIT_STREAK = 3`). After that every draw was a cold start, recorded at the
  cold-start cost, until one came up GO.
- **A draw that timed out counted as not starting**, and at the redo prompt as not
  redoing. In the app nothing times out: a slow click still carries out the choice, and
  only the fit reads it as undecided.

How much of the fitted data that was, counted on 2026-10-25 with the V3 code (seed 0).
"Rows a person could give" are every GO plus the first non-GO of each run of them, all in
session:

| run | cold-start rows | row after a non-GO is non-GO again | rows a person could give |
|---|---|---|---|
| `collect_single`, f 0.8, k 0.8, 1200 decisions | 47% | 81% | 42% |
| `collect_single`, f 1.5, k 0.5 | 16% | 57% | 71% |
| `collect_single`, f 0.5, k 1.5 | 82% | 95% | 14% |
| `controller.run`, default setting, f 0.8, k 0.8, 400 decisions | 10% | 47% | 84% |

What it did to the results:

- **The cold rows carried nearly all the cost variation at the start node.** The V3
  precision table (below) put the sd of cost at 0.154 with them and 0.012 without, and the
  expected SD of `k` from 300 start decisions at 0.140 against 0.509. Every recovery from
  start decisions alone (the single, coupled, four-learner and `T0` sweeps, and the
  single-`diff` part of the misspecification sweep) measured `k` with information a person never
  gives. The coupled pipeline's design, two learners at different cold-start costs, existed
  only in those rows.
- **Engagement and reward were counted per draw**, and a cold stop counted as a decision.
  The people who rarely start contributed long runs of cold stops (82% of the rows for f
  0.5, k 1.5), so their runs had hundreds of "decisions" that were time away.

Every simulated number in this file under V1, V2 and V3 came from such a simulator.
Results that do not run it are unaffected: the uncoupled pipeline (designed data), the
drift-plane and identifiability figures, and anything from human sessions. The app always
followed the rule; its scripted regression session did not change with the fix.

---

## V3, retired 2026-10-25 (the recording flaw)

These are the results the README held under V3 on 2026-10-25, copied as they stood. Each
was re-measured under V4 the same day with the same command; the README has the new
numbers. Cross-references ("above", "see ...") point to the README as it was then; the
results files are in git history (commit `487a533`), and so is the V3 coupled recovery
plot.

### Which difficulty and whether to show feedback; lockout

**2026-10-15.** Question-type task, skips on, one setting held for each run of 300 start decisions, review 0.25, skipped questions
dropped, no controller. The target is the predicted chance of a right answer each
question is aimed at; it is a prediction from the ratings, and the chance actually
achieved was not recorded. γ = 0.9, `ACHIEVEMENT_SCALE = 4`, continuing bootstrap,
`T_DUR = 7.0`; run with `python -m experiments.task_choice`. Claims fixed before the run
(paired on seed): a peak is determined if it beats both neighbouring targets beyond 2 SE;
the feedback crossover is established if shown − hidden is positive beyond 2 SE at one
target and negative beyond 2 SE at a harder one.

Engagement, P(start next set), by target, feedback shown, 16 seeds (descriptive; no
claim was fixed in advance for it):

| person | 90% | 75% | 60% | 45% | 30% | 20% | targets within 2 SE of the best |
|---|---|---|---|---|---|---|---|
| f 0.6, k 1.2 | 0.19 | 0.47 | **0.60** | 0.53 | 0.09 | 0.04 | 60, 45 |
| f 1.0, k 1.0 | 0.42 | **0.80** | 0.79 | 0.78 | 0.39 | 0.16 | 75, 60, 45 |
| f 0.8, k 0.8 | 0.42 | 0.77 | 0.78 | **0.79** | 0.50 | 0.26 | 75, 60, 45 |
| f 0.4, k 0.4 | 0.48 | 0.67 | 0.72 | 0.73 | **0.74** | 0.51 | 60, 45, 30 |
| f 1.2, k 0.4 | 0.76 | **0.92** | 0.90 | 0.89 | 0.90 | 0.81 | 75, 60, 45, 30 |

- **Engagement is a plateau, and effort aversion sets its width.** Each person engages
  about equally well over two to four targets. Low `k` keeps the plateau wide into hard
  questions; the most effort-averse person's is narrow (60 to 45%) and collapses at 30%.
- **Easy questions lose people.** At a 90% target, P(start) falls to about 0.4 for most:
  near-certain right answers pay almost nothing, which is the model's boredom.

Total reward by target, the same runs:

| person | 90% | 75% | 60% | 45% | 30% | 20% | peak |
|---|---|---|---|---|---|---|---|
| f 0.6, k 1.2 | 15.9 | 59.6 | 95.1 | **100.4** | 14.3 | 4.2 | 45%, not determined |
| f 1.0, k 1.0 | 25.6 | 80.3 | 94.8 | **117.6** | 61.4 | 17.5 | 45%, determined |
| f 0.8, k 0.8 | 25.3 | 76.3 | 93.7 | **119.8** | 81.4 | 30.7 | 45%, determined |
| f 0.4, k 0.4 | 31.3 | 74.4 | 99.6 | 110.9 | **136.2** | 68.9 | 30%, determined |
| f 1.2, k 0.4 | 39.3 | 82.1 | 95.7 | 119.2 | **139.6** | 116.6 | 30%, not determined |

- **Reward peaks harder than engagement.** Within the engagement plateau each right answer
  is worth more at a harder target, so reward keeps rising until engagement falls. The two
  people with `k = 0.4` peak at 30%. For the most effort-averse person, reward picks 45%
  and engagement 60%. That is why the target range reaches down to 20%.

Feedback shown − hidden, `f = k = 0.8`, 96 paired seeds:

| target | 90% | 75% | 60% | 45% | 30% | 20% |
|---|---|---|---|---|---|---|
| P(start) | **+0.068 ± 0.010** | **+0.080 ± 0.010** | **+0.042 ± 0.006** | **+0.072 ± 0.014** | **+0.070 ± 0.023** | **+0.051 ± 0.021** |
| total reward | **+2.9 ± 0.9** | **+5.6 ± 1.5** | −1.1 ± 1.4 | +3.5 ± 2.6 | +8.4 ± 4.8 | **+8.1 ± 3.7** |

- **Showing feedback raises engagement at every target**, by 0.04 to 0.08, each beyond
  2 SE (2.4 to 7.9 SE). A seen right answer lands as reward at once instead of being
  discounted to the end of the set.
- **On reward it helps at easy targets and is too noisy to call at hard ones.** At 20 to
  30% a run's total depends heavily on runs of luck, so the paired difference varies about
  five times more than at 90%. The 20% result (2.2 SE) is one of six tests and should not
  carry weight alone.
- **Across people (2026-10-15).** The 16 people of the 4×4 grid, six targets, shown and
  hidden, 16 paired seeds each (`python -m experiments.feedback_grid`, which also prints
  the next two bullets). Claim fixed before the run: showing feedback never lowers
  engagement if at most 4 of the 96 person × target cells fall below zero beyond 2 SE and
  none beyond 3 SE. **It holds: 0 cells below zero beyond 2 SE**, 87 of 96 positive. The
  per-person claim that it helps beyond 2 SE at every target where the person starts at
  least 10% of the time held for 1 of 16; 16 seeds per cell are too few. Pooled over
  targets (descriptive), it raised P(start) beyond 2 SE for 13 of 16 people, by 0.015 to
  0.081; the other three have `k = 1.5` and barely start.
- **By difficulty, over all 16 people** (256 paired runs per target): P(start) shown −
  hidden was +0.031, +0.034, +0.027, +0.039, +0.050, +0.044 at 90, 75, 60, 45, 30, 20%,
  each beyond 2 SE. Total reward shown − hidden was +1.0 ± 0.5, +1.4 ± 1.0, −0.2 ± 1.2,
  +2.6 ± 1.9, **+7.7 ± 2.1**, **+8.1 ± 1.8**: with feedback shown, people earn more at
  hard targets, where the one-person run above could not tell.
- **Lockout (2026-10-15, `python -m experiments.lockout`).** Lockout is a run in which the
  person almost never starts again (start rate below 0.10 over 300 decisions, defined
  before the run). Four
  effort-averse people (f, k = 0.6/1.0, 0.6/1.2, 1.0/1.2, 1.0/1.5), targets 75, 60 and 45%,
  64 paired seeds each. Claims fixed before the run: pooled, lockout is more common with
  feedback hidden: **holds narrowly**, 11.6% against 9.1% of 768 runs, +0.025 ± 0.011
  (2.3 SE). Per person: holds for the two with `f = 1.0`, not for the other two, and
  `f = 0.6, k = 1.2` went the other way within 2 SE. A 16-run pattern that suggested
  feedback protects that person from lockout did not replicate. In runs that did not lock
  out, showing feedback raised the start rate by 0.07 to 0.08 for all four (descriptive).
- **f and k set the two ends of the engagement curve** (same runs, feedback shown). At a
  90% target P(start) is 0.03 to 0.43 for `f = 0.2` and 0.53 to 0.84 for `f = 1.5`: easy
  questions lose low-`f` people most. At 20% it is 0.51 to 0.88 for `k = 0.2` and 0.02 to
  0.07 for `k = 1.5`. The best target ranges from 30% to 75% across the 16 people; single
  peaks are noisy at 16 seeds, the two ends are not.
- **No crossover.** Hiding feedback never won beyond 2 SE on either measure. By the reward
  rule above the sign flip counts as "suggested", but it is not the predicted one. The
  earlier result that pointed to a flip at 30% is in `RETIRED.md`.

### The controllers

- **It found the redo prompts on its own.** The redo decision is where cost varies most
  (`0.15 · min(1/p, 8)`, about 4× across difficulties), and the controller chose a
  redo-prompt setting on 93% of its updates. Random probing over the same 96 settings
  picked one 11% of the time and saw about 5 redo prompts per person
  (`python -m experiments.controllers`).

**Measuring costs no detectable reward (2026-10-11).** Four ways of choosing settings were
run on the same 32 simulated people: a 4 × 4 grid of f and k in {0.2, 0.6, 1.0, 1.5}, 2
seeds each. Each run had 300 start decisions, the question-type task, and skips on, and
the controllers never saw the true f and k. Settings: γ = 0.9, `ACHIEVEMENT_SCALE = 4`,
continuing bootstrap, `T_DUR = 7.0`. Pass conditions were fixed before the run.
`python -m experiments.controllers` prints this table and the three after it. It fits
every arm at 100, 200 and 300 decisions, and its fits at 300 match the original run's to
within 0.01.

| how settings were chosen | f RMSE | k RMSE | total reward | P(start next set) | redo settings chosen |
|---|---|---|---|---|---|
| fixed default | 0.093 | 0.310 | 60.0 | 0.64 | 0% |
| for reward, 84 scheduled-review settings (the app's controller until 2026-10-11) | 0.054 | 0.294 | 147.0 | 0.76 | 0% |
| for reward, all 96 settings | 0.056 | 0.223 | 142.7 | 0.75 | 35% |
| for information, all 96 (`InfoController`) | **0.038** | **0.046** | **152.4** | 0.75 | 93% |

**Over time (2026-10-15).** The same people and seeds, fitted at 100, 200 and 300 start
decisions. The spread of
cost is the standard deviation of cost across the decisions collected so far.

| how settings were chosen | k RMSE at 100 / 200 / 300 | f RMSE at 100 / 200 / 300 | sd of cost at 300 |
|---|---|---|---|
| fixed default | 0.575 / 0.622 / 0.310 | 0.152 / 0.141 / 0.093 | 0.134 |
| random every 25 decisions, 84 settings | 0.309 / 0.247 / 0.126 | 0.103 / 0.067 / 0.071 | 0.171 |
| for reward, 84 settings | 0.471 / 0.391 / 0.294 | 0.079 / 0.066 / 0.054 | 0.112 |
| for reward, 96 settings | 0.358 / 0.346 / 0.223 | 0.079 / 0.077 / 0.056 | 0.126 |
| for information (`InfoController`) | **0.090 / 0.059 / 0.046** | **0.075 / 0.056 / 0.038** | **0.272** |

Choosing for information measures `k` better at 100 decisions than every other way does at
300, and it collects data in which cost varies about twice as much from the start. Reward
planning measures `k` worse than random settings at every checkpoint.

**What measuring costs, block by block (2026-10-15).** The same 160 runs (5 arms × 32
people), with reward and start decisions recorded at 100, 200 and 300 start decisions. No
claims were fixed before this run, so these are descriptive.

| how settings were chosen | reward 1-100 / 101-200 / 201-300 | P(start) 1-100 / 101-200 / 201-300 |
|---|---|---|
| fixed default | 34.0 / 16.8 / 9.2 | 0.70 / 0.67 / 0.56 |
| random every 25 decisions, 84 settings | 41.2 / 15.0 / 7.5 | 0.60 / 0.60 / 0.44 |
| for reward, 84 settings | 62.6 / 48.8 / 35.6 | 0.73 / 0.79 / 0.76 |
| for reward, 96 settings | 64.0 / 48.4 / 30.3 | 0.74 / 0.79 / 0.71 |
| for information (`InfoController`) | 49.0 / 48.6 / 54.9 | 0.70 / 0.76 / 0.81 |

Paired differences, information − reward planning (84 / 96 settings):

- **First 100 decisions:** reward −13.6 ± 3.4 / −15.1 ± 3.2, both beyond 2 SE. P(start)
  −0.040 ± 0.024 / −0.049 ± 0.024: the first is undetermined, the second just past 2 SE.
- **101-200:** reward −0.2 ± 4.7 / +0.2 ± 4.4; P(start) −0.032 ± 0.018 / −0.038 ± 0.014.
  Reward is undetermined; the P(start) gap against the 96-setting controller is 2.7 SE.
- **201-300:** reward +19.3 ± 6.3 / +24.6 ± 6.5, both beyond 2 SE. P(start) +0.049 ±
  0.031 / +0.102 ± 0.037: the first is undetermined, the second 2.8 SE.

So measuring does cost something, early: about 14 reward and 4 to 5 points of P(start)
in the first 100 decisions. It is repaid by the third block, where the reward
planners' engagement and reward fall off and measuring's keep rising. The totals over
300 decisions (above) hide this by netting the two.

Paired differences over the 32 people (same people and seeds); a difference counts only
beyond 2 SE:

- **P1 passes.** Squared error in k, measuring − reward planning (96 settings), was
  −0.0476 ± 0.0231. That is just past 2 SE (0.0462), so it is a narrow pass. For f it was
  −0.0017 ± 0.0013, undetermined, so f is not worse. Against the app's previous controller
  the k gain is clearer: −0.0841 ± 0.0327 (2.6 SE).
- **P2 passes.** Reward, measuring − fixed default: +92.4 ± 9.1.
- **Over the whole run, measuring cost no detectable reward.** Measuring − reward planning:
  +9.7 ± 10.2, and measuring − previous controller: +5.5 ± 10.1. Both are undetermined, and
  both point estimates favor measuring. Block by block it does cost reward early and
  repay it late (above).
- **Fixing the reward controller's rollouts changed little.** Reward, 96 − 84 settings:
  −4.3 ± 3.7. k squared error: −0.0365 ± 0.0399. Both are undetermined. Given redo
  prompts it used them on 35% of updates (28 redo decisions per person, against 338 for
  measuring).
- **Reward planning leaves k poorly identified.** At 0.22 to 0.29 RMSE, k is barely
  better than with the fixed default (0.31). k came out below 0.05 for 3 of 32 people
  under the previous controller and 1 under the 96-setting one, all with true k = 0.2.
  None did under measuring. That fits the reason the measuring controller exists: reward
  planning settles on a setting, so cost stops varying. The spread of cost in "Over time"
  bears it out: 0.11 to 0.13 under reward planning at 300 decisions, against 0.27 under
  measuring.

### Recovery

**Recovery from one person (2026-10-10, `python -m experiments.recovery_sweep`).** One
simulated person, mid cold-start cost, scheduled review, 1200 recorded decisions per fit
split across four learners after 0, 200, 500 and 1000 warm-up decisions, single-`diff`
task, 4×4 grid × 3 reps (48 fits), 95% bootstrap intervals over the fits:

| f RMSE | k RMSE |
|---|---|
| **0.035** [0.028, 0.042] | **0.034** [0.025, 0.043] |

- **The pass condition is met.** It was fixed on 2026-09-13: `f` and `k` RMSE ≤ 0.05, the
  step the controller rounds its estimates to (`snap_value`), so an estimate off by about
  one step plans like the truth or its neighbour. Both upper bounds are under 0.05.
- **The worst grid points are effort-averse people who rarely start**: `f` RMSE 0.067 at
  f 0.6, k 1.5, and `k` RMSE 0.067 at f 0.2, k 1.5. The next worst for `k` is a low-`k`
  person (0.059 at f 1.0, k 0.2).

**The recovery pipelines (2026-10-20, seeded per grid point).** The same grid;
`python -m temporal recovery --mode MODE --reps R --out results/pipelines`:

| pipeline | f: r | f: RMSE | k: r | k: RMSE |
|---|---|---|---|---|
| **single**, one continuous learner, 1200 decisions (3 reps) | 0.997 | 0.037 | 0.998 | 0.033 |
| **coupled**, two stacked learners at different cold-start costs, 700 decisions (3 reps) | 0.998 | 0.031 | 0.998 | 0.030 |
| **uncoupled**, value and cost set by design, 400 per condition (5 reps) | 0.999 | 0.026 | 0.999 | 0.023 |

Use `single` (or the four-learner design above) when modelling one real participant, and
`coupled` or `uncoupled` when the experimenter controls the design. A single learner
generates its own value and cost, correlated and confined to what it happens to
experience, so it recovers less well than a designed contrast.

**Prediction error is a large part of the variation in value (2026-10-20,
`python -m experiments.value_variation`).** On the question-type task, where the target
holds `p` away from 1 and `δ` stays alive, the carried error's standard deviation is 59%
of value's (46% to 76%). Eight fits, 700 decisions each, the four people of the
persistence test × 2 seeds, settings switched by random probes: `f` RMSE 0.045, `k` RMSE
0.058, value sd 0.31 to 0.40, `corr(value, cost)` −0.50 to −0.22 (see "How value and cost
together decide what can be measured"). On the single-`diff` task a learner masters the
fixed difficulty, so `p → 1` and `δ` shrinks.

**Skips do not help estimate `f` and `k` (2026-10-15).** Twelve simulated people (4
profiles × 3 reps) who can give up on questions, 300 start decisions each (human-sized
data), settings switched at random (`python -m temporal recovery --mode persistence --reps 3`):

| fitted from | f RMSE | k RMSE |
|---|---|---|
| decisions only | **0.045** | **0.083** |
| answer and skip times only | 0.597 | 0.432 |
| both, jointly | 0.062 | 0.133 |

Adding answer and skip times to the decisions makes both parameters worse, and on their
own they cannot place `f`. A question's value, `p(1 − p)`, hardly varies across
questions, while its cost, `1/p`, varies about 4×, so skip times mostly measure `k`, and
the redo prompts already carry that. So the controller fits `f` and `k` from decisions,
and uses time on questions only for patience.

**What each part of the fit contributes, alone and together (2026-10-15,
`python -m experiments.fit_factorial`).** The 32 people of "Planning to measure" (4×4
grid × 2 seeds), question-type task, settings chosen by `InfoController`, 300 start
decisions, with a true non-decision time of 0.3 s and true patience 10 s. Each dataset is fitted 8 ways: timed-out trials kept or dropped, `T0`
fitted or pinned at 0, answer and skip times added or not (at a fixed 10 s patience,
their best case). Claims fixed before the run: a choice matters alone if switching only
it worsens squared error beyond 2 SE; two interact if the bias of switching both differs
from the sum of switching each, beyond 2 SE.

| timeouts | T0 | times | f RMSE | k RMSE | f bias | k bias |
|---|---|---|---|---|---|---|
| **kept** | **fitted** | **no** | **0.054** | **0.047** | −1.9% | −1.2% |
| dropped | fitted | no | 0.068 | 0.127 | +5.7% | +12.8% |
| kept | pinned | no | 0.282 | 0.160 | −22.8% | −16.8% |
| kept | fitted | yes | 0.064 | 0.074 | −3.5% | −5.4% |
| dropped | pinned | no | 0.274 | 0.129 | −16.8% | −5.5% |
| dropped | fitted | yes | 0.058 | 0.082 | +2.8% | +1.9% |
| kept | pinned | yes | 0.278 | 0.161 | −22.8% | −16.3% |
| dropped | pinned | yes | 0.274 | 0.139 | −17.7% | −9.8% |

- **Pinning `T0` costs the most**: `f` error about fivefold, both parameters 17 to 23% low,
  whatever else is done. Dropping timed-out trials mainly hurts `k` (beyond 2 SE for `k`,
  not `f`). Adding times worsens `f` beyond 2 SE, slightly.
- **Every pair interacts beyond 2 SE, but by 1 to 7 percentage points of bias**, small
  next to the single effects, and mostly by cancellation: dropping timeouts pushes the
  estimates up, pinning `T0` and adding times push them down. Dropping timeouts and adding
  times together leaves almost no mean bias (+2.8%, +1.9%) while `k` error is still 0.082
  against 0.047. Report error, not only bias.
- **The pipeline holds with a real non-decision time present**: `T0` recovered at
  0.31 ± 0.03 s; 8.7% of decisions timed out.

**Patience can be recovered, and errors in `f` do not spoil it (2026-10-15).** Twelve
simulated people (4 profiles × 3 reps), 300 start decisions each, settings switched at
random. `f` and `k` were fitted from decisions; patience was then fitted from time on
questions, once with the fitted `f` and `k` and once with the true ones (`python -m
temporal recovery --mode patience --reps 3`):

| true patience | fitted with fitted `f`, `k` (3 reps) | give-ups per person |
|---|---|---|
| 2 s | 2.0, 2.2, 2.6 s | 153 to 182 |
| 5 s | 5.4, 5.7, 5.7 s | 35 to 50 |
| 10 s | 11.8, 12.0, 18.6 s | 13 to 18 |
| 20 s | 12.6, 20.5, 27.9 s | 4 to 6 |

Patience was typically off by a factor of 1.31 with fitted `f` and `k`, against 1.28 with
the true ones, so errors in `f` and `k` add little. Someone who gives up at a glance (2 s)
was never mistaken for a patient person (10 s). With no give-ups patience is not
identified, so the controller keeps the 10 s default until there is one.

### Censored trials are data

PyDDM's likelihood is not renormalized over the decided region. Fit only the trials
that reached a boundary and the optimizer inflates drift at no likelihood cost,
shrinking the predicted undecided mass. The correction is to feed the censored counts
back in as undecided trials. On the recovery design (48 fits of 1200 decisions, 4×4
grid × 3 reps, four learners across the learning curve, `T_DUR = 7.0`, continuing
bootstrap, 2026-10-10, `python -m experiments.recovery_sweep`), 6,032 of 57,600
decisions (10.5%) timed out. Kept as undecided, the mean bias was −0.2% for `f` and
−0.5% for `k` (RMSE 0.035, 0.034). Dropped, it was +12.9% and +10.6% (RMSE 0.099, 0.099).

### Non-decision time

`python -m temporal recovery --mode t0 --reps 3`, 2026-10-20, seeded. One learner per
run, 1200 decisions, 3 reps per value, true `f = k = 0.8`, current model. Each dataset is fitted twice: with `T0` pinned at 0 and
with `T0` free. Means over reps:

| true T0 | f pinned | k pinned | f free | k free | T0 recovered |
|---|---|---|---|---|---|
| 0.0 s | 0.81 | 0.79 | 0.81 | 0.79 | 0.02 |
| 0.2 s | 0.78 | 0.75 | 0.86 | 0.82 | 0.21 |
| 0.3 s | 0.69 | 0.71 | 0.79 | 0.80 | 0.34 |
| 0.5 s | 0.66 | 0.67 | 0.81 | 0.80 | 0.47 |

- **Pinning it biases both parameters down, more as the true value grows**: 3 to 6% at
  0.2 s, 11 to 14% at 0.3 s, 16 to 18% at 0.5 s.
- **Freeing it removes most of the bias.** `f` and `k` land within 0.02 of the truth at
  0.0, 0.3 and 0.5 s. At 0.2 s `f` comes out 0.86. `T0` itself lands within 0.04.
- **It costs almost nothing when there is none.** With a true 0 the free fit returns 0.02.
- **It is free where real people are fitted**: the controller's estimate and the
  interface's "fit my parameters". The recovery pipelines pin it, since their simulated
  people have none.

### The person may not share the model's constants

The fit holds the discount factor γ at 0.9, the TD learning rate α at 0.05, `SIGMA` at 0.7 and `BOUND` at 1.5. Here each
simulated person differs from the model in one of the four, and the fit isn't told.

γ and α never enter the fit directly. They set the person's learned value, and the app
computes value with the defaults. To match that, a second TD learner at the defaults runs
on the same rewards as the person, and the analyst fit uses its value; this is what real
data would get. The oracle fit uses the person's own value instead, so the difference
between the two fits is the cost of a wrong γ or α. `SIGMA` and `BOUND` change only the
person's own start, redo and give-up draws. The controller's rollouts and every fit keep
the pinned values.

The setup is the app's: question types with Elo ratings, skips on, and `InfoController`
steering from the default setting. Each run has 300 start decisions plus their redo
decisions. People (f, k) = (0.8, 0.8), (1.2, 0.4) and (0.6, 1.2), 5 seeds each, with the
same seeds at every level (2026-10-21, `python -m experiments.misspecification`). Bias of
the analyst fit is (estimate − true) / true, mean ± SE of 15 fits:

| person | bias in f | bias in k | bias in f / k | undecided |
|---|---|---|---|---|
| all defaults | −2 ± 1% | +1 ± 1% | −2 ± 1% | 6.7% |
| γ = 0.5 | −54 ± 5% | +81 ± 20% | −68 ± 6% | 3.4% |
| γ = 0.7 | −47 ± 6% | +48 ± 9% | −60 ± 6% | 5.0% |
| γ = 0.8 | −41 ± 4% | +15 ± 2% | −49 ± 3% | 7.9% |
| γ = 0.95 | +51 ± 2% | −3 ± 2% | +56 ± 4% | 6.1% |
| γ = 0.99 | +145 ± 5% | −9 ± 2% | +173 ± 10% | 6.9% |
| α = 0.01 | −34 ± 2% | −10 ± 6% | −20 ± 9% | 8.1% |
| α = 0.025 | −15 ± 1% | −7 ± 2% | −8 ± 3% | 7.7% |
| α = 0.1 | +8 ± 1% | +6 ± 2% | +2 ± 2% | 6.0% |
| α = 0.2 | +11 ± 1% | +8 ± 2% | +3 ± 2% | 6.2% |
| `SIGMA` = 0.5 | −2 ± 2% | −3 ± 1% | +0 ± 2% | 11.0% |
| `SIGMA` = 0.6 | +1 ± 1% | −1 ± 1% | +2 ± 1% | 9.3% |
| `SIGMA` = 0.85 | +3 ± 2% | +1 ± 2% | +2 ± 1% | 3.1% |
| `SIGMA` = 1.0 | +6 ± 2% | +2 ± 2% | +4 ± 3% | 1.3% |
| `BOUND` = 1.0 | +61 ± 1% | +67 ± 4% | −3 ± 2% | 0.8% |
| `BOUND` = 1.25 | +20 ± 2% | +23 ± 2% | −2 ± 1% | 3.0% |
| `BOUND` = 1.75 | −15 ± 1% | −17 ± 1% | +2 ± 1% | 11.3% |
| `BOUND` = 2.0 | −25 ± 1% | −28 ± 1% | +3 ± 1% | 15.5% |

`BOUND` and `SIGMA` are constants of the accumulator. A wrong value for either changes
`f` and `k` by about the same proportion, so the ratio `f / k` stays close to the truth.
γ and α are constants of the learner. They change the value that goes into the drift,
and the ratio does not stay close.

A bound of 1.0 instead of 1.5 makes both `f` and `k` come out about 60% too high, but
`f / k` stays within 3% of the truth at every bound from 1.0 to 2.0. The share of
undecided decisions also changes with the bound, from 0.8% at 1.0 to 15.5% at 2.0, so
that share can be used to check the pinned bound against real data. `SIGMA` has little
effect: from 0.5 to 1.0, `f`, `k` and their ratio all stay within 6%, although the share
of undecided decisions falls from 11.0% to 1.3%.

γ has the largest effect. `f / k` comes out 49% too low at γ = 0.8 and 56% too high at
0.95. The oracle fit is within 5% at every γ, so the whole error comes from computing
value with γ = 0.9. A person who discounts more steeply than 0.9 looks less sensitive to
reward and more averse to effort than they are. A person who discounts less looks much
more sensitive to reward. This puts a size on the limitation that `f` is confounded with
the discount factor (see "Known limitations"). `k` is least precise at low γ, because
such a person starts few sets (12% of start decisions at γ = 0.5) and sees few redo
prompts (24, against 314 at the defaults).

A wrong learning rate mostly changes `f`: −34% at α = 0.01 and +11% at 0.2. The ratio
moves at slow rates (−20 ± 9% at 0.01, −8 ± 3% at 0.025) and stays within 3% at fast
ones.

The same command also runs the single-`diff` task of the recovery pipelines (one learner,
1200 decisions, no redo decisions). The pattern is the same with one exception: a high γ
pushes `k` toward 0. At γ = 0.95, `k` comes out 28 ± 6% too low; at 0.99 it is fitted as 0
and `f` comes out 236 ± 21% too high. A low γ biases `f` more on this task (−75 ± 3% at
0.5) and `k` less (+17 ± 3%). For the ratio, `BOUND` and `SIGMA` keep `f / k` within 4%,
γ moves it (−52 ± 1% at 0.8), and α moves it by at most 8%. The likely reason `k` holds up
on the app's task is the redo decisions, whose cost varies more than at any other
decision. All of these fits used γ = 0.9, `ACHIEVEMENT_SCALE = 4`, the continuing bootstrap
and `T_DUR = 7.0`.

### How value and cost decide what can be measured

Correlation between value and cost, or too little variation in cost, costs **precision**
instead: more decisions for the same accuracy. Expected standard deviation from 300 start decisions,
from the Fisher information of the decisions at the true `f`, `k` (16 people × 2 seeds,
default setting for starts, a fixed redo-prompt setting for starts with redos; medians;
`python -m experiments.precision`):

| decisions in the data | SD of f | SD of k | sd of cost |
|---|---|---|---|
| start decisions, cold and warm | 0.065 | 0.140 | 0.154 |
| start decisions, warm only | 0.120 | **0.509** | 0.012 |
| starts and redos, cold and warm | 0.026 | 0.098 | 0.065 |
| starts and redos, warm only | 0.025 | **0.107** | 0.057 |

A real person gives only warm starts (opening the app is the cold decision, and its "RT"
is hours), so their start decisions alone barely measure `k`. Redo decisions, whose cost
varies with how hard the missed question is, bring it within the range of the simulated
cold-and-warm data. That is why `InfoController` leans on redo prompts (see "Planning to
measure"). The correlation of value and cost is still the number to re-check after any
change to cost or value: a lever that set cost from the person's state, as `reentry_cost`
once did, can push it toward proportional (V1 cases in `RETIRED.md`).

Carrying prediction error into value (2026-09-21) raised the correlation only slightly,
because it added more variance to value than it did shared variance with cost. On the
question-type task at the default setting, `f = k = 0.8`, 400 decisions, means over 8
seeds (2026-10-20, `python -m experiments.value_variation`):

| | corr(V(0), cost) | corr(δ, cost) | corr(value, cost) | sd of value, V(0) → with δ |
|---|---|---|---|---|
| feedback on | −0.29 | −0.11 | −0.32 | 0.221 → 0.273 |
| feedback off | −0.32 | −0.13 | −0.34 | 0.221 → 0.294 |

Both channels correlate with cost for the same reason: cost rises with review load,
review brings back questions the person already missed, and those earn less than
expected. The correlation is a property of the task, not of the carry.

### Smaller V3 measurements

- At the start node, effort cost is weak and retrospective: review load adds about 0.01
  to cost on average, with a standard deviation under 0.02 (same runs as the clip check
  below). Nearly all start-node cost variation in simulation comes from sessions
  going cold and warm, which human data never shows the fit (see Human sessions). The
  redo node's cost varies far more, which is why `InfoController` leans on it.
- The prediction error is clipped at `±DELTA_CLIP` (1.0) so `value` stays on a bounded
  grid. At the default setting it binds on 0.4 to 0.5% of start decisions with feedback
  shown and 0.7 to 0.9% with it hidden, and on no redo decisions (four people, f, k =
  0.8/0.8, 1.2/0.4, 0.6/1.2, 1.2/1.2, 2 seeds × 300 start decisions, with and without redo
  prompts; 2026-10-24).
- **Why it matters for measurement.** Under the continuing bootstrap value is larger
  (mean 0.57 against 0.22) and varies about 1.5 times as much (sd 0.27 against 0.18),
  and `f` is identified from variation in value. Re-measured under V3 on 2026-10-24: f =
  k = 0.8, default setting, 8 seeds × 400 start decisions, the same runs with
  `value.CONTINUING` switched. The V2 measurements, which put the spread at about twice
  on the single-`diff` recovery task (sd 0.222 against 0.115), and whether the switch moved the reward controller closer to the best fixed setting
  (it did not), are in `RETIRED.md`.
- **Planning is about 3× faster.** One `InfoController` plan (96 settings × 6 draws
  × 25 decisions, f 0.8, k 0.8) took 15.3 s before this change and 4.6 s after it
  (2026-10-11). Re-timed on 2026-10-24, it takes about 1.5 s.

---

## V3, superseded by later V3 measurements (also under the recording flaw)

### Recovery pipelines and non-decision time, unseeded (superseded 2026-10-20)

The `single`, `coupled`, `uncoupled` and `t0` sweeps drew their datasets from whatever
state the random generators were in, so a re-run could not reproduce them. On 2026-10-20
each dataset got its own seed (`recovery.seed_point`) and the four were re-run, and the
README now holds the seeded numbers. The model, grid and reps are the same, so the
differences are sampling. The 2026-10-15 numbers:

| pipeline | f: r | f: RMSE | k: r | k: RMSE |
|---|---|---|---|---|
| single (3 reps) | 0.995 | 0.049 | 0.996 | 0.044 |
| coupled (3 reps) | 0.998 | 0.030 | 0.997 | 0.035 |
| uncoupled (5 reps) | 0.999 | 0.026 | 0.998 | 0.027 |

| true T0 | f pinned | k pinned | f free | k free | T0 recovered |
|---|---|---|---|---|---|
| 0.0 s | 0.77 | 0.80 | 0.78 | 0.81 | 0.04 |
| 0.2 s | 0.73 | 0.75 | 0.81 | 0.81 | 0.22 |
| 0.3 s | 0.69 | 0.71 | 0.79 | 0.79 | 0.28 |
| 0.5 s | 0.64 | 0.66 | 0.78 | 0.79 | 0.48 |

That run read the table as "freeing `T0` puts `f` and `k` within 0.03 of the truth at
every value". The seeded run does not bear that out: at a true 0.2 s, `f` came out 0.86.

### How much of value is prediction error, and the carry table (superseded 2026-10-20)

These were measured by hand around 2026-09-21. No script was kept, and the people,
seeds and setting were not recorded. `python -m experiments.value_variation` replaces
them and states all three, so the README's numbers can be re-run. It does not reproduce
"about 80%": in the new run the carried error has 59% of value's standard deviation (46%
to 76%). The old figures:

- `δ` supplied about 80% of the standard deviation of value. Eight fits, 700 decisions,
  four `(f, k)` pairs × 2 seeds, random probes: `f` RMSE 0.049, `k` RMSE 0.027, value sd
  0.35 to 0.41, `corr(value, cost)` −0.22 to −0.50.
- At a held target, 400 decisions:

| | corr(V(0), cost) | corr(δ, cost) | corr(value, cost) | sd of value |
|---|---|---|---|---|
| feedback on | −0.16 | −0.14 | −0.24 | 0.178 → 0.228 |
| feedback off | −0.30 | −0.23 | −0.40 | 0.156 → 0.223 |

### Feedback shown against hidden, one person (superseded 2026-10-15)

Replaced by the README's "Which difficulty and whether to show feedback" (16 and 96
paired seeds, six targets, from a fresh person). This run planned from an unsaturated
measured state, which the new run does not, so the two are not paired. One person
(`f` 0.8, `k` 0.8), 96 paired seeds × 300 decisions, review 0.25:

| target chance right | feedback on | feedback off | paired difference |
|---|---|---|---|
| 90% | 45.8 | 44.0 | +1.75 ± 0.52 (+3.4 SE), feedback on |
| 30% | 63.4 | 64.2 | −0.79 ± 1.16 (−0.7 SE), undetermined |

At 32 seeds over 200 decisions the 30% target reached −3.90 ± 2.19 (−1.8 SE). These were
read as a suggested sign flip (hide feedback from someone missing most questions). The
2026-10-15 run found no flip: feedback shown was ahead or undetermined at every target,
and ahead beyond 2 SE at 20%.

### Single-`diff` recovery after the reaction terms were removed (superseded 2026-10-10)

One learner, 4×4 × 3 reps, 1200 decisions, continuing bootstrap, `T_DUR = 7.0`:
`f` RMSE 0.054 → 0.038, `k` 0.033 → 0.034, value sd 0.222 → 0.287, before → after the
change. Unpaired (different seeds). The "before" is V2. Replaced by the README's
recovery table (the reseeded design, four learners across the learning curve, run on
2026-10-10: `f` 0.035, `k` 0.034).

### Skips and `f`, `k` (superseded 2026-10-15)

Twelve people (4 profiles × 3 reps), 300 start decisions, settings switched at random.
The model version of this run is uncertain (after 2026-09-21, before the rerun), so it was
re-measured. The "was" column is the same test under the V2 five-parameter fit, before
patience and lapses existed, at a fixed 10 s, with one person defined by reaction terms.

| fitted from | f RMSE | k RMSE |
|---|---|---|
| decisions only | 0.049 (was 0.380) | 0.099 (was 0.160) |
| answer and skip times only | 0.517 (was 1.238) | 0.396 (was 0.475) |
| both, jointly | 0.066 (was 0.376) | 0.123 (was 0.165) |

The conclusion is unchanged under the rerun. The "was" column showed thin data is where
dropping three parameters pays: `f` RMSE on human-sized data went from 0.380 to 0.049.

### Patience recovery (superseded 2026-10-15)

Same design, model version uncertain. The rerun gives up far more often per person
(153 to 182 at 2 s, against 24 to 42 here), so these runs did not share a model.

| true patience | fitted, range over 3 reps | give-ups per person |
|---|---|---|
| 2 s | 2.5 to 2.8 s | 24 to 42 |
| 5 s | 3.8 to 5.4 s | 26 to 66 |
| 10 s | 9.1 to 16.4 s | 7 to 54 |
| 20 s | 14.9 to 24.9 s | 2 to 15, and one person with none |

Factor 1.30 with fitted `f` and `k`, 1.26 with true ones; with the person who never gave
up included, 1.50.

---

## V2 (`T_DUR = 7.0`, continuing bootstrap, reaction terms still in the drift; recording flaw)

### Single-learner recovery and its pass condition

1200 recorded decisions per fit, 4×4 grid × 3 reps, `T_DUR = 7.0`, 95% bootstrap
intervals over the 48 fits:

| pipeline | f RMSE | k RMSE | value sd |
|---|---|---|---|
| one learner, terminal bootstrap | 0.079 [0.059, 0.098] | 0.029 [0.024, 0.033] | 0.115 |
| one learner, continuing bootstrap | 0.054 [0.042, 0.064] | 0.033 [0.027, 0.037] | 0.222 |
| four learners across the learning curve, continuing | 0.049 [0.036, 0.061] | 0.041 [0.034, 0.049] | 0.230 |

- Reseeding against one continuous learner, both continuing: `f` −0.005 [−0.015, +0.006],
  `k` +0.009 [0.000, +0.018].
- Continuing against terminal bootstrap, one learner: `f` −0.026 [−0.045, −0.006].
- Errors concentrated at `f = 1.5` (per-point `f` RMSE 0.03 to 0.11).

The original pass condition, `f` RMSE near 0.035, belonged to a design where the
experimenter sets value and cost orthogonally (the uncoupled pipeline reached 0.030 that
way) and was retired on 2026-09-13. Its replacement, fixed before its test reps, was
`f` and `k` RMSE ≤ 0.05 on 3 fresh reps (one person, mid cold-start cost, scheduled
review, continuing bootstrap, `T_DUR = 7.0`, 1200 decisions split across learners after
0, 200, 500 and 1000 warm-up decisions). It passed: `f` 0.045 [0.037, 0.053], `k` 0.042
[0.030, 0.057]; worst at (f 0.2, k 1.5) 0.065 and (0.6, 1.5) 0.061. The same condition is
met under V3 (README, Current results).

### Value bootstrap: what the switch did

At `f = k = 0.8`, `T_DUR = 7.0`, 600 decisions: `V(0)` settled near 0.17 (continuing fixed
point `γ³r/(1−γ⁴)` ≈ 0.15) against 0.07 under the terminal bootstrap, and the same person
played 268 sets instead of 146. The fixed default earned more under the continuing
bootstrap: 54.6 → 74.5 for f 0.8, k 0.8, and 16.3 → 30.0 for f 0.6, k 1.2.

Whether the continuing bootstrap brings the reward controller (given true parameters)
closer to the best fixed setting. Same five people, 8 seeds, `T_DUR = 7.0`, best fixed
setting picked on held-out seeds, ratio of mean rewards:

| person | terminal | continuing | continuing − terminal |
|---|---|---|---|
| f 0.8, k 0.8 | 77.8% ± 8.9% | 95.1% ± 12.8% | +17.3% ± 15.7%, undetermined |
| f 1.2, k 0.4 | 77.3% ± 10.3% | 74.2% ± 10.2% | −3.1% ± 14.5%, undetermined |
| f 0.6, k 1.2 | 103.0% ± 32.2% | 90.8% ± 22.1% | −12.2% ± 39.3%, undetermined |
| put off by visible misses | 82.5% ± 15.9% | 83.1% ± 6.6% | +0.6% ± 17.2%, undetermined |
| spurred on by visible misses | 71.8% ± 19.7% | 81.4% ± 7.9% | +9.6% ± 21.2%, undetermined |
| mean over the five | 84.3% ± 8.7% | 85.4% ± 5.9% | +1.0% ± 10.5%, undetermined |

It did not. Under the continuing bootstrap the gap to the best fixed setting was beyond
2 SE for three people: −67.2 ± 27.4 (f 1.2, k 0.4), −40.5 ± 17.3 (put off), −44.6 ± 19.7
(spurred on). The last two people need the reaction terms and cannot be rebuilt.

### Does the reward controller help? (8 seeds, terminal bootstrap)

Five simulated people measured like a human (Elo-rated question types, controller never
sees the truth, patience 10 s, leaving from the first question after coming back), 300
start decisions, 8 seeds, `T_DUR = 7.0`, held to the terminal bootstrap. Two of the five
people are defined by `b_seen` / `b_unseen`. Total reward, mean ± SE over seeds:

| person | fixed default | random switching | controller | controller given true parameters | best fixed setting* |
|---|---|---|---|---|---|
| f 0.8, k 0.8 | 54.6 ± 2.5 | 58.8 ± 6.6 | 148.8 ± 19.3 | 140.9 ± 20.0 | 181.2 ± 13.5 |
| f 1.2, k 0.4 | 68.3 ± 5.1 | 83.9 ± 4.5 | 188.2 ± 27.8 | 194.1 ± 29.8 | 251.2 ± 5.0 |
| f 0.6, k 1.2 | 16.3 ± 4.4 | 18.0 ± 5.7 | 38.2 ± 11.8 | 39.1 ± 9.7 | 37.9 ± 11.7 |
| f 0.8, k 0.8, put off by misses they see (`b_seen` −1.0, `b_unseen` +0.8) | 8.4 ± 1.4 | 24.7 ± 9.1 | 37.5 ± 16.2 | 195.9 ± 17.4 | 237.6 ± 33.7 |
| f 0.8, k 0.8, spurred on by misses they see (`b_seen` +0.8, `b_unseen` −1.0) | 63.4 ± 4.7 | 26.1 ± 8.2 | 179.2 ± 23.3 | 170.5 ± 29.8 | 237.6 ± 33.7 |

\* The best of the 84 settings held fixed, picked on 8 other seeds and scored on these;
for every person review 0, skipped questions dropped, and a 45% or 60% target.

| person | controller − fixed default | controller − random switching | controller − given true parameters |
|---|---|---|---|
| f 0.8, k 0.8 | +94.3 ± 19.3, win | +90.1 ± 22.2, win | +7.9 ± 10.6, undetermined |
| f 1.2, k 0.4 | +119.9 ± 27.7, win | +104.2 ± 28.4, win | −5.9 ± 5.8, undetermined |
| f 0.6, k 1.2 | +21.8 ± 12.3, undetermined | +20.2 ± 12.5, undetermined | −0.9 ± 14.7, undetermined |
| put off by visible misses | +29.1 ± 14.9, undetermined | +12.7 ± 20.8, undetermined | −158.4 ± 22.8, loss |
| spurred on by visible misses | +115.8 ± 25.8, win | +153.1 ± 25.9, win | +8.7 ± 10.1, undetermined |

Also from this run: the put-off person's controller hid feedback on 14% ± 6% of decisions
with the fit and 76% ± 7% given true parameters; given true parameters it reached about
77% of the best fixed setting for two people (−40.3 ± 17.9, −57.1 ± 27.0); skip shares
ran 1.3% to 14.0% under the controller and 4.0% to 17.8% under random switching, not
compared under the rule; a quarter of runs ended on a random probe. It replaced a 2-seed
table at `T_DUR = 3.0` whose "four of five", "84 to 99% of the best fixed setting" and
"probes cost reward" claims did not survive. The reward controller is no longer what the
app runs (README, "Planning to measure").

### Non-decision time pinned at 0 (V2 drift)

One learner, 1200 decisions, 3 reps per value, `T_DUR = 7.0`, continuing bootstrap,
true `f = k = 0.8`, fitted with `T0` pinned and free:

| true T0 | f pinned | k pinned | f free | k free | T0 recovered |
|---|---|---|---|---|---|
| 0.0 s | 0.82 | 0.80 | 0.83 | 0.81 | 0.03 |
| 0.2 s | 0.74 | 0.74 | 0.80 | 0.80 | 0.23 |
| 0.3 s | 0.73 | 0.72 | 0.82 | 0.80 | 0.31 |
| 0.5 s | 0.69 | 0.68 | 0.82 | 0.80 | 0.48 |

Pinning cost 8 to 15%; freeing it removed the bias. An earlier version at `T_DUR = 3.0`,
terminal bootstrap, pinned fits only, reported −9.9% to −15.5%.

### What the longer deadline did to simulated people

At `f = k = 0.8`, terminal bootstrap: 141 sets in 400 start decisions at 7.0 s against 51
at 3.0 s. Someone bored by easy wins: 109 sets at a 90% target and 170 at 60%, against
49 and 74. With the continuing bootstrap: 243, 195 and 250; the boredom effect shrank
from 36% fewer sets at a 90% target to 22%.

### Reaction-term recovery

On a balanced design (every miss × feedback × cost combination equally often) 288
decisions recovered the signs at about a quarter of the true sizes, and 720 recovered all
five. Measured like a human (one person, Elo-rated question types, settings switched at
random every 25 decisions, four people × 3 reps, 1200 decisions): RMSE `f` 0.083, `k`
0.048, `b_seen` 0.077, `b_unseen` 0.060, `b_fb` 0.033; all 36 reaction estimates had the
right sign. Retired with the terms on 2026-09-21: they had no mechanism behind them.

---

## V1 (`T_DUR = 3.0`, terminal bootstrap; recording flaw)

### Recovery pipelines

| pipeline | f: r | f: RMSE | k: r | k: RMSE |
|---|---|---|---|---|
| single, one learner (3 reps) | 0.986 | 0.088 | 0.997 | 0.036 |
| coupled, two stacked learners (3 reps) | 0.994 | 0.055 | 0.998 | 0.029 |
| uncoupled, value as a design constant (5 reps) | 0.998 | 0.030 | 0.998 | 0.029 |

The figure of this run, `assets/initiation_recovery_coupled.png`, was redrawn under V3 on
2026-10-15; the V1 version is in git history.

### The hard-right-answers reward against fraction right

| pipeline | f RMSE before → after | k RMSE before → after |
|---|---|---|
| single | 0.058 → 0.088 | 0.059 → 0.036 |
| coupled | 0.049 → 0.055 | 0.032 → 0.029 |

Sets played in 400 decisions, `f = k = 0.8`, 4 seeds:

| reward | target 90% | 75% | 60% | 45% |
|---|---|---|---|---|
| fraction right (old) | 120 | 97 | 66 | 41 |
| hard right answers, scale 1 | 24 | 24 | 24 | 24 |
| hard right answers, scale 4 | 46 | 60 | 86 | 106 |

Unscaled, value sat about 4× below cost and every simulated person gave up whatever the
difficulty. With the old fraction-right reward, the same learner, settings and seeds
reproduced the pre-refactor result exactly (246.8 sets per learner).

### Which difficulty earns most (superseded 2026-10-15 under V3)

Total reward peaked mid-range for all five people (400 decisions, 4 seeds):

| person | peak target | total reward at 90% / 60% / 45% / 30% / 20% |
|---|---|---|
| f 0.8, k 0.8 | 45% | 22 / 62 / 88 / 69 / 46 |
| f 1.2, k 0.4 | 30% | 42 / 89 / 113 / 138 / 123 |
| f 0.6, k 1.2 | 45% | 9 / 14 / 16 / 10 / 7 |
| f 0.4, k 0.4 | 45% | 28 / 65 / 94 / 77 / 52 |
| f 1.0, k 1.0 | 45% | 21 / 53 / 77 / 62 / 34 |

The three objectives rejected against total reward were measured here too: episodes
completed and questions answered right both picked the easiest setting for everyone;
challenge attempted (`1 − p` over every question) kept rising to a 20% target for some
people. The first reason does not hold under V3: sets played and P(start) are
among the lowest at a 90% target (README, "Which difficulty and whether to show
feedback").

### Who decides whether a missed question comes back

`f = 0.8`, 800 decisions, 5 seeds, fraction-right reward:

| k | scheduled: sets / ability / P(succ) | choice: sets / ability / P(succ) |
|---|---|---|
| 0.2 | 486 / 5.29 / 0.992 | 431 / 3.40 / 0.948 |
| 0.8 | 247 / 4.66 / 0.984 | 140 / 1.66 / 0.754 |
| 1.2 | 55 / 2.55 / 0.826 | 32 / 0.68 / 0.545 |

The effort-aversion spiral under `choice`: at `k = 1.2`, 32 sets in 800 decisions against
55 under scheduled review.

### The redo node and identification (single-`diff` task, redo on every failure)

| `diff` | redo draws | f start only | f stacked | k start only | k stacked |
|---|---|---|---|---|---|
| 0.1 | 39 | 0.051 | 0.052 | 0.109 | 0.108 |
| 0.5 | 91 | 0.096 | 0.095 | 0.048 | 0.044 |

Redo cost barely varies on that task. On the question-type task under V3 the redo node
carries most of what identifies `k` (README, "Planning to measure").

### Smaller V1 measurements

- **Fitting decided trials only** returned (1.28, 1.27) at a true (0.8, 0.8), a 60%
  overestimate. Replaced by the README's V3 measurement (+12.9% / +10.6%).
- **Before the cost restructure**, one run charged one constant cost, and `k̂` sat at 0.42
  to 0.55 against a truth of 0.80 regardless of data; `collect_coupled` pooled two
  simulations at different costs to make the contrast.
- **Value-cost correlation.** `+0.90` once collapsed the `k` estimate to zero; `−0.89`
  appeared at (f 1.5, k 1.0) under scheduled review, where recovery held. The principle
  stands (README); these numbers are V1.
- **The reward controller's rollouts.** For a person put off by misses they see, hiding
  feedback earned about 40× the reward, yet 4 × 80 rollouts measured the gain at +21 ± 12
  and the controller kept feedback on; given true parameters it finished one run at 52
  against 210 for the best fixed setting. With the long re-test it hid feedback from its
  first plan: 89 reward over 200 decisions against 2.6 at the default.
- **Lockout.** At high re-entry cost a learner recorded 0 GOs in 200 decisions.
- **Elo serving.** On a person perfect at addition and ~75% at 2-digit × 1-digit
  multiplication, 60 answers at a 75% target stopped serving addition; 11 of the last 20
  questions were 2-digit × 1-digit. Measured before the first-showing rule.

### Planning time before the 2026-10-11 speedup

Over 84 settings at `T_DUR = 7.0`, about 150 decisions of history: 21 to 27 s with known
parameters, 60 to 250 s with a refit. Replaced by the README's 4.6 s.

---

## V0 (engagement score, particle filter)

The first model scored engagement from `f`, an effort cost and a boredom rate `b`, and
estimated them with a particle filter. Recovery failed: in the last recorded sweep `b`
came back near 0.94 whatever its true value. Replaced by the drift-diffusion model in May 2026.

---

## Claims withdrawn under V4 (2026-10-28)

| claim | where it stood | replaced by | why |
|---|---|---|---|
| "`SIGMA`, `BOUND` and `T_DUR` are held fixed so they cannot absorb variance; freeing them removes the identifiability" | README, "Fixed parameters carry the identifiability"; `accumulator.py`'s docstring | README, "What can be fixed, and what fixing costs": `SIGMA` must be fixed, `BOUND` can be fitted, `T_DUR` is not a parameter | Only a common rescaling of drift, bound and noise is unobservable, so one of the three sets the unit; with the noise fixed the bound is identified, which is the standard DDM's own convention. `T_DUR` is where the analysis stops reading a decision, not something a person has. The measured test is "The bound can be fitted". |
| "`f` is identified only up to γ", as a property of the model | README, "Reporting conventions" and the first known limitation | "up to γ unless γ is fitted" | γ can be recovered from the app's decisions (README, "Fitting the discount factor": the estimate was the true γ in 60 of 60 runs, and `f` unbiased once it was fitted) |
| Fisher information "has not been used as a diagnostic", under deferred next steps | README, "Next steps" | `experiments/recovery_lhc.py` computes it for `f`, `k`, γ, `BOUND` and `T0` and checks its standard errors against the fits | built |

## Implementations that did not pan out

| implementation | model | what replaced it | why |
|---|---|---|---|
| particle filter for `f`, `k`, `b` | V0 | pyddm maximum likelihood | did not recover the parameters |
| boredom rate `b`, time-since-engagement cost | V0 | boredom as sustained `|δ| ≈ 0` (V3) | no mechanism; not recoverable |
| fraction-right reward | V1 | hard right answers `4·Σ(1 − p)/n` | made the easiest setting always the most rewarding |
| `difficulty` as a TD feature | V1 | `φ = [1, stage position]` | collinear with the intercept, ~1% of `V(0)` |
| 3-second countdown on prompts | V1 interface | prompts wait; slow choices count as undecided at fit time | logged "decisions" when nobody chose |
| `T_DUR = 3.0` | V1 | 7.0 s | discarded human choices slower than 3 s |
| terminal value bootstrap | V1 | continuing bootstrap | the task is a cycle; `f` recovered worse |
| `T0` pinned at 0 for real people | V1 | fitted in [0, 0.5] s | biased `f` and `k` 8 to 15% low |
| reaction terms `b_seen`, `b_unseen`, `b_fb` (five-parameter fit) | V1, V2 | prediction error carried in value; reward lands when shown | patched a learner that produced no mid-set error; no mechanism |
| `f` RMSE ≈ 0.035 pass condition for one learner | V2 | `f`, `k` RMSE ≤ 0.05 | belonged to a designed, not self-generated, value range |
| seeding learners across the learning curve to widen value | V2 | kept as the recovery design, not as a fix | did nothing detectable for `f` |
| joint fit of decisions with answer and skip times | V2, V3 | decisions for `f` and `k`; times for patience only | made `f` and `k` worse in simulation (under V4 it changes neither detectably, and the choice stands) |
| reward-planning `Controller` as the app's controller | V2, V3 | `InfoController` (2026-10-11) | left `k` poorly measured (RMSE 0.22 to 0.29) |
| 84 settings, rollouts always scheduling review | V3 | 96 settings, rollouts follow each setting's review mode | planned a task the person never saw |
| recording every start draw: stops after stops, and every cold start | V1 to V3 simulator | one stop ends the session; time away and the return are not recorded (V4) | a person in the app cannot give those rows, and they carried most of the start node's cost variation |
| a timed-out draw counted as not starting, or not redoing | V1 to V3 simulator | the diffusion runs on past `T_DUR` to the bound it reaches; the fit still reads the draw as undecided (V4) | in the app a slow click still carries out the choice |
| counting runs and rollouts in decisions | V1 to V3 | ticks: a start decision in a session, or a chance to come back while away (V4) | with stops unrecorded, a run counted in decisions no longer charges for time away |
| precision table rows "cold and warm" and "warm only" | V3 | one row per kind of decision, all warm (V4) | cold starts are never recorded |
