# Modeling motivation through ordinary task behavior

A computational model of task initiation. From choices, reaction times and how long
someone works on a question before giving up, it recovers how strongly a person is
pulled by reward, how strongly they are pushed away by effort, and how quickly they
give up on a question. How they react to the set they just played needs no parameter
of its own: it reaches the decision as prediction error, scaled by the reward
sensitivity.

The goal is to use these parameters to reshape the environment (how hard the
questions are for this person, how often mistakes come back, whether right/wrong
is shown, whether questions they gave up on come back) so that engaging becomes
easier than stopping.

This is an active research project. It runs on simulated people, and on human
sessions played in the app (`temporal/app/`) that are recorded across sessions. No estimate
from a real person has been validated yet.

---

## Main equation

Every time a person finishes a set, they face a small decision: start the next
one, or stop. The model formalizes that decision as

```
Drive = f · Value − k · EffortCost
Value = V(0) + Σ δ                  Σ δ: the last set's prediction error, carried
```

- `f` is reward sensitivity: how strongly the expected payoff pulls the learner in.
- `Value` is what the learner believes the next set is worth, `V(0)`, plus how the
  set just played came out against what was predicted of it.
- `k` is effort aversion: how strongly cost pushes the learner away.
- `EffortCost` is how much it takes to get going again.

The drift has only these two parameters. What the last set did to this person has no
term of its own. It is prediction error, which reaches the accumulator through `Value`
and is scaled by the same `f` as everything else. When you see that you got a question
wrong, that slot paid nothing against a value that expected something, which is a
negative `δ`. A miss you are not shown cannot produce one. "Feedback is timing, not
reward" (under "The learner") covers what this gains and what it gives up.

Drive does not produce a decision instantly. It feeds a noisy accumulator (a drift
diffusion model) that races toward one of two boundaries, GO (start the next set)
or STAY (stop), and yields both a choice and a reaction time. The reaction time is
what makes the parameters separable: choices alone cannot distinguish a high `f`
from a low `k`, but the shape of the RT distribution can.

### Giving up on a question

A person can skip a question, but only after trying it; there is no separate
"answer it or skip it" choice. While they work, the same drive runs the other way:

```
quit drift = k · cost − f · value
value = ACHIEVEMENT_SCALE · p(1 − p) / stage_amt + δ  what getting it right would add,
                                                      plus the error at the slot just left
cost  = REPEAT_COST_SCALE · min(1/p, 8)               the effort it takes
```

with `p` their chance on that question. It feeds a one-boundary accumulator (same
`SIGMA` and `BOUND`); if it reaches the bound before they reach an answer, they give
up. The time to give up then has a closed form, the inverse Gaussian (Wald)
distribution (`model/persistence.py`). Giving up is a skip, or leaving if they close the app
on that question.

- **Patience is per person.** Model time becomes seconds through `patience`, the
  seconds per unit of model time; someone who skips at a glance has a small one. It
  was a fixed 10 s at first. The first real player's skips took 1.5 to 4.2 s, which at
  10 s are 1-in-10,000 to 1-in-10,000,000 events. Their fitted patience is 1.7 s
  (−log L 73.9, against 102.2 at 10 s; 65 questions, 14 given up), with `f = k = 0.8`,
  the controller's prior, since 26 decisions are under its 40-row floor.
- **Answered questions count too.** An answer after 25 s says "kept going at least
  25 s", so it enters the likelihood through the survival function, as a right-censored
  time to give up. Every question is an observation, skipped or not.
- **2% of questions are lapses**: given up on, or kept at, for reasons the model does
  not describe, at a uniform time within 60 s. One odd skip (79 + 85, an
  83% chance for that player, dropped after 2.2 s) then cannot decide the estimate
  on its own.
- **Leaving from the first question after coming back** is giving up on it too. In
  the simulator, giving up on the first question of a set started cold abandons the
  set, and the next start decision is cold again.
- **A skip is neither right nor wrong**, so it does not move the ratings.
- **The last question carries into this one.** `δ` is the prediction error from the
  slot just left: seeing that you missed the last one makes giving up on this one more
  likely, and seeing that you got a hard one right makes it less. With feedback hidden
  there is no error to carry, so `δ` is 0 and only `p` matters.
- **It is used for patience only.** `f` and `k` come from decisions; answer and skip
  times add nothing to them (see "Skips do not help estimate `f` and `k`" under
  Current results).

---

## 1. Inference

From behavior alone, recover `f` and `k` for a specific person (`inference.fit`). How
the fit works is under "Model structure → 4. Inference"; how well it recovers them is
under "Current results".

## 2. Control (`controller.py`)

This is what the project is for: using the recovered parameters to adjust the
environment so the person keeps engaging and keeps being challenged.

The controller plans every 5 start decisions in the app (every 25 in the simulations
below). When at least 15 new decided rows have come in since its last fit, it re-fits the
person's `f` and `k` (shrunk toward a prior while data is thin) and their patience from
time on questions. It then copies the
person exactly as they are (measured ability, how hard each question type is for them,
what they still owe, their current value) and simulates 96 candidate settings forward
with the *estimated* parameters:

- a **target chance of getting a question right**, 90% down to 20%;
- a **review rate**: how an unanswered question comes back;
- **feedback shown or hidden**;
- **whether questions they skipped come back** through review, or are dropped. With
  review at 0 nothing comes back;
- **redo prompts**: after each miss they see, ask whether to take it again (the app's
  "offer repeat-or-continue"). These 12 settings (6 targets × feedback) have review 0, so
  a miss comes back only when the person chooses it.

Until 2026-10-11 there were 84 settings, all with scheduled review, and the rollouts
always scheduled review even though the app asks redos. The planner was simulating a
task the person never saw. Rollouts now follow the setting's review mode.

Rollouts include giving up on questions, driven by the same estimated `f` and `k` and
the person's estimated patience, so a hard target is worth less for someone who would
skip most of it.

There are two controllers. `Controller`, described here, plans for reward.
`InfoController` (see "Planning to measure") plans to measure `f` and `k`, and is the one
the app runs.

It screens all candidates with short rollouts (4 × 80 decisions), re-tests the best four and
the current setting with long rollouts on fresh seeds (16 × 300), and switches to the
one earning the most **total reward** only if that re-test beats the current setting
by more than twice its standard error. Short rollouts alone were too noisy to act on
(the V1 case that showed it is in `RETIRED.md`). The interface plans in a separate
process, so play continues meanwhile; one `InfoController` plan takes about 4.6 s (see
"Code layout, tests and speed").

About 15% of updates pick a setting at random instead. Those probes are independent of
the person, and when the best setting keeps feedback on they are the only source of
sets with it hidden, so they are what gets both ways of timing a set's reward into
the same record. `InfoController` makes no random probes: it varies the setting because
that is what it plans for.

The reward controller maximizes **total reward**: right answers weighted by how hard each
was for the person, `1 − p`, summed over the sets they go on to play. It is engagement
weighted by challenge, not engagement itself. Under V1, sets played was rejected as the
objective because it always favored the easiest setting (`RETIRED.md`). That reason no
longer holds: under the current model the easiest target is among the worst for
engagement (below). Whether the reward controller should maximize engagement instead is
open (Next steps). `InfoController`, which the app runs, plans to measure `f` and `k`; its
only engagement rule is the safety floor, which counts sets started.

**Which difficulty and whether to show feedback (2026-10-15).** Question-type task, skips
on, one setting held for each run of 300 start decisions, review 0.25, skipped questions
dropped, no controller. The target is the predicted chance of a right answer each
question is aimed at; it is a prediction from the ratings, and the chance actually
achieved was not recorded. γ = 0.9, `ACHIEVEMENT_SCALE = 4`, continuing bootstrap, `T_DUR = 7.0`.
Claims fixed before the run (paired on seed): a peak is determined if it
beats both neighbouring targets beyond 2 SE; the feedback crossover is established if
shown − hidden is positive beyond 2 SE at one target and negative beyond 2 SE at a
harder one.

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
| total reward | **+2.9 ± 0.9** | **+5.6 ± 1.5** | −1.1 ± 1.5 | +3.5 ± 2.6 | +8.4 ± 4.8 | **+8.1 ± 3.7** |

- **Showing feedback raises engagement at every target**, by 0.04 to 0.08, each beyond
  2 SE (2.4 to 7.9 SE). A seen right answer lands as reward at once instead of being
  discounted to the end of the set.
- **On reward it helps at easy targets and is too noisy to call at hard ones.** At 20 to
  30% a run's total depends heavily on runs of luck, so the paired difference varies about
  five times more than at 90%. The 20% result (2.2 SE) is one of six tests and should not
  carry weight alone.
- **Across people (2026-10-15).** The 16 people of the 4×4 grid, six targets, shown and
  hidden, 16 paired seeds each. Claim fixed before the run: showing feedback never lowers
  engagement if at most 4 of the 96 person × target cells fall below zero beyond 2 SE and
  none beyond 3 SE. **It holds: 0 cells below zero beyond 2 SE**, 87 of 96 positive. The
  per-person claim that it helps beyond 2 SE at every target held for 1 of 16; 16 seeds
  per cell are too few. Pooled over targets (descriptive), it raised P(start) beyond 2 SE
  for 13 of 16 people, by 0.015 to 0.081; the other three have `k = 1.5` and barely start.
- **By difficulty, over all 16 people** (256 paired runs per target): P(start) shown −
  hidden was +0.031, +0.034, +0.027, +0.039, +0.050, +0.044 at 90, 75, 60, 45, 30, 20%,
  each beyond 2 SE. Total reward shown − hidden was +1.0 ± 0.5, +1.4 ± 1.0, −0.2 ± 1.2,
  +2.6 ± 1.9, **+7.7 ± 2.1**, **+8.1 ± 1.8**: with feedback shown, people earn more at
  hard targets, where the one-person run above could not tell.
- **Lockout (2026-10-15).** Lockout is a run in which the person almost never starts
  again (start rate below 0.10 over 300 decisions, defined before the run). Four
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

Two settings are not levers. `reentry_cost` is left out because cheaper re-entry
always wins, and setting it from the person's state couples cost to value, which earlier
collapsed the `k` estimate to zero. Showing the stage count or the score is left out
because nothing in the model represents them.

**Decision rule for controller comparisons, fixed on 2026-09-13.**
- A claim that the controller beats the fixed default, or beats random switching, for a
  person needs the mean over seeds of the paired difference to exceed twice its standard
  error. Paired means the same seed: the same simulated person and question
  difficulties. The controller applies this margin to itself before switching.
- A difference inside that margin is reported as undetermined, in either direction.
- Every cell reports the mean ± standard error over seeds.
- The best fixed setting is chosen on 8 seeds and scored on 8 others, so picking and
  scoring on the same runs does not flatter it.

The reward controller's evaluation under this rule (8 seeds, five people, two of them
defined by the retired reaction terms) is in `RETIRED.md` (V2). It has not been repeated
under the current model; the comparison that has is under "Planning to measure".

### Planning to measure (`InfoController`, since 2026-10-11)

A controller that plans for reward settles on one setting, and once it does, cost stops
varying and `k` stops being identified. `InfoController` instead picks the setting
whose rollouts would most sharpen the `(f, k)` estimate.

Drift is `f · value − k · cost`, so a decision depends on `(f, k)` only through its
drift `μ`, and its Fisher information about `(f, k)` is
`I_μ(μ) · [[v², −vc], [−vc, c²]]`. `I_μ` covers choice, RT and undecided past the
deadline, and is computed once from pyddm solutions on a grid of `μ`. A setting scores
the rise in `log det(prior + data so far + its rollouts)`, which favours settings that
move value and cost in directions the data has not covered yet.

- **It plans for every person the data still allows, not one.** The score is averaged
  over 6 `(f, k)` draws: from the prior (`PRIOR`, sd 0.5) until the fit is used, then
  from a Laplace approximation, `N(estimate, (prior + data information)⁻¹)`. Each draw
  is rolled out with its own parameters (25 start decisions per rollout). Planning for
  the single current estimate collected the best data for the wrong person early on.
- **Safety floor.** A setting is eligible only if its rollouts start at least half as
  many sets as the best setting's, so measuring cannot drive the person away.
- **It found the redo prompts on its own.** The redo decision is where cost varies most
  (`0.15 · min(1/p, 8)`, about 4× across difficulties), and the controller chose a
  redo-prompt setting on 93% of its updates. Random probing over the same 96 settings
  picked one 11% of the time and saw about 5 redo prompts per person.

**Measuring costs no detectable reward (2026-10-11).** Four ways of choosing settings were
run on the same 32 simulated people: a 4 × 4 grid of f and k in {0.2, 0.6, 1.0, 1.5}, 2
seeds each. Each run had 300 start decisions, the question-type task, and skips on, and
the controllers never saw the true f and k. Settings: γ = 0.9, `ACHIEVEMENT_SCALE = 4`,
continuing bootstrap, `T_DUR = 7.0`. Pass conditions were fixed before the run, in
`controller_compare.py`, and the measuring rows are info_design2's `bayes_redo` rows. The
port check reproduced those rows' simulated decisions, reward and redo prompts exactly;
the fits matched to within 1e-4.

| how settings were chosen | f RMSE | k RMSE | total reward | P(start next set) | redo settings chosen |
|---|---|---|---|---|---|
| fixed default | 0.093 | 0.310 | 60.0 | 0.64 | 0% |
| for reward, 84 scheduled-review settings (the app's controller until 2026-10-11) | 0.054 | 0.293 | 147.0 | 0.76 | 0% |
| for reward, all 96 settings | 0.056 | 0.223 | 142.7 | 0.75 | 35% |
| for information, all 96 (`InfoController`) | **0.038** | **0.046** | **152.4** | 0.75 | 93% |

**Over time (2026-10-15).** The same people and seeds, fitted at 100, 200 and 300 start
decisions (the reward and fixed arms re-run with checkpoints; all 96 reproduced the
original simulations exactly, with fits within optimizer jitter of 0.01). The spread of
cost is the standard deviation of cost across the decisions collected so far.

| how settings were chosen | k RMSE at 100 / 200 / 300 | f RMSE at 100 / 200 / 300 | sd of cost at 300 |
|---|---|---|---|
| fixed default | 0.575 / 0.622 / 0.310 | 0.152 / 0.141 / 0.093 | 0.134 |
| random every 25 decisions | 0.309 / 0.247 / 0.126 | 0.103 / 0.067 / 0.071 | 0.171 |
| for reward, 84 settings | 0.471 / 0.391 / 0.294 | 0.079 / 0.066 / 0.054 | 0.112 |
| for reward, 96 settings | 0.358 / 0.346 / 0.223 | 0.079 / 0.077 / 0.056 | 0.126 |
| for information (`InfoController`) | **0.090 / 0.059 / 0.046** | **0.075 / 0.056 / 0.038** | **0.272** |

Choosing for information measures `k` better at 100 decisions than every other way does at
300, and it collects data in which cost varies about twice as much from the start. Reward
planning measures `k` worse than random settings at every checkpoint.

**What measuring costs, block by block (2026-10-15).** The same 160 runs (5 arms × 32
people), re-run with reward and start decisions recorded at 100, 200 and 300 start
decisions (`private/experiments/controller_blocks.py`). All 160 reproduced the saved total
reward exactly. No claims were fixed before this run, so these are descriptive.

| how settings were chosen | reward 1–100 / 101–200 / 201–300 | P(start) 1–100 / 101–200 / 201–300 |
|---|---|---|
| fixed default | 34.0 / 16.8 / 9.2 | 0.70 / 0.67 / 0.56 |
| random every 25 decisions | 41.2 / 15.0 / 7.5 | 0.60 / 0.60 / 0.44 |
| for reward, 84 settings | 62.6 / 48.8 / 35.6 | 0.74 / 0.79 / 0.76 |
| for reward, 96 settings | 64.0 / 48.4 / 30.3 | 0.74 / 0.79 / 0.71 |
| for information (`InfoController`) | 49.0 / 48.6 / 54.9 | 0.70 / 0.76 / 0.81 |

Paired differences, information − reward planning (84 / 96 settings):

- **First 100 decisions:** reward −13.6 ± 3.4 / −15.1 ± 3.2, both beyond 2 SE. P(start)
  −0.040 ± 0.024 / −0.049 ± 0.024: the first is undetermined, the second just past 2 SE.
- **101–200:** reward −0.2 ± 4.7 / +0.2 ± 4.4; P(start) −0.032 ± 0.018 / −0.038 ± 0.014.
  Reward is undetermined; the P(start) gap against the 96-setting controller is 2.7 SE.
- **201–300:** reward +19.3 ± 6.3 / +24.6 ± 6.5, both beyond 2 SE. P(start) +0.049 ±
  0.031 / +0.102 ± 0.037: the first is undetermined, the second 2.8 SE.

So measuring does cost something, early: about 14 reward and 4 to 5 points of P(start)
in the first 100 decisions. It is repaid by the third block, where the reward
planners' engagement and reward fall off and measuring's keep rising. The totals over
300 decisions (above) hide this by netting the two.

Paired differences over the 32 people (same people and seeds); a difference counts only
beyond 2 SE:

- **P1 passes.** Squared error in k, measuring − reward planning (96 settings), was
  −0.0477 ± 0.0232. That is just past 2 SE (0.0464), so it is a narrow pass. For f it was
  −0.0017 ± 0.0013, undetermined, so f is not worse. Against the app's previous controller
  the k gain is clearer: −0.0837 ± 0.0322 (2.6 SE).
- **P2 passes.** Reward, measuring − fixed default: +92.4 ± 9.1.
- **Over the whole run, measuring cost no detectable reward.** Measuring − reward planning:
  +9.7 ± 10.2, and measuring − previous controller: +5.5 ± 10.1. Both are undetermined, and
  both point estimates favor measuring. Block by block it does cost reward early and
  repay it late (above).
- **Fixing the reward controller's rollouts changed little.** Reward, 96 − 84 settings:
  −4.3 ± 3.7. k squared error: −0.0360 ± 0.0395. Both are undetermined. Given redo
  prompts it used them on 35% of updates (28 redo decisions per person, against 338 for
  measuring).
- **Reward planning leaves k poorly identified.** At 0.22 to 0.29 RMSE, k is barely
  better than with the fixed default (0.31). k came out below 0.05 for 3 of 32 people
  under the previous controller and 1 under the 96-setting one, all with true k = 0.2.
  None did under measuring. That fits the reason the measuring controller exists: reward
  planning settles on a setting, so cost stops varying. This run did not log cost spread,
  so it does not test that reason directly.

---

## Model structure

Four layers: an environment, a learner, decision nodes, and an inference step that
inverts all three.

### 1. Environment

A staged task. Each set is `stage_amt` slots (default 4), one question in each.

**Question types, rated per person** (`model/questions.py`). Questions come in types, each
with a rough prior difficulty in logits:

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

The prior is only a starting guess. After every answer an Elo-style update re-rates
both the person and that type,

```
p = sigmoid(ability − difficulty[type])
ability         += K_a · (correct − p)
difficulty[type] −= K_d · (correct − p)        steps shrink as answers accumulate
```

so a type's rating ends up describing how hard it is *for this person*. Each slot
then serves a type whose predicted chance is nearest the current **target chance**
(types within 0.1 of the best match are all eligible, so near-equivalent types all
get rated). Hard is relative to the person by construction.

**Only a question's first showing rates it.** A review or redo of the same question is
not a fresh sample of its type. Before this rule, a single question (34 × 8, shown 8
times) was all the evidence behind one player's 23% chance on every 2-digit × 1-digit
multiplication.

**Typos are not misses** (interface). A miss put right on the redo within 2 s
(`SLIP_SECONDS`) is treated as a slip: the ratings are restored to before the miss and
the question is rated as answered right, it is not owed, and it does not count as a
miss. The first player's log had six wrong-then-right redos;
two were fixed in under a second (11 + 20 → 21, then 31 in 0.4 s).

The recovery pipelines (`python -m temporal recovery`) still use the older single-`diff` task (with
optional per-stage `diff_spread`); question types are used by `controller.py`'s
simulated people and by the app.

**What a set is worth.** A set's reward is its right answers, each weighted by how
hard it was, times a scale (when each part of it lands is under "The learner"):

```
reward = ACHIEVEMENT_SCALE · Σ_right (1 − p) / stage_amt        ACHIEVEMENT_SCALE = 4
```

A trivial right answer earns almost nothing, so a set of easy wins is worth little
and a learner can stop out of boredom. Under the previous reward (fraction right)
the model could not represent that: it believed the easiest setting was always the
most rewarding, which is why a human player given only single-digit questions was
never offered anything harder (the V1 comparison is in `RETIRED.md`).

The scale is a calibration. The expected reward per question,
`p(1 − p)`, peaks at 0.25; scale 4 makes that peak 1.0, the value of a perfect set
under the old reward, so `f` keeps its earlier meaning. Since drive
is `f · Value`, the scale and `f` are interchangeable: changing one is equivalent to
rescaling the other.

The effort a set costs is its review load (the share of slots spent
going back over mistakes), and it raises the cost of the next start decision.

A slot normally holds a new question. It can instead revisit one missed
earlier, displacing the new question: sets never get longer, so review trades
against coverage. Who decides that is `review_mode`, described under the learner.

When right/wrong is hidden for a set, there is also no redo prompt: it
would only ever appear after a miss, so offering it would reveal the miss. Owed
questions still come back through review.

A skipped question spends its slot and earns nothing. It is held apart
from the questions missed: whether skipped questions join what review brings back is
a setting (`return_skipped`, a controller lever). One that comes back and is answered
right counts as a correction; answered wrong, it is owed like any miss; skipped
again, it stays skipped.

**Questions that come back** (interface). A question is never shown twice in one set,
except as a redo the person chose. One that comes back and is missed or skipped 3 times
(`RETIRE_AFTER`; redos do not count) is retired: it stops coming back, and a `retired`
event is logged. Before this, one player was shown 853 − 163 twice in a row and skipped
it both times in 1.5 s. The simulator's owed items are question types rather than
individual questions, so neither rule applies there.

When the learner is the one deciding redos (`review_mode = "choice"`), effort
aversion produces a spiral nothing stipulates: high `k` declines the redos, so less
is learned, so fewer questions are answered right, so `V` stays low, so
re-engagement drops. It was measured under V1 (`RETIRED.md`) and has not been
re-measured.

### 2. The learner

The learner maintains two quantities, kept separate by design.

Value, learned by TD(0) with linear function approximation:

```
V(s) = θ · φ(s)                    φ(s) = [1, stage position]
δ    = r + γ·V(s+1) − V(s)         θ ← θ + α·δ·φ(s)
```

At a set's last slot, `V(s+1)` is `V(0)`, the start of the next set: the continuing
bootstrap (see "Value bootstrap").

**A slot's reward arrives when the person is shown it.** With feedback on, a question's
worth lands at that slot: `ACHIEVEMENT_SCALE · (1 − p) / stage_amt` if they got it
right, nothing if they did not.
With feedback off nothing lands until the end-of-set score, which pays the whole set at
once. A set totals the same either way: feedback moves *when* the signal arrives, not
how much (`value.slot_reward`). That timing is the whole of feedback's effect on
motivation, and it is what the three reaction terms used to stand in for (see below).

With feedback hidden this is exactly the old schedule, one lump of
`ACHIEVEMENT_SCALE · achievement / stage_amt` at the last slot, so the only change is
that showing an outcome moves its reward forward to the slot it happened at.

#### Feedback is timing, not reward (since 2026-09-21)

Until 2026-09-21 the drift had three more terms,
`b_seen · seen_miss + b_unseen · unseen_miss + b_fb · feedback`. They patched a hole in
the learner: mid-set outcomes produced no prediction error at all, because every slot but
the last paid zero reward, so a miss the person was shown moved `δ` by nothing and
feedback never entered the TD system. The terms read, off to the side, what the value
system should have been computing.

They were replaced by the mechanism they stood in for. A slot's worth lands when the
person is shown it; with feedback hidden it lands at the end-of-set score, and a set
totals the same either way. A seen miss is then a slot that paid nothing against a value
that expected something, a negative `δ`, and an unseen miss cannot produce one. That
error is carried into the next decision through `value`, at full weight, scaled by the
same `f` as everything else. Three fitted parameters went and none came in.

- **Whether to show feedback is derived, not fitted.** Showing an outcome delivers its
  `(1 − p)` early, discounted less, and also delivers the misses instead of burying them
  in the end-of-set lump, so its effect depends on difficulty. What that does to reward
  is measured under "Which difficulty and whether to show feedback".
- **Boredom is a state the model can express**: sustained `|δ| ≈ 0`. A trained person on
  an easy question has `(1 − p) → 0`, so being shown they were right confirms an
  expectation and pays nothing.
- **What it gives up** is a person *spurred on* by a visible miss. `b_seen > 0` was the
  only way to express that sign; under pure RPE a seen miss is negative for everyone, in
  proportion to their `f` (see "Known limitations").

The reaction terms' recovery and the controller results that depended on them are in
`RETIRED.md` (V1, V2).

`φ` used to carry `difficulty` as a second feature. It was doing no work: `diff`
never varies during learning, so `θ[1]` collapsed to exactly `θ[0]·d`, perfectly
collinear with the intercept, and contributed ~1% of `V(0)`.

Competence, in the simulator, is learned only from correcting what was missed:

```
P(success) = sigmoid(ability − difficulty)       ← Rasch / IRT
outcome    ~ Bernoulli(P(success))               ← per question
ability   += CORRECTION_GAIN   only when a question answered wrong (or skipped)
                               earlier is shown again and answered right
```

A first-time correct answer teaches nothing, and neither does a failure; only a
correction does (`competence.update_ability`, `competence.settle_skipped`). This is deliberate practice rather than
error correction: working a question out between sessions, without being shown it
again, teaches nothing here.

**Measured versus simulated competence.** For a real person there is nothing to
simulate: the app *measures* competence with the Elo ratings above, and the
controller's rollouts start from that measurement. The corrected-question rule is
how a simulated person's true ability changes; the ratings are what the environment
can know about it. The controller only ever sees the ratings, never a simulated
person's truth.

**Who decides whether a missed question comes back** is `review_mode` on
`Simulation`, not the learner in general:

| mode | who decides | learning gated by |
|---|---|---|
| `scheduled` (default) | the controller, at `review_rate` per slot | the schedule |
| `choice` | the learner, offered a redo after each miss | `k` |
| `quiet` | nobody; missed questions never return | nothing is learned |

That choice is what keeps `k` a motivational parameter. When the learner decides,
`k` sets the learning rate as well as the willingness to start, and effort cost and
skill can no longer be manipulated separately (V1 measurement in `RETIRED.md`).

Competence is kept off `δ` because these are different learning systems. Prediction
error is the teaching signal for value; competence comes from repetition and error
correction, a largely separate learning system. Driving skill from `max(0, δ)`
would halt competence growth as soon as predictions become accurate, which is the
wrong direction: expertise continues to refine long after outcomes become
predictable.

### 3. Decision nodes

The two choices, starting the next set and redoing a miss, use the **same
accumulator** (same drift form, noise, bound and deadline) fed different conditions.
They are two observations of the same person, and their draws stack into one fit.
`GO = 1` means "took the effortful option" in both.

**The initiation node** sits between sets and decides whether to start the next one.

- Value is `V(state, 0)` plus the last set's total prediction error, `Σ δ`, clipped to
  `±DELTA_CLIP` so the condition grid stays bounded. The error is carried at full
  weight; there is no fitted carry coefficient.
- EffortCost is `structural(in_session) + EFFORT_WEIGHT · last_episode_effort`:
  `WARM_COST` (0.05) while in a session, `state.reentry_cost` (the cold-start
  profile: `low` 0.3, `mid` 0.6, `high` 1.0) after leaving, plus `EFFORT_WEIGHT` (0.1)
  times `1 + review load` of the last set.
- Noise, boundary and deadline are pinned at `SIGMA = 0.7`, `BOUND = 1.5`,
  `T_DUR = 7.0` so they cannot absorb variance that belongs to the parameters. The
  deadline was 3.0 s until 2026-09-13; see "The deadline was discarding human choices".

**The repeat-or-continue node** fires after a missed new question, when the learner
decides redos and feedback is shown. It fires *after* the miss has landed as reward,
not before: they see how the question went, that produces the error, and then they
decide whether to take it again. Value is the slot's share of the terminal reward,
discounted by slots to come, plus that error, so missing an easy question, which is
the bigger surprise, discourages the redo more than missing a hard one. Cost is how
hard the missed question is for them: `REPEAT_COST_SCALE` (0.15) times the expected
attempts `1/P(success)`, capped at 8. Cost varies more here than anywhere else, so what
this node carries is `k`.

**The persistence node** runs on every question while it is being worked on (see
"Giving up on a question"). It shares `f` and `k` with the other two but is a
one-boundary process with no deadline. Seconds become model time through the person's
`patience` (default 10 s), estimated per person. Simulated people also need a time
to solve each question; it is lognormal with a median of `4 s / p^0.7`: about 4 s for
a sure thing and 20 s at a 10% chance, close to the 1 to 30 s one human took in
`data/`. Real people simply take the time they take.

### 4. Inference

PyDDM's `Model.fit()` solves the Fokker-Planck equation for the first-passage time
distribution and fits by differential evolution: `(f, k)`, bounded to `[0, 3]`. Those two
are the whole drift. Where real people are fitted (the controller's estimate and the
interface's fit), the non-decision time `T0` is fitted as well, bounded to [0, 0.5] s;
the recovery pipelines pin it at 0 (see "Recovery demonstrates identifiability, not
accuracy"). What the last set did to the person is already inside `value`, so it needs
no parameters of its own and is not fitted separately.

Answer and skip times have a closed-form likelihood (`persistence.neg_log_likelihood`,
per second, with lapses). The controller uses it only for patience
(`persistence.fit_patience`, with `f` and `k` held at the decisions' estimate). Passing
it to `fit` as `attempts` adds it to pyddm's likelihood through a custom loss, fitting
decisions and persistence as one set of parameters. That joint fit is kept for
comparison and is not used for planning.

A decision that reached no boundary by `T_DUR` is an undecided trial. That includes
a *person* who took longer than 7 s to click, since pyddm refuses outright to fit an
RT past the model's deadline. Undecided trials are fed back in as counts, not
discarded; see "Censored trials are data".

---

## Human sessions (`temporal/app/`)

You answer the questions, choose whether to redo one you got wrong, and choose
whether to start the next set. There is no countdown: both prompts wait for you, and
the time to click is the reaction time. **"I'm done" closes the app** and is logged
as the stop decision. Closing the window without choosing is logged as an abandon.

**Skip** (the button under the answer box, or Esc) gives up on the question on
screen. It is logged with how long you worked on it and your predicted chance on it,
and the set moves on to the next slot. Whether skipped questions come back is the
"skipped questions come back too" checkbox, or the controller's call.

**Leaving is recorded with what was on screen.** Closing the window, Ctrl-C in the
terminal, closing the terminal, or `kill` all log an `abandon` with the screen you were
on. On a question it also records the question, your chance on it, how long you had
it, and whether it was the first question since opening the app. Looking at a question
and leaving is giving up on it, so it joins the persistence data. An exit nothing sees
(a crash, power loss) is found on the next launch and logged as `unlogged`, with the
question that was on screen and no time.

**Old saves are brought up to the current rules.** The first-showing, typo and
retirement rules apply to the whole record, not only to new answers. A state saved
under older rules gets its ratings and failed-return counts rebuilt from
`decisions.jsonl` on the next launch, logged as `rebuilt`, and questions that have
already failed 3 returns are retired. The record itself is never rewritten. On the
first player's data:
- their chance on 2-digit × 1-digit multiplication went from 37% to 55%, now rated on
  1 answer instead of the 8 showings of 34 × 8;
- measured ability went from 1.26 to 1.47;
- 853 − 163 was retired.

Replaying a record written under the current rules reproduces the ratings kept live
exactly, so the rebuild is the same rule applied after the fact, not an approximation.

`RULES` went to 3 on 2026-09-21, when the reaction terms were removed. That one is not
replayable: a start decision logged under rules 1 and 2 holds a `value` that predates the
prediction-error carry, so its `value` is right for the drift it was drawn under and
not comparable with a later row. The rows are kept and still fit, since the drift form
`f · value − k · cost` is unchanged, but they under-represent how much value varies.

**Everything persists** in `data/` (gitignored):

- `decisions.jsonl`: the record, append-only and synced to disk before the app moves
  on: every start and redo decision, every answer and skip (question, type, your
  answer, your predicted chance, time on it), exits with what was on screen, retired
  questions, set starts and ends, controller decisions, launches.
- `state.json`: the snapshot to pick up from: value weights, ratings, owed and
  skipped questions with their exact text, how often each came back and failed,
  settings, and what is on screen right now. Written atomically, so a crash never leaves half a
  file.
- `controller.pkl`: the controller, including its estimates.

Reopening the app logs a `return` with the time since you left, and every fit and
plan uses every session. "Export all decisions to csv" writes `data/decisions.csv`.

Tick **"controller sets target, review, feedback, skipped, redo prompts"** and
`InfoController` plans in a separate process every 5 start decisions and moves the
settings panel itself, with a line saying why. With it off, the target chance, review
rate, and the feedback, skipped and redo-prompt checkboxes are yours. Feedback changes
apply from the next set.

Three properties of human data follow from this flow, and all three matter for inference:

- **Every stop is a session end.** Each session contributes at most one "no" at the
  start prompt, and those are what identify `k` there. Many sessions are needed. In
  practice people stop by closing the app: the first player's four sessions ended with
  one "I'm done", two window closes, and one exit nothing logged. Those exits are why
  leaving is now recorded with what was on screen.
- **The fit never sees a cold start from a human.** Opening the app is the cold
  decision, and its "reaction time" is hours. Every start decision the fit sees is
  made mid-session, at warm cost. For humans, most of what identifies `k` comes from
  the redo prompts, whose cost does vary with how hard the missed question is, and
  now also from time spent on questions.
- **Every question is a persistence observation.** A set gives one start decision but
  four or more questions, each with a time and whether it was skipped. Earlier
  sessions' answers are used too: they were logged with a time and a predicted
  chance, so they enter as questions nobody gave up on.

A redo in the interface is an extra question rather than the next slot, so a human
set can run past `stage_amt` questions and its review load can exceed 1; in the
simulator's `choice` mode a redo takes the next slot.

### On a website (`temporal/app/web.py`)

`python -m temporal.app.web` serves the same task in a browser at `http://127.0.0.1:8000`. The
browser only draws screens and times them; every rule is `session.Session`'s, run on the
server by a subclass that sends its screens to the page instead of Tk. Nothing about
the task, the record or the inference is re-implemented.

**Data collection is off until it is approved.** Research with other people needs
approval first, so by default (`TEMPORAL_COLLECT=0`) nobody else's data is kept:

- The start screen shows the consent information
  (a **draft**, in `temporal/app/static/index.html`, to be replaced with the approved wording), and
  offers two ways in: **as a guest** or **with an email**.
- **Guests are never saved.** Their record lives in a temporary folder that is deleted
  when they leave the page.
- **Email** is accepted only for the addresses in `TEMPORAL_RESEARCHERS`
  (comma-separated): the researchers, playing themselves. Their records are kept in
  `$TEMPORAL_DATA/<email>/` (default `data/web/<email>/`), with the same
  `decisions.jsonl` / `state.json` / `controller.pkl` as `data/`, and they see
  "dev stuff".
- With `TEMPORAL_COLLECT=1` anyone may take part with an email after ticking the
  consent box. The first event of each session is a `consent` record with the email,
  whether they agreed, and `CONSENT_VERSION` (in `temporal/app/web.py`; bump it whenever the consent
  text changes).

The email is not verified: anyone who types an address plays as that person. That
is enough to keep one person's sessions together, but it is not a login.

- **Reaction times are measured in the browser**, from the moment a screen is painted
  to the key press or click, so network latency never enters an RT. The 600 ms
  "correct" / "wrong" flash plays in the browser before the next screen, and its clock
  starts after the flash.
- **Starting from the start screen is launching the app**, so each start is a new
  session. Closing the tab, reloading or navigating away is closing the window: it is
  logged as an `abandon` with what was on screen and the browser's time on it. Starting
  again elsewhere (another tab or device, same email) closes the first one the same way.
  A page left silent for 2 hours is closed as well. A server that dies without closing
  is caught on the next launch, as `unlogged`, the same as a crash on the desktop.
- **Plans and fits run in one process pool** shared by all players.
- **Dev stuff** (settings, behind the scenes, fit, export) is shown to researchers
  only, or to everyone with `TEMPORAL_PANEL=1`.

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

Anywhere else, build the `Dockerfile`, mount a persistent volume at `/data`, and
set the same environment variables. It listens on `$PORT` (default 8000).

---

## Running it

```bash
uv sync
```

```bash
# Play it yourself. Data is kept in data/ between sessions.
python -m temporal.app.desktop

# The same task in a browser, at http://127.0.0.1:8000. Only researchers' data is kept,
# in data/web/<email>/, until TEMPORAL_COLLECT=1.
TEMPORAL_RESEARCHERS=you@example.com python -m temporal.app.web

# One simulated person: a fixed setting vs the reward and measuring controllers
python -m temporal.controller --values 0.8,0.8

# One simulated learner on the single-diff task
python -m temporal run --values 0.8,0.8

# Parameter recovery from ONE learner (the default)
python -m temporal recovery --mode single

# Parameter recovery from two learners pooled at different cold-start costs
python -m temporal recovery --mode coupled

# Parameter recovery, value as a fixed design constant (the reference)
python -m temporal recovery --mode uncoupled

# f and k from decisions, from answer and skip times, and from both
python -m temporal recovery --mode persistence

# patience, estimated the way the controller does
python -m temporal recovery --mode patience

# one learner's data split across four learners at different points on the learning curve
python -m temporal recovery --mode reseeded

# what pinning the non-decision time at 0 costs, and whether freeing it gets f and k back
python -m temporal recovery --mode t0

# every pipeline above
python -m temporal recovery --mode all

# Human decision times against the deadline
python -m temporal audit

# Regression tests: seeded runs of the model, both controllers and the app, pinned in
# tests/golden.json. Re-record with --record only when a change is meant to alter output.
uv run pytest
```

Bare `python -m temporal` is shorthand for `python -m temporal run`.

For `run`: `--values` takes `f,k`; `--cost` picks the cold-start profile; `--review-mode`
and `--review-rate` set review; `--diff-spread` sets per-stage difficulty spread;
`--decisions` caps the number of start decisions (default 200). For `recovery`: `--mode`
picks the pipeline and `--reps` sets repetitions per grid point (default 5; most tables
below used 3). For `audit`: `--path` points at a `decisions.jsonl` (default
`data/decisions.jsonl`). For `temporal.controller`: `--values`, `--decisions` (default
300), `--seed`.

---

## Files

```
temporal/
  cli.py            entry point (python -m temporal): single runs, recovery sweeps, audit
  model/            the person, as the environment models them
    params.py         UserParams (f, k, discount, learning rate, patience, T0) and ModelState
    accumulator.py    the drift-diffusion process behind start and redo decisions; cached solves and draws
    conditions.py     the (value, cost) each decision node feeds it
    value.py          TD value of a set, and when a slot's reward lands
    competence.py     chance of a right answer, and how it grows
    persistence.py    giving up on a question: quit drift, patience, closed-form likelihood
    questions.py      question types, their priors, and per-person Elo ratings
    simulation.py     the simulated learner: sets, review, redo prompts, start decisions
  inference.py      sample construction, censoring, the two-parameter fit, sanity checks
  recovery.py       data generation and recovery pipelines, including persistence and patience
  controller.py     Controller (plans for reward) and InfoController (plans to measure f and k)
  plots.py          recovery figures
  app/              the playable task
    session.py        the rules of a session, shared by both frontends
    store.py          the record on disk: append-only log, snapshot, saved controller
    desktop.py        the Tk window
    web.py            the same session served to a browser; static/index.html draws it
tests/              seeded regression tests (golden.json)
RETIRED.md          results from earlier model versions, and ones later measurements replaced
Dockerfile, fly.toml   hosting the web app; data in a volume at /data
data/               human session records (gitignored)
assets/             generated figures
```

---

## Current results

Every number in this section was measured under the current model (V3 in `RETIRED.md`):
the two-parameter drift `f · value − k · cost` with prediction error carried in `value`,
reward landing when shown, γ = 0.9, `ACHIEVEMENT_SCALE = 4`, continuing bootstrap,
`T_DUR = 7.0`, `T0` pinned at 0 in simulation. Earlier measurements,
and the ones these replace, are in `RETIRED.md` by model version. Recovery runs sweep a
4×4 grid over `f, k ∈ {0.2, 0.6, 1.0, 1.5}`.

**Recovery from one person (2026-10-10).** One simulated person, mid cold-start cost,
scheduled review, 1200 recorded decisions per fit split across four learners after 0,
200, 500 and 1000 warm-up decisions, single-`diff` task, 4×4 grid × 3 reps (48 fits),
95% bootstrap intervals over the fits:

| f RMSE | k RMSE |
|---|---|
| **0.035** [0.028, 0.042] | **0.034** [0.025, 0.043] |

- **The pass condition is met.** It was fixed on 2026-09-13: `f` and `k` RMSE ≤ 0.05, the
  step the controller rounds its estimates to (`snap_value`), so an estimate off by about
  one step plans like the truth or its neighbour. Both upper bounds are under 0.05.
- **The worst grid points are effort-averse people who rarely start** (`f` RMSE 0.067 at
  f 0.6, k 1.5) and low-`k` people for `k` (0.067 at f 0.2, k 1.5; 0.059 at f 1.0, k 0.2).

**The recovery pipelines (2026-10-15).** The same grid; `python -m temporal recovery`:

| pipeline | f: r | f: RMSE | k: r | k: RMSE |
|---|---|---|---|---|
| **single**, one continuous learner, 1200 decisions (3 reps) | 0.995 | 0.049 | 0.996 | 0.044 |
| **coupled**, two stacked learners at different cold-start costs, 700 decisions (3 reps) | 0.998 | 0.030 | 0.997 | 0.035 |
| **uncoupled**, value and cost set by design, 400 per condition (5 reps) | 0.999 | 0.026 | 0.998 | 0.027 |

![Coupled recovery](assets/initiation_recovery_coupled.png)

Use `single` (or the four-learner design above) when modelling one real participant, and
`coupled` or `uncoupled` when the experimenter controls the design. A single learner
generates its own value and cost, correlated and confined to what it happens to
experience, so it recovers less well than a designed contrast.

**Prediction error is most of the variation in value.** On the question-type task, where
the target holds `p` away from 1 and `δ` stays alive, `δ` supplies about 80% of the
standard deviation of value. Eight fits, 700 decisions each, four `(f, k)` pairs × 2
seeds, settings switched by random probes: `f` RMSE 0.049, `k` RMSE 0.027, value sd 0.35
to 0.41, `corr(value, cost)` −0.22 to −0.50 (see "How value and cost together decide what can be measured"). On the single-`diff` task a learner masters the fixed difficulty, so
`p → 1` and `δ` shrinks.

**Skips do not help estimate `f` and `k` (2026-10-15).** Twelve simulated people (4
profiles × 3 reps) who can give up on questions, 300 start decisions each (human-sized
data), settings switched at random (`python -m temporal recovery --mode persistence`):

| fitted from | f RMSE | k RMSE |
|---|---|---|
| decisions only | **0.045** | **0.083** |
| answer and skip times only | 0.597 | 0.432 |
| both, jointly | 0.062 | 0.133 |

Adding answer and skip times to the decisions makes both parameters worse, and on their
own they cannot place `f`. A question's value, `p(1 − p)`, hardly varies across
questions, while its cost, `1/p`, varies about 4×, so skip times mostly measure `k`, and
the redo prompts already carry that. On the first real player's record, adding their
answer and skip times at a fixed 10 s patience pulls the minimum of −log L from `f` 2.75
to 2.15 (pyddm's own fits: 2.77 and 2.14). So the controller fits `f` and `k` from
decisions, and uses time on questions only for patience.

**What each part of the fit contributes, alone and together (2026-10-15).** The 32 people
of "Planning to measure" (4×4 grid × 2 seeds), question-type task, settings chosen by
`InfoController`, 300 start decisions, with a true non-decision time of 0.3 s and true
patience 10 s. Each dataset is fitted 8 ways: timed-out trials kept or dropped, `T0`
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
temporal recovery --mode patience`):

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

**The redo node matters on the question-type task.** Its cost varies with how hard the
missed question is, and a controller that seeks out redo prompts brought `k` RMSE from
0.22 to 0.046 (see "Planning to measure"). On the single-`diff` task redo cost barely
varies, and stacking redo draws did little (V1 measurement in `RETIRED.md`).

---

## Methodological notes

### Censored trials are data

PyDDM's likelihood is not renormalized over the decided region. Fit only the trials
that reached a boundary and the optimizer inflates drift at no likelihood cost,
shrinking the predicted undecided mass. The correction is to feed the censored counts
back in as undecided trials. On the recovery design (48 fits of 1200 decisions, 4×4
grid × 3 reps, four learners across the learning curve, `T_DUR = 7.0`, continuing
bootstrap, 2026-10-10), 6,032 of 57,600 decisions (10.5%) timed out. Kept as undecided,
the mean bias was −0.2% for `f` and −0.5% for `k` (RMSE 0.035, 0.034). Dropped, it was
+12.9% and +10.6% (RMSE 0.099, 0.099).

### Every model here carries a 2% lapse mixture

pyddm's `gddm` mixes a lapse into every model by default (`mixture_coef = 0.02`): 2% of
trials are drawn from a uniform distribution over the whole deadline instead of from the
accumulator. It is left at its default, and it matters in three ways.

- **Simulation and fitting stay consistent.** The same mixture is in the draws the
  simulator takes and in the likelihood the fit maximizes, so recovery is unaffected.
- **It explains RTs faster than the non-decision time.** A model with `T0 = 0.3` still
  produces a few decisions under 0.3 s; they come from the mixture, not the accumulator.
- **It floors the likelihood.** No observed RT has zero density, so a single odd human
  decision cannot make a parameter set impossible. That is the same job the explicit
  lapse term does for giving up on questions (`persistence.LAPSE`), which had to be
  written by hand because that likelihood is ours rather than pyddm's.

### The deadline was discarding human choices

Audited on 2026-09-13 with `python -m temporal audit`: the first player's 26 start and redo
decisions, against the deadline of the time, 3.0 s.

| decisions | n | median | 90th percentile | longest | past 3.0 s |
|---|---|---|---|---|---|
| start | 12 | 1.02 s | 5.52 s | 6.07 s | 3 (25%) |
| redo | 14 | 0.94 s | 1.69 s | 2.99 s | 0 |
| all | 26 | 0.98 s | 3.82 s | 6.07 s | **3 (11.5%)** |

11.5% is far past a 2% tolerance, so `T_DUR` moved to 7.0 s, past the longest RT; none
of the 26 is past it now. All three slow decisions (4.64, 5.61 and 6.07 s) were choices
to start the next set, and two came right after a set in which the player skipped three
questions. Three decisions are too few to call that a pattern, but they are exactly the
hesitations a model of reluctance should see, and the fit had been recoding each one as
"no decision".

Under the current two-parameter model the same 26 rows give `f` 2.77, `k` 0.82, `T0`
0.00, with `f` near the edge of its bound, which is what 26 rows buys. How far the three
recoded choices moved the old five-parameter fit, and what the longer deadline did to
simulated people, are V2 results in `RETIRED.md`.

### The lockout state is a model prediction

At high re-entry cost a learner can stop starting altogether. Never engages →
never improves → `V` stays near zero → drift stays negative → still does not engage.
This matches the avoidance-maintained deficit cycle that behavioral activation therapy
targets in depression. Review mode decides what an escape
is worth: under `quiet` an escapee learns nothing, under `choice` a high-`k` escapee
declines the redos and learns little, under `scheduled` escapes compound.

### Recovery demonstrates identifiability, not accuracy

The generating model and the fitted model are identical here, so successful recovery is
expected. Real data is never generated by the fitted model.

**Non-decision time is the one misspecification measured** (`python -m temporal recovery
--mode t0`, 2026-10-15). One learner per run, 1200 decisions, 3 reps per value, true
`f = k = 0.8`, current model. Each dataset is fitted twice: with `T0` pinned at 0 and
with `T0` free. Means over reps:

| true T0 | f pinned | k pinned | f free | k free | T0 recovered |
|---|---|---|---|---|---|
| 0.0 s | 0.77 | 0.80 | 0.78 | 0.81 | 0.04 |
| 0.2 s | 0.73 | 0.75 | 0.81 | 0.81 | 0.22 |
| 0.3 s | 0.69 | 0.71 | 0.79 | 0.79 | 0.28 |
| 0.5 s | 0.64 | 0.66 | 0.78 | 0.79 | 0.48 |

- **Pinning it biases both parameters down, more as the true value grows**: 7 to 8% at
  0.2 s, 11 to 13% at 0.3 s, 17 to 20% at 0.5 s.
- **Freeing it removes the bias.** `f` and `k` land within 0.03 of the truth at every
  value, and `T0` itself within 0.03.
- **It costs almost nothing when there is none.** With a true 0 the free fit returns 0.04.
- **It is free where real people are fitted**: the controller's estimate and the
  interface's "fit my parameters". The recovery pipelines pin it, since their simulated
  people have none.

### Fixed parameters carry the identifiability

`SIGMA`, `BOUND` and `T_DUR` are held fixed precisely so they cannot absorb variance
belonging to the parameters. Freeing them removes the identifiability.

### How value and cost together decide what can be measured

Drift is `f · value − k · cost`, so `f` is read from how decisions change with value and
`k` from how they change with cost, or from cost's constant offset on the drift. The only
case in which `f` and `k` cannot be told apart is cost exactly proportional to value
(`cost = b · value`): drift is then `(f − b·k) · value`, and only `f − b·k` is measured.
No decision in this task is like that. Constant cost still identifies `k` through the
offset, and cost correlated with value (`cost = a + b · value`) identifies both.

What correlation, and too little variation in cost, do cost is **precision**: more
decisions for the same accuracy. Expected standard deviation from 300 start decisions,
from the Fisher information of the decisions at the true `f`, `k` (16 people × 2 seeds,
default setting for starts, a fixed redo-prompt setting for starts with redos; medians):

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

Carrying prediction error into value (2026-09-21) raised the correlation and improved
recovery anyway, because it added far more variance to value than it did shared
variance with cost. On the question-type task at a held target, 400 decisions:

| | corr(V(0), cost) | corr(δ, cost) | corr(value, cost) | sd of value |
|---|---|---|---|---|
| feedback on | −0.16 | −0.14 | −0.24 | 0.178 → 0.228 |
| feedback off | −0.30 | −0.23 | −0.40 | 0.156 → 0.223 |

Both channels correlate with cost for the same reason: cost rises with review load,
review brings back questions the person already missed, and those earn less than
expected. The correlation is a property of the task, not of the carry.

---

## Known limitations

- `f` is confounded with the discount factor. Under the continuing bootstrap
  `V(0) ≈ γ^(n−1)·E[r] / (1 − γ^n)`, so `f` is only identified up to that factor, and,
  since `ACHIEVEMENT_SCALE` multiplies value, up to that scale too. State `γ`, the scale
  and the bootstrap whenever an `f` value is reported; an `f` fitted under the terminal
  bootstrap is not on the same scale.
- The non-decision time is bounded to [0, 0.5] s when fitted. Below that bound only the
  2% lapse mixture gives a fast decision any density, so a person whose non-decision time
  exceeds their own fastest decision cannot be fitted, and the bound has to move with the
  data. The first player's 26 decisions put `T0` at 0.00 with a fastest decision of
  0.51 s, which cannot be told apart from a small positive value.
- The deadline is set from one person's decisions. `T_DUR = 7.0` sits past the longest
  RT seen so far (6.07 s); anyone slower is censored again. `python -m temporal audit`
  flags a share past the deadline above 2%.
- The value range is endogenous. A stalled learner stops generating value variation,
  so there is less leverage on `f`; the worst recovery checked sat at high `k`.
- At the start node, effort cost is weak and retrospective: review load moves cost by
  about 0.02. Nearly all start-node cost variation in simulation comes from sessions
  going cold and warm, which human data never shows the fit (see Human sessions). The
  redo node's cost varies far more, which is why `InfoController` leans on it.
- A type's rating only moves when that type is served. Types far from every target the
  person ever gets keep their prior indefinitely.
- The model cannot represent someone *spurred on* by a visible miss. Under pure RPE a
  miss they see is a negative error for everyone, scaled by their own `f`; only a free
  `b_seen > 0` could give it the other sign, and that is what was removed. If a real
  person turns out to work that way it has to re-enter as a cost effect, not a value one.
- `f` now carries two jobs: how much the expected payoff pulls, and how hard the last
  set's surprise hits. The old `b` terms let those differ per person; folding them into
  value asserts they are one sensitivity. That is a substantive claim about
  people rather than a simplification, and it has not been tested against a human.
- The prediction error is clipped at `±DELTA_CLIP` (1.0) so `value` stays on a bounded
  grid. It binds on about 0.5% of decisions with feedback hidden and never with it on.
- Learning still saturates at a fixed difficulty, and under `choice` and `quiet` the
  missed list grows without bound; nothing forgets.
- Value and cost are discretized (`VALUE_STEP`, `COST_STEP`) to keep the number of
  Fokker-Planck solves bounded.
- Both decision nodes share one accumulator and one set of parameters. That is an assumption:
  a real task might give the redo choice a different deadline.
- Patience is estimated with `f` and `k` held at the decisions' estimate, so their
  errors pass into it, and it absorbs anything about giving up that the drift does not
  describe. A skip at a glance and giving up after a long try are one process on a
  different time scale here, not two processes.
- Retiring questions, never repeating one within a set, and treating a quick fix as a
  typo are interface rules with no simulator counterpart. `RETIRE_AFTER = 3` and
  `SLIP_SECONDS = 2` are judgment calls: the first player got 34 × 8 right on its 8th
  showing, which retirement would now prevent.
- Treating an answer as a censored time to give up assumes how long a question takes
  to solve does not depend on motivation. Someone who rushes when they want to stop
  breaks that.
- Time on a question includes anything else they did meanwhile. Walking away mid-question
  reads as long persistence.
- Simulated solve times are an assumption (median `4 s / p^0.7`, lognormal), checked only
  against one person's 30 answers.

---

## Next steps

### Near term: prepare the inference for real data

1. Done on 2026-09-15: the non-decision time is fitted rather than assumed away, for the
   controller's estimate and the interface's fit (see "Recovery demonstrates
   identifiability"). What is left is to watch it per person as sessions accumulate, and
   to move the [0, 0.5] s bound if anyone's fastest decision falls near it.
2. Keep the deadline past human decision times: run `python -m temporal audit` after each
   batch of sessions. Done once, on 2026-09-13, which moved it from 3.0 to 7.0 s.
3. Add across-trial drift variability (Ratcliff's `sv`).
4. Value for a real person has an observable proxy. Under the continuing bootstrap it
   is `V(0) ≈ γ^(n−1) · E[set reward] / (1 − γ^n)`, and set reward is computable from
   the measured chance of each question, so no TD fit is needed. Confirm it tracks
   during learning, not only at the fixed point.

### Next: individual-level estimation

5. Hierarchical estimation. Real sessions give tens of decisions per person. Partial
   pooling toward a group distribution is likely the largest available accuracy gain.
6. Posterior predictive checks and held-out prediction on recorded sessions.
7. Seeding learners across the learning curve (`--mode reseeded`) is the recovery design
   (see "Current results"). Under V2 it did nothing detectable for `f` against one
   continuous learner (`RETIRED.md`).

### Then: close the loop

8. Improve the controller. Since 2026-10-11 the app runs `InfoController`, which
   measures `f` and `k` far better than reward planning at no detectable cost in reward
   (see "Planning to measure"). What remains:
   - **Re-measure the reward controller under the current model.** Its only evaluation
     against the fixed default, random switching and the best fixed setting is the V2
     8-seed table in `RETIRED.md`, two of whose five people cannot be rebuilt. Open
     questions from it: the reluctant, effort-averse person (too few sets to decide), the
     gap to the best fixed setting given true parameters, and whether exploration probes
     cost reward.
   - **Which objective.** Total reward and engagement disagree on where to aim (see "Which
     difficulty and whether to show feedback"), and the V1 reason for preferring reward
     no longer holds. Decide whether the reward controller should maximize engagement,
     reward, or engagement with a floor on challenge.
   - **Feedback.** Under the current model showing feedback raised engagement at every
     target for f 0.8, k 0.8 and never lost on reward beyond 2 SE. Check other people, and
     reward at hard targets with more seeds, before the controller relies on hiding it.
   - **Difficulty is one target per set**, from six levels, served from nine question
     types. There is no within-set sequencing (easy then hard), and at extreme targets some
     people cannot be served questions that match the label.
   - **The skipped-questions lever is unproven.** Nothing measured shows when returning
     skipped questions helps. A person whose ability grows from corrections would be the
     test.
   - `stage_amt` is not a lever yet, and show count / show score have no model channel.

9. Online re-estimation. Parameters drift within a session with fatigue, mood and
   boredom, so the controller must track them rather than fit once.

### Deferred, but kept in view

- Fisher information analysis, to diagnose unidentifiable parameter combinations
  directly rather than inferring them from wide scatter. `InfoController` already
  computes the `(f, k)` information of a set of decisions (`controller.fisher`), but only
  to plan; it has not been used as a diagnostic.
- Two-state fast/slow ability (Smith, Ghazizadeh & Shadmehr 2006), if the protocol ever
  gains washout and re-exposure blocks.
- Model a process that degrades: fatigue, forgetting, or the rising cost of continued
  exertion.

---

## Value bootstrap: continuing since 2026-09-13

This was the open question: is a set its own episode, or is the task a cycle? The start
decision re-enters the task, so it is a cycle. A set's last slot now bootstraps from the
start of the next set, `δ = r + γ·V(0) − V(last)`, which makes `V(0)` the fixed point of a
recurrence over all future sets (`value.CONTINUING = True`). The old terminal bootstrap,
`V_next = 0`, is still available for comparison. `value_at_choice_point` is still `V(0)`,
the state being entered.

- **Why it matters for measurement.** Under the continuing bootstrap value is larger and
  varies about twice as much, and `f` is identified from variation in value. The V2
  measurements of both, and of whether the switch moved the reward controller closer to
  the best fixed setting (it did not), are in `RETIRED.md`.
- **Real sessions.** A player's saved value weights were learned under the terminal
  bootstrap and adapt from their next set on.

---

## Code layout, tests and speed: since 2026-10-11

The model's equations did not change on 2026-10-11. What changed is which controller
the app runs, how the code is organised, how it is checked, and how fast it plans.

- **The app runs `InfoController`** (see "Planning to measure"), over 96 settings: the 84
  scheduled-review settings plus 12 with redo prompts. Rollouts now follow each
  setting's review mode. Before, they always scheduled review while the app asked redos.
  A saved controller from before this change is replaced on the next launch. The
  replacement is logged as `controller_reset`, and the estimate is refit from the record.
- **One package, `temporal/`** (see "Files"). `config.py` was split by what each part
  models: the person (`params`), TD value (`value`), competence, and the (value, cost)
  each node sees (`conditions`). `interface.py` was split into the session's rules
  (`app/session.py`), the record on disk (`app/store.py`) and the Tk window
  (`app/desktop.py`). The web frontend subclasses the same `Session`, so both still
  write the same record. Commands are now `python -m temporal ...` ("Running it").
- **Regression tests** (`uv run pytest`, about 25 s). They pin seeded outputs in
  `tests/golden.json`: the simulator under scheduled review and under redo prompts,
  people under a fixed setting and under both controllers, a fit to designed data and
  one to a learner, the give-up likelihood and patience, and a scripted app session
  (48 events). They were recorded on the code before the move, and the restructured
  code reproduces them exactly. The exceptions are fits, compared at 1e-3, because
  pyddm's optimiser stops a few 1e-5 apart between runs. Re-record (`--record`) only
  for a change meant to alter output, and say why in the commit.
- **Planning is about 3× faster.** One `InfoController` plan (96 settings × 6 draws
  × 25 decisions, f 0.8, k 0.8, after 26 decisions) took 15.3 s and now takes 4.6 s.
  The same speedup applies to every simulation and recovery sweep. Two changes did it:
  - A start or redo draw no longer goes through pyddm's `Solution.sample(1)`. That
    call rebuilt its sampling tables and ran its argument-checking decorators on every
    draw. `accumulator._draw` builds the tables once per condition and makes the same
    calls on numpy's global generator in the same order, so every seeded run is
    unchanged (the tests above confirm it). If pyddm is upgraded, check `_draw`
    against `Solution.sample` again.
  - `sigmoid` on a single number uses `math` instead of numpy.
- **Dependencies** are the eight packages the code imports, with pytest and ruff as dev
  tools. The other 22 of the 32 earlier pins were dependencies of dependencies, or
  optional pyddm backends this code never calls (`pybads`, `gpyreg`, `pathos`).
