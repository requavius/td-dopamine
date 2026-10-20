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

| version | dates | drift | reward | deadline `T_DUR` | value bootstrap | fitted |
|---|---|---|---|---|---|---|
| V0 | 2026-03 to 2026-05 | engagement score with boredom `b` and effort cost, no accumulator | stages completed | none | none | particle filter |
| V1 | 2026-05 to 2026-09-13 | `f·value − k·cost`, plus `b_seen·seen_miss + b_unseen·unseen_miss + b_fb·feedback` from mid-2026 | fraction right, then hard right answers `4·Σ(1 − p)/n` | 3.0 s | terminal (`V_next = 0`) | 5 parameters by pyddm, `T0` pinned at 0 |
| V2 | 2026-09-13 to 2026-09-21 | as V1 | hard right answers | 7.0 s | continuing | as V1; `T0` free for real people from 2026-09-15 |
| **V3 (current)** | since 2026-09-21 | `f·value − k·cost`; prediction error carried in `value`; reward lands when shown | hard right answers | 7.0 s | continuing | 2 parameters (`f`, `k`), `T0` free for real people |

Changes since 2026-09-21 that did not change the model: the information-seeking
controller and redo-prompt settings (2026-10-11), the package restructure (2026-10-11,
regression tests reproduce every seeded output).

`f` values are on different scales across versions: `f` is identified only up to γ,
`ACHIEVEMENT_SCALE` and the bootstrap, all of which changed between V1 and V2.

---

## V3, superseded by later V3 measurements

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

### Patience on the first player's record (superseded 2026-10-15)

"Their fitted patience is 2.8 s (−log L 38, against 55 at 10 s)." The `f` and `k` it was
computed with are not recorded, and it does not reproduce under the current code: with
the controller's estimate on that record (the prior, `f = k = 0.8`, since 26 decisions
are under its 40-row floor) patience is 1.7 s (−log L 73.9 against 102.2 at 10 s). The
record now also counts leaving from a question as giving up (65 questions, 14 given up).

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

## V2 (`T_DUR = 7.0`, continuing bootstrap, reaction terms still in the drift)

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

### The first player's decisions refitted at the new deadline (five-parameter fit)

| deadline | decided / undecided | f | k | b_seen | b_unseen | b_fb |
|---|---|---|---|---|---|---|
| 3.0 s | 23 / 3 | 2.34 | 0.69 | +0.26 | +0.19 | +0.20 |
| 7.0 s | 26 / 0 | 1.76 | 0.49 | +0.30 | +0.95 | +0.18 |

Three recoded decisions moved `f` by 25% and `k` by 29%. `b_unseen` was unconstrained
(no set with feedback hidden). A five-parameter fit on this data took 38 s instead of 17.

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

## V1 (`T_DUR = 3.0`, terminal bootstrap)

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
- **The joint fit on the first real player** moved `f` from 3.0 to 0.0 (fast skips pulling
  it down), under rules before the record was rebuilt. On the current record and model
  the pull is 2.75 → 2.15 (README, "Skips do not help").
- **Elo serving.** On a person perfect at addition and ~75% at 2-digit × 1-digit
  multiplication, 60 answers at a 75% target stopped serving addition; 11 of the last 20
  questions were 2-digit × 1-digit. Measured before the first-showing rule.

### Planning time before the 2026-10-11 speedup

Over 84 settings at `T_DUR = 7.0`, about 150 decisions of history: 21 to 27 s with known
parameters, 60 to 250 s with a refit. Replaced by the README's 4.6 s.

---

## V0 (engagement score, particle filter)

The first model scored engagement from `f`, an effort cost and a boredom rate `b`, and
estimated them with a particle filter. Recovery failed: in the last recorded sweep `b` came back near 0.94 whatever its true
value. Replaced by the drift-diffusion model in May 2026.

---

## Implementations that did not pan out

| implementation | model | what replaced it | why |
|---|---|---|---|
| particle filter for `f`, `k`, `b` | V0 | pyddm maximum likelihood | did not recover the parameters |
| boredom rate `b`, time-since-engagement cost | V0 | boredom as sustained `|δ| ≈ 0` (V3) | no mechanism; not recoverable |
| fraction-right reward | V1 | hard right answers `4·Σ(1 − p)/n` | made the easiest setting always the most rewarding |
| `difficulty` as a TD feature | V1 | `φ = [1, stage position]` | collinear with the intercept, ~1% of `V(0)` |
| 3-second countdown on prompts | V1 interface | prompts wait; slow choices count as undecided at fit time | invented 20 consecutive "decisions" in one session |
| `T_DUR = 3.0` | V1 | 7.0 s | discarded 11.5% of a real player's choices |
| terminal value bootstrap | V1 | continuing bootstrap | the task is a cycle; `f` recovered worse |
| `T0` pinned at 0 for real people | V1 | fitted in [0, 0.5] s | biased `f` and `k` 8 to 15% low |
| reaction terms `b_seen`, `b_unseen`, `b_fb` (five-parameter fit) | V1, V2 | prediction error carried in value; reward lands when shown | patched a learner that produced no mid-set error; no mechanism |
| `f` RMSE ≈ 0.035 pass condition for one learner | V2 | `f`, `k` RMSE ≤ 0.05 | belonged to a designed, not self-generated, value range |
| seeding learners across the learning curve to widen value | V2 | kept for the paper's design, not as a fix | did nothing detectable for `f` |
| joint fit of decisions with answer and skip times | V2, V3 | decisions for `f` and `k`; times for patience only | made `f` worse; moved a real player's `f` from 3.0 to 0.0 |
| reward-planning `Controller` as the app's controller | V2, V3 | `InfoController` (2026-10-11) | left `k` poorly measured (RMSE 0.22 to 0.29) |
| 84 settings, rollouts always scheduling review | V3 | 96 settings, rollouts follow each setting's review mode | planned a task the person never saw |
