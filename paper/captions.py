"""What each figure's data was generated under, as a caption, from the conditions.json of
the results it draws (experiments.common writes them). Drawn from the paper's results,
these are the paper's conditions; drawn from other results (`python -m paper --results`),
they are those results'.

    python -m paper --captions                       # print every caption, draw nothing
    python -m paper --captions --results controllers=controllers_long
"""
import json

import pandas as pd

from temporal.model import accumulator as acc

from . import explainers
from .common import RESULTS, conditions, results

TASK_APP = ("the app's task (question types with Elo ratings, skips on, InfoController "
            "choosing the settings from the default one, never shown the truth)")
TASK_SINGLE = ("the single-difficulty task (mid cold-start cost, scheduled review at rate "
               "0.25)")


def _c(name):
    c = conditions(name)
    if c is None:
        raise SystemExit(f"no conditions.json in {results(name)}: run the experiment with "
                         f"`python -m experiments.{name}`, which writes it")
    return c


# The conditions of the recovery_lhc run whose histories experiment `name` read.
def _histories(name):
    folder = _c(name)["conditions"]["histories"]
    path = RESULTS / folder / "conditions.json"
    if not path.exists():
        raise SystemExit(f"no conditions.json in {path.parent}")
    return json.loads(path.read_text()), folder


def _list(xs):
    xs = [f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _people(pairs):
    return "; ".join(f"f {f:g}, k {k:g}" for f, k in pairs)


def _grid(levels, reps):
    n, m = len(levels) ** 2 * reps, len(levels)
    return (f"{n} simulated people (f and k each at {_list(levels)}, a {m} × {m} grid, "
            f"{reps} seed{'s' * (reps != 1)} at each point)")


# The constants every simulated person and fit share, unless a caption says otherwise.
def _model(c, skip=()):
    m = c["model"]
    parts = {"g": f"γ {m['g']:g}", "a": f"α {m['a']:g}",
             "SIGMA": f"SIGMA {m['SIGMA']:g}", "BOUND": f"BOUND {m['BOUND']:g}"}
    head = "".join(f"{v}, " for k, v in parts.items() if k not in skip)
    return (f"Model constants: {head}ACHIEVEMENT_SCALE {m['ACHIEVEMENT_SCALE']:g}, "
            f"{m['bootstrap']} value bootstrap, deadline T_DUR {m['T_DUR']:g} s.")


def _changed(*names):
    notes = []
    for name in names:
        c = conditions(name)
        if c and c.get("changed"):
            notes.append(f"{name} changed from the paper's: "
                         + ", ".join(f"{k} {v}" for k, v in c["changed"].items()))
    return (" " + "; ".join(notes) + ".") if notes else ""


def _sweep(c):
    d = c["conditions"]
    w = d["warmups"]
    split = (f"from one learner recorded after {w[0]} warm-up decisions" if len(w) == 1 else
             f"split evenly across {len(w)} learners recorded after {_list(w)} warm-up decisions")
    return (f"{_grid(d['levels'], d['reps'])}, one dataset each, on {TASK_SINGLE}. Each "
            f"dataset has {d['decisions']} recorded start decisions, {split}. True "
            f"non-decision time T0 0, and T0 pinned at 0 in the fit.")


def fig4():
    c = _c("recovery_sweep")
    return (f"{_sweep(c)} f and k fitted with timed-out trials kept as undecided. "
            f"{_model(c)}{_changed('recovery_sweep')}")


def figS2():
    c = _c("recovery_sweep")
    return (f"{_sweep(c)} Each dataset fitted twice, timed-out trials kept as undecided and "
            f"dropped. {_model(c)}{_changed('recovery_sweep')}")


def _hypercube(d):
    names = {"f": "f", "k": "k", "g": "γ", "bound": "BOUND", "t0": "T0 (s)", "a": "α (log scale)",
             "lam": "λ (the share of BOUND kept under a countdown)"}
    spans = ", ".join(f"{names[k]} {lo:g} to {hi:g}" for k, (lo, hi) in d["ranges"].items())
    return f"{d['people']} simulated people drawn from a Latin hypercube (seed {d['seed']}) over {spans}"


def _lhc(c, task=TASK_APP):
    d = c["conditions"]
    return (f"{_hypercube(d)}. Each plays {task} for {d['decisions']} start decisions plus "
            f"their redo decisions.")


def fig4lhc():
    c = _c("recovery_lhc")
    prof, folder = _histories("lhc_profile")
    return (f"{_lhc(c)} f, k, γ, BOUND and T0 fitted together by maximum likelihood (the "
            f"closed-form likelihood). (f) Standard errors from the observed and the Fisher "
            f"information, and profile-likelihood intervals for the people of results/{folder}. "
            f"{_model(c, skip=('g', 'BOUND'))}{_changed('recovery_lhc', 'lhc_profile')}")


def figS8():
    lhc, folder = _histories("lhc_size")
    sizes = _c("lhc_size")["conditions"]["sizes"]
    return (f"The people of results/{folder}: {_lhc(lhc)} Each history cut at "
            f"{_list(sizes)} start decisions, with the same share of its redo decisions, and "
            f"fitted for each person alone (maximum likelihood) and pooled over people "
            f"(empirical Bayes). {_model(lhc, skip=('g', 'BOUND'))}{_changed('lhc_size')}")


def figS9():
    c = _c("alpha_reward")
    d = c["conditions"]
    return (f"{_hypercube(d)}; even-numbered people value right answers at 1 − p, odd-numbered "
            f"ones the fraction right. Each plays {TASK_APP} for {d['decisions']} start "
            f"decisions plus their redo decisions. α fitted over the grid {_list(d['alphas'])}, "
            f"under each reward, with f, k, γ, BOUND and T0. "
            f"{_model(c, skip=('g', 'a', 'BOUND'))}{_changed('alpha_reward')}")


def fig9():
    c = _c("controller_full")
    d = c["conditions"]
    return (f"{_hypercube(d)}. Each plays the app's task (question types with Elo ratings, "
            f"skips on) for {d['checkpoints'][-1]} ticks twice with the same seed: once with "
            f"the app's controller (γ 0.9 and BOUND 1.5 held), once with a controller that "
            f"fits and plans with all five parameters. All five fitted by maximum likelihood "
            f"at {_list(d['checkpoints'])} ticks in both. "
            f"{_model(c, skip=('g', 'BOUND'))}{_changed('controller_full')}")


def figS10():
    c = _c("feedback_blocks")
    d = c["conditions"]
    return (f"{_hypercube(d)}. Each plays the app's task (question types with Elo ratings, "
            f"skips on) for {d['checkpoints'][-1]} ticks three times with the same seed: with "
            f"the app's controller; with it choosing within blocks of 25 start decisions that "
            f"alternate feedback shown and hidden; and the same with the hidden blocks' redo "
            f"prompts moved to the end-of-set screen. The simulated person sees the score at "
            f"the end of a hidden set. f, k, γ, BOUND and T0 fitted by maximum likelihood at "
            f"{_list(d['checkpoints'])} ticks in every arm. "
            f"{_model(c, skip=('g', 'BOUND'))}{_changed('feedback_blocks')}")


def figS11():
    c = _c("redo_pressure")
    d = c["conditions"]
    return (f"{_hypercube(d)}. Each plays {TASK_APP} for {d['decisions']} start decisions plus "
            f"their redo decisions twice with the same seed: with no countdown, and with a "
            f"{d['deadline']:g} s countdown on the redo prompt in alternate blocks of "
            f"{d['block']} start decisions. A prompt still open when the countdown ends closes "
            f"without a redo. f, k, γ, BOUND and T0 fitted by maximum likelihood, and λ as well "
            f"with the countdown. {_model(c, skip=('g', 'BOUND'))}{_changed('redo_pressure')}")


def _held(c, part="app"):
    d = c["conditions"]
    task = (f"{TASK_APP}, {d['decisions_app']} start decisions plus their redo decisions"
            if part == "app" else f"{TASK_SINGLE}, {d['decisions_single']} start decisions")
    return (f"{len(d['people'])} simulated people ({_people(d['people'])}), {d['reps']} seeds "
            f"each, the same people and seeds at every level, on {task}. The person's γ is "
            f"set to {_list(d['g'])}, α to {_list(d['a'])}, SIGMA to {_list(d['sigma'])} or "
            f"BOUND to {_list(d['bound'])}, one at a time, and the fit holds the defaults.")


def _profile(c, task=TASK_APP):
    d = c["conditions"]
    return (f"{len(d['people'])} simulated people ({_people(d['people'])}), true γ "
            f"{_list(d['true_g'])}, {d['reps']} seeds each, on {task}, {d['decisions']} "
            f"start decisions plus their redo decisions; (f, k) fitted at each γ of the grid "
            f"{_list(d['grid'])} and the grid γ with the highest likelihood taken.")


def fig5():
    held, prof = _c("misspecification"), _c("gamma_profile")
    rob, lhc = _c("gamma_robustness"), _c("recovery_lhc")
    r = rob["conditions"]
    same = "the same task"
    return (f"γ held at 0.9: {_held(held)} γ fitted: {_profile(prof, same)} With one other "
            f"constant wrong as well: the same people, true γ {_list(r['true_g'])}, "
            f"{r['reps']} seeds, the person's α {_list(r['a'])} or BOUND {_list(r['bound'])}. "
            f"All five fitted:{_lhc(lhc, same)} {_model(held, skip=('g',))}"
            f"{_changed('misspecification', 'gamma_profile', 'gamma_robustness', 'recovery_lhc')}")


def figS7():
    held, prof = _c("misspecification"), _c("gamma_profile")
    d = held["conditions"]
    return (f"With redo: {_held(held)} No redo: the same people and seeds on {TASK_SINGLE}, "
            f"{d['decisions_single']} start decisions. γ fitted, with redo: "
            f"{_profile(prof, 'the with-redo task')} "
            f"{_model(held, skip=('g', 'a', 'SIGMA', 'BOUND'))} The fit holds γ 0.9, α 0.05, "
            f"SIGMA {acc.SIGMA:g} and BOUND {acc.BOUND:g}."
            f"{_changed('misspecification', 'gamma_profile')}")


# The `temporal recovery` sweeps save no conditions; their repetitions are read from the CSV.
def _pipelines():
    t0 = pd.read_csv(results("pipelines") / "t0.csv")
    pr = pd.read_csv(results("pipelines") / "persistence.csv")
    t0_reps = int(t0.groupby("t0_true").size().iloc[0])
    pr_people = sorted(set(zip(pr.f_true, pr.k_true)))
    pr_reps = len(pr) // len(pr_people)
    return t0, t0_reps, pr_people, pr_reps


def fig6():
    sweep = _c("recovery_sweep")
    lhc, folder = _histories("precision_full")
    t0, t0_reps, pr_people, pr_reps = _pipelines()
    return (f"(a) {_sweep(sweep)} (b) One simulated person (f 0.8, k 0.8) on {TASK_SINGLE}, "
            f"1200 start decisions, true T0 {_list(sorted(t0.t0_true.unique()))} s, {t0_reps} "
            f"seeds each; T0 pinned at 0 and fitted. (c) {len(pr_people)} simulated people "
            f"({_people(pr_people)}), {pr_reps} seeds each, the app's task with every setting "
            f"chosen at random, 300 start decisions plus their redo decisions; true patience "
            f"10 s. (d) Expected SD from the Fisher information at the truth, for the people "
            f"of results/{folder}: {_lhc(lhc)} "
            f"{_model(sweep)}{_changed('recovery_sweep', 'precision_full')}")


def figS3():
    c = _c("fit_factorial")
    d = c["conditions"]
    return (f"{_grid(d['levels'], d['reps'])} on {TASK_APP}, {d['decisions']} start decisions "
            f"plus their redo decisions; true T0 {d['t0_true']:g} s, true patience 10 s. Each "
            f"dataset fitted 8 ways, every combination of timed-out trials kept or dropped, "
            f"T0 fitted or pinned at 0, and answer and skip times used or not. The same people "
            f"and seeds in every condition. {_model(c)}{_changed('fit_factorial')}")


def _fixed_task(d):
    return (f"the question-type task with one setting held for a run of {d['ticks']} ticks "
            f"(a tick is a start decision in a session, or a chance to come back while away), "
            f"review rate {d['review']:g}, skipped questions dropped, no controller")


def _same_task(d, e):
    if list(e["people"]) == list(d["people"]) and (e["ticks"], e["review"]) == (d["ticks"], d["review"]):
        return "The same people and task"
    return f"{len(e['people'])} simulated people ({_people(e['people'])}) on {_fixed_task(e)}"


# Figure 7 (c)'s numbers: how many people pass G1 (experiments.gamma_feedback), and the
# γ 0.9 row against the same difference in the runs of (a) and (b), which are separate.
def _feedback_by_gamma():
    from experiments.gamma_feedback import paired
    from .common import se
    g = paired(pd.read_csv(results("gamma_feedback") / "runs.csv"))
    lo, hi = g.g.min(), g.g.max()
    passed = 0
    for _, p in g.groupby(["f", "k"]):
        w = p.groupby(["g", "seed"]).p_go.mean().unstack("g")
        diff = w[hi] - w[lo]
        passed += abs(diff.mean()) > 2 * se(diff)
    row = g[g.g == 0.9].groupby(["f", "k", "seed"]).p_go.mean().groupby(["f", "k"]).mean().mean()
    r = pd.read_csv(results("task_choice") / "runs.csv")
    key = ["f", "k", "target", "seed"]
    shown, hidden = r[r.part == "a"].set_index(key), r[r.part == "a_hidden"].set_index(key)
    diff = (shown.p_go - hidden.p_go).dropna().rename("d").reset_index()
    c = diff.groupby(["f", "k", "seed"]).d.mean().groupby(["f", "k"]).mean().mean()
    return passed, g[["f", "k"]].drop_duplicates().shape[0], lo, hi, row, c


def fig7():
    c, g = _c("task_choice"), _c("gamma_feedback")
    d, e = c["conditions"], g["conditions"]
    passed, n, lo, hi, row, mean_c = _feedback_by_gamma()
    return (f"(a, b) {len(d['people'])} simulated people ({_people(d['people'])}) on "
            f"{_fixed_task(d)}, at each of six difficulty targets with feedback shown and "
            f"hidden, {d['seeds_a']} seeds paired across targets and feedback. (c) "
            f"{_same_task(d, e)}, the person's γ at {_list(e['gammas'])}, "
            f"{e['seeds']} seeds paired across γ, target and feedback, in runs separate from "
            f"(a) and (b). Each cell is the shown − hidden difference in P(start) averaged "
            f"over the {n} people, hatched where it is within 2 SE (over seeds) of zero. "
            f"The effect shrinks as γ rises; averaged over targets it differed between γ "
            f"{lo:g} and {hi:g} by more than 2 SE for {passed} of the {n} people. A circle "
            f"marks each row's largest cell where that cell clears 2 SE; where the peak "
            f"falls is descriptive, since no claim about it was fixed and neighbouring "
            f"cells mostly differ by less than 2 SE. The outlined γ 0.9 row is the "
            f"condition of (a) and (b) and replicates their runs: {row:+.3f} averaged over "
            f"the row, against {mean_c:+.3f} for the same difference averaged over the "
            f"five people and six targets there. "
            f"{_model(c)}{_changed('task_choice', 'gamma_feedback')}")


def figS4():
    c = _c("task_choice")
    d = c["conditions"]
    f, k = d["person_b"]
    return (f"One simulated person (f {f:g}, k {k:g}) on {_fixed_task(d)}, at each of six "
            f"difficulty targets with feedback shown and hidden, {d['seeds_b']} seeds paired "
            f"across targets and feedback; reward by difficulty from the {len(d['people'])} "
            f"people of Figure 7, {d['seeds_a']} seeds. {_model(c)}{_changed('task_choice')}")


def figS6():
    g = _c("gamma_feedback")
    e = g["conditions"]
    return (f"{len(e['people'])} simulated people ({_people(e['people'])}) on {_fixed_task(e)}, "
            f"at each of six difficulty targets with feedback shown and hidden, the person's "
            f"γ at {_list(e['gammas'])}, {e['seeds']} seeds paired across γ, target and "
            f"feedback. {_model(g, skip=('g',))}{_changed('gamma_feedback')}")


def fig8():
    c = _c("controllers")
    d = c["conditions"]
    return (f"{_grid(d['levels'], d['reps'])} on the question-type task, skips on, "
            f"{d['checkpoints'][-1]} ticks (a tick is a start decision in a session, or a "
            f"chance to come back while away); f and k fitted at {_list(d['checkpoints'])} "
            f"ticks from the start and redo decisions recorded so far. Six ways of choosing "
            f"the settings, none shown the truth; the same people and seeds under each. "
            f"{_model(c)}{_changed('controllers')}")


figS5 = fig8


def fig1():
    return "Schematic; no data."


def fig2():
    f, k = explainers.PLANE_TRUE
    decisions = "; ".join(f"value {v:g}, cost {c:g}" for v, c in explainers.PLANE_DECISIONS)
    return (f"No simulation: exact drifts for one person (f {f:g}, k {k:g}) at three "
            f"decisions ({decisions}).")


def fig3():
    return ("From a recorded human session (the decisions.jsonl named by $TEMPORAL_RECORD, "
            "default data/decisions.jsonl), not a simulation.")


def identifiability():
    return fig2().replace("No simulation", "No simulation, the decisions of Figure 2")


CAPTIONS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig4lhc": fig4lhc,
            "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8, "fig9": fig9,
            "figS2": figS2, "figS3": figS3, "figS4": figS4, "figS5": figS5, "figS6": figS6,
            "figS7": figS7, "figS8": figS8, "figS9": figS9, "figS10": figS10, "figS11": figS11,
            "identifiability": identifiability}


# Which experiments' results each figure reads, for `--results`.
READS = {"fig4": ("recovery_sweep",), "figS2": ("recovery_sweep",),
         "fig4lhc": ("recovery_lhc", "lhc_profile"),
         "fig5": ("misspecification", "gamma_profile", "gamma_robustness", "recovery_lhc"),
         "figS7": ("misspecification", "gamma_profile"),
         "fig6": ("recovery_sweep", "pipelines", "precision_full"),
         "figS3": ("fit_factorial",), "fig7": ("task_choice", "gamma_feedback"),
         "figS4": ("task_choice",), "figS6": ("gamma_feedback",),
         "fig8": ("controllers",), "figS5": ("controllers",), "fig9": ("controller_full",),
         "figS8": ("lhc_size",), "figS9": ("alpha_reward",),
         "figS10": ("feedback_blocks",), "figS11": ("redo_pressure",)}
