"""Paper Figures 5 and S7.

Figure S7: bias in f, k and f/k when the person's discount factor, learning rate, noise
or bound differ from the constants the fit holds, from results/misspecification
(`python -m experiments.misspecification`). The fit keeps gamma 0.9, alpha 0.05, SIGMA 0.7,
BOUND 1.5, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s. 3 people x 5 seeds per
level; error bars are 2 SE over the 15 fits.

Three lines. "Analyst" fits value computed at the defaults, as the app does; "oracle"
fits the person's own value, and is drawn for gamma and alpha, the only constants it changes.
    no redo      single-diff task (Figure 4's), 1200 start decisions
    with redo    question-type task, skips on, InfoController, 300 start decisions plus
                 their redo decisions. Its oracle is left out: f and k stay within 5% of
                 the truth at every gamma and alpha (f/k within 10%).

A fourth line in column (a), "γ fitted, with redo", is experiments.gamma_profile: the
same task as "with redo", γ fitted (profile likelihood over the grid 0.5 to 0.99) instead
of held at 0.9; 3 people × 5 seeds per true γ (0.7, 0.8, 0.9, 0.95). The other lines in
column (a) show γ held at 0.9 when the person's differs, which is how every other result
in the paper was fitted.

Figure 5: what stays biased once γ is fitted. Bias in f and k, mean and 95% t interval
over runs, when the person's γ, learning rate α or bound differ from the fit's.
    γ held at 0.9   experiments.misspecification, "with redo" (3 people x 5 seeds); in
                    (b) the person's γ is 0.9, so holding it is correct there
    γ fitted        experiments.gamma_profile (true γ 0.7 to 0.95, defaults otherwise) and
                    experiments.gamma_robustness (true γ 0.8 and 0.95, one other constant
                    wrong, 3 people x 2 seeds), each at its best grid γ
    all five fitted experiments.recovery_lhc (f, k, γ, BOUND, T0 by closed-form maximum
                    likelihood), 100 people binned by true γ or true bound
The same task throughout: question types, skips on, InfoController, 300 start decisions
plus their redo decisions; ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s.

    python -m paper fig5 figS7
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import style as fs
from .common import out, results, se

FACTORS = [("g", r"$\gamma$", 0.9), ("a", r"learning rate $\alpha$", 0.05),
           ("sigma", r"$\sigma$", 0.7), ("bound", "bound", 1.5)]
ROWS = [("f", r"Bias in $\hat{f}$ (%)"), ("k", r"Bias in $\hat{k}$ (%)"),
        ("ratio", r"Bias in $\hat{f}/\hat{k}$ (%)")]
# (part, fit, label, colour, line, marker, constants it is drawn for)
LINES = [
    ("single", "analyst", "analyst, no redo", fs.DARK, "-", "o", "g a sigma bound"),
    ("single", "oracle", "oracle, no redo", fs.LIGHT, "--", "o", "g a"),
    ("app", "analyst", "analyst, with redo", fs.MID, "-", "s", "g a sigma bound"),
]


# The default runs are saved once, under gamma; they are every constant's default level.
def with_defaults(df):
    base = df[(df.factor == "g") & (df.level == 0.9)]
    return pd.concat([df] + [base.assign(factor=fa, level=d) for fa, _, d in FACTORS[1:]],
                     ignore_index=True)


def _bias(sub, row, fit):
    if row == "ratio":
        est = sub[f"f_{fit}"] / sub[f"k_{fit}"]
        return 100 * (est / (sub.f_true / sub.k_true) - 1)
    return 100 * (sub[f"{row}_{fit}"] - sub[f"{row}_true"]) / sub[f"{row}_true"]


# gamma_profile: each run's fit at its best grid γ, as bias in f, k and f/k by true γ.
def gamma_fitted():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(results("gamma_profile") / "runs_*.csv"))],
                   ignore_index=True)
    best = df.loc[df.groupby(["f_true", "k_true", "g_true", "rep"]).nll.idxmin()]
    return best.rename(columns={"f_hat": "f_fitted", "k_hat": "k_fitted", "g_true": "level"})


def figS7():
    fs.apply()
    runs = with_defaults(pd.read_csv(results("misspecification") / "runs.csv"))
    fitted = gamma_fitted()

    fig, axes = fs.figure(3, 4, width=7.4, height=7.8, sharex="col", sharey="row")
    for col, (factor, label, default) in enumerate(FACTORS):
        for r, (row, ylabel) in enumerate(ROWS):
            ax = axes[r, col]
            ax.axhline(0, color=fs.INK, lw=1)
            ax.axvline(default, color=fs.GREY, ls="--", lw=1.2)
            for part, fit, name, color, ls, marker, drawn in LINES:
                if factor not in drawn.split():
                    continue
                sub = runs[(runs.part == part) & (runs.factor == factor)]
                # f/k is undefined where a fit put k at 0 (start decisions only, gamma 0.99).
                if row == "ratio":
                    sub = sub.groupby("level").filter(lambda g: (g[f"k_{fit}"] > 1e-6).all())
                g = _bias(sub, row, fit).groupby(sub.level)
                ax.errorbar(g.mean().index, g.mean().values, yerr=2 * g.agg(se).values,
                            color=color, ls=ls, marker=marker, ms=6, mec="black", mew=0.6,
                            capsize=0, label=name)
            if factor == "g":
                g = _bias(fitted, row, "fitted").groupby(fitted.level)
                ax.errorbar(g.mean().index, g.mean().values, yerr=2 * g.agg(se).values,
                            color=fs.INK, ls=":", marker="D", ms=5, mec="black", mew=0.6,
                            capsize=0, label=r"$\gamma$ fitted, with redo")
            if factor == "a":
                ax.set_xscale("log")
                ax.set_xticks([0.01, 0.05, 0.2], ["0.01", "0.05", "0.2"])
                ax.minorticks_off()
            if r == len(ROWS) - 1:
                ax.set_xlabel(label)
            if r == 0:
                fs.panel_label(ax, "abcd"[col])
            if col == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.9)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               fontsize=10, handlelength=1.8, columnspacing=1.2)
    fs.save(fig, out("figS7_misspecification_pinned"))


def _ci(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


# gamma_robustness: each run at its best grid γ, with the constant it changed.
def robustness():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(results("gamma_robustness") / "runs_*.csv"))],
                   ignore_index=True)
    run = ["factor", "level", "f_true", "k_true", "g_true", "rep"]
    return df.loc[df.groupby(run).nll.idxmin()]


def fig5():
    fs.apply()
    held = pd.read_csv(results("misspecification") / "runs.csv")
    held = held[held.part == "app"]
    held = held.rename(columns={"f_analyst": "f_hat", "k_analyst": "k_hat"})
    prof = gamma_fitted().rename(columns={"f_fitted": "f_hat", "k_fitted": "k_hat",
                                          "level": "g_true"})
    rob = robustness()
    lhc = pd.read_csv(results("recovery_lhc") / "fits.csv")

    def bias(d, p):
        return 100 * (d[f"{p}_hat"] - d[f"{p}_true"]) / d[f"{p}_true"]

    COLS = [("g", r"person's $\gamma$"), ("a", r"person's learning rate $\alpha$"),
            ("bound", "person's bound")]
    # (label, colour, line, marker)
    HELD = (r"$\gamma$ held at 0.9", fs.GREY, "--", "o")
    FIT = {0.8: (r"$\gamma$ fitted, true $\gamma$ 0.8", fs.DARK, "-", "s"),
           0.95: (r"$\gamma$ fitted, true $\gamma$ 0.95", fs.LIGHT, "-", "s")}
    LHC = ("all five fitted", fs.MID, ":", "D")

    fig, axes = fs.figure(2, 3, width=6.6, height=5.6, sharey="row")

    def draw(ax, xs, d, p, style, dx=0.0):
        col, ls, mk = style[1:]
        m = [_ci(bias(g, p)) for g in d]
        ax.errorbar(np.asarray(xs, float) + dx, [a for a, _ in m], yerr=[b for _, b in m],
                    color=col, ls=ls, marker=mk, ms=6, mec="black", mew=0.6, capsize=0,
                    elinewidth=1.2, label=style[0])

    for r, p in enumerate(("f", "k")):
        # (a) γ: held against fitted (profile), and all five fitted, by true γ
        ax = axes[r, 0]
        lv = sorted(held[held.factor == "g"].level.unique())
        draw(ax, lv, [held[(held.factor == "g") & (held.level == v)] for v in lv], p, HELD)
        lv = sorted(prof.g_true.unique())
        draw(ax, lv, [prof[prof.g_true == v] for v in lv], p, (r"$\gamma$ fitted, other constants correct",
             fs.INK, "-", "^"))
        edges = np.linspace(0.6, 0.97, 5)
        bins = pd.cut(lhc.true_g, edges)
        lhc_ = lhc.rename(columns={"est_f": "f_hat", "est_k": "k_hat", "true_f": "f_true",
                                   "true_k": "k_true"})
        draw(ax, (edges[:-1] + edges[1:]) / 2, [lhc_[bins == b] for b in bins.cat.categories],
             p, LHC)
        # (b) α: γ correct and held (true γ 0.9), and γ fitted at true γ 0.8 and 0.95
        ax = axes[r, 1]
        lv = [0.01, 0.025, 0.05, 0.1, 0.2]
        base = held[held.factor == "g"]
        draw(ax, lv, [held[(held.factor == "a") & (held.level == v)] if v != 0.05 else
                      base[base.level == 0.9] for v in lv], p, HELD)
        for j, g in enumerate((0.8, 0.95)):
            ra = rob[(rob.factor == "a") & (rob.g_true == g)]
            d = [ra[ra.level.astype(float) == v] if v != 0.05 else prof[prof.g_true == g]
                 for v in lv]
            draw(ax, lv, d, p, FIT[g])
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.05, 0.2], ["0.01", "0.05", "0.2"])
        ax.minorticks_off()
        # (c) bound: γ fitted with the bound held at 1.5, and all five fitted
        ax = axes[r, 2]
        lv = [1.0, 1.5, 2.0]
        for j, g in enumerate((0.8, 0.95)):
            rb = rob[(rob.factor == "bound") & (rob.g_true == g)]
            d = [rb[rb.level.astype(float) == v] if v != 1.5 else prof[prof.g_true == g]
                 for v in lv]
            draw(ax, lv, d, p, FIT[g])
        edges = np.linspace(1.0, 2.0, 5)
        bins = pd.cut(lhc.true_bound, edges, include_lowest=True)
        draw(ax, (edges[:-1] + edges[1:]) / 2, [lhc_[bins == b] for b in bins.cat.categories],
             p, LHC)

    defaults = {"g": 0.9, "a": 0.05, "bound": 1.5}
    for c, (key, label) in enumerate(COLS):
        for r in range(2):
            ax = axes[r, c]
            ax.axhline(0, color=fs.INK, lw=1)
            ax.axvline(defaults[key], color=fs.GREY, ls=":", lw=1.0)
            if r == 0:
                ax.tick_params(labelbottom=False)
                fs.panel_label(ax, "abc"[c])
            else:
                ax.set_xlabel(label)
        if key == "bound":
            for r in range(2):
                axes[r, c].set_xticks([1.0, 1.5, 2.0])
    axes[0, 0].set_ylabel(r"Bias in $\hat{f}$ (%)")
    axes[1, 0].set_ylabel(r"Bias in $\hat{k}$ (%)")
    seen = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.subplots_adjust(top=0.84, hspace=0.18, wspace=0.18)
    fig.legend(seen.values(), seen.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=10, handlelength=2.2, columnspacing=1.2)
    fs.save(fig, out("fig5_misspecification"))


def figS9():
    """Supplementary Figure S9: the learning rate and the reward a person values, measured
    as γ is, from results/alpha_reward (`python -m experiments.alpha_reward`). People from
    a Latin hypercube over f, k, γ, BOUND, T0 and log α; half value right answers at
    1 − p (achievement), half the fraction right (accuracy). (a) α̂ under the person's true
    reward against the true α, with its 95% profile-likelihood interval. (b) the evidence
    for the true reward: the lower profile minimum of the wrong reward minus that of the
    true one (positive: the true reward is chosen). (c) bias in f, mean and 95% t interval:
    the current fit (α 0.05, achievement), α fitted with achievement assumed, and α and
    the reward both chosen by the profile."""
    from experiments.alpha_reward import per_person

    fs.apply()
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(results("alpha_reward") / "runs_*.csv"))],
                   ignore_index=True)
    p = per_person(df)
    style = {"achievement": (fs.MID, "o", "values 1 $-$ p"),
             "accuracy": (fs.LIGHT, "s", "values the fraction right")}
    fig, (a, b, c) = fs.figure(1, 3, width=9.6, height=3.4)
    for r, (col, mk, name) in style.items():
        g = p[p.reward_true == r]
        est, lo, hi = g[f"a_hat_{r}"], g[f"a_lo_{r}"], g[f"a_hi_{r}"]
        a.errorbar(g.a_true, est, yerr=[est - lo, hi - est], ls="none", marker=mk, ms=5,
                   color=col, mec="black", mew=0.5, elinewidth=0.6, capsize=0, label=name,
                   ecolor=col, alpha=0.9)
    a.plot([0.008, 0.25], [0.008, 0.25], color=fs.GREY, ls="--", lw=1.2, zorder=0)
    a.set_xscale("log")
    a.set_yscale("log")
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_locator(plt.FixedLocator([0.01, 0.03, 0.1, 0.2]))
        axis.set_major_formatter(plt.FixedFormatter(["0.01", "0.03", "0.1", "0.2"]))
        axis.set_minor_locator(plt.NullLocator())
    a.set_xlabel(r"true $\alpha$")
    a.set_ylabel(r"recovered $\alpha$")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2,
               fontsize=10, handletextpad=0.3)
    rng = np.random.default_rng(0)
    for j, (r, (col, mk, name)) in enumerate(style.items()):
        g = p[p.reward_true == r]
        ev = g.lr_reward if r == "achievement" else -g.lr_reward
        b.scatter(j + rng.uniform(-0.15, 0.15, len(ev)), ev, s=14, color=col, marker=mk,
                  edgecolors="black", linewidths=0.4)
    b.axhline(0, color="black", lw=1.2, ls="--")
    b.set_yscale("symlog", linthresh=1)
    b.set_yticks([-10, -1, 0, 1, 10, 100], ["−10", "−1", "0", "1", "10", "100"])
    b.yaxis.set_minor_locator(plt.NullLocator())
    b.set_xticks([0, 1], ["1 $-$ p", "fraction\nright"])
    b.set_xlim(-0.6, 1.6)
    b.set_xlabel("reward the person values")
    b.set_ylabel(r"$-\log L$, wrong $-$ true reward")
    methods = (("current", "current"), ("afit_achievement", r"$\alpha$ fitted"),
               ("both", "both chosen"))
    for j, (r, (col, mk, name)) in enumerate(style.items()):
        g = p[p.reward_true == r]
        for i, (m, _) in enumerate(methods):
            rel = 100 * (g[f"f_{m}"] - g.true_f) / g.true_f
            mean, half = rel.mean(), stats.t.ppf(0.975, len(rel) - 1) * rel.std(ddof=1) / np.sqrt(len(rel))
            c.errorbar(i + (j - 0.5) * 0.25, mean, yerr=half, color=col, marker=mk, ms=7,
                       mec="black", mew=0.6, capsize=0, elinewidth=1.4)
    c.axhline(0, color="black", lw=1.2)
    c.set_xticks(range(len(methods)), [lab for _, lab in methods], fontsize=10)
    c.set_ylabel(r"Bias in $\hat{f}$ (%)")
    for axis, letter in zip((a, b, c), "abc"):
        fs.panel_label(axis, letter)
    fig.subplots_adjust(wspace=0.45)
    fs.save(fig, out("figS9_alpha_reward"))
