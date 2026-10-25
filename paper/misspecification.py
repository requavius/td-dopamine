"""Paper Figure 5: bias in f, k and f/k when the person's discount factor, learning rate, noise
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

    python -m paper fig5
"""
import pandas as pd

from . import style as fs
from .common import OUT, RESULTS, se

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


def fig5():
    fs.apply()
    runs = with_defaults(pd.read_csv(RESULTS / "misspecification" / "runs.csv"))

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
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=10, handlelength=1.8, columnspacing=1.2)
    fs.save(fig, OUT / "fig5_misspecification")
