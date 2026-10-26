"""Paper Figure 8: how fast each way of choosing settings measures k, and why (the spread of
cost each collects). Figure S5 adds f and what each costs in engagement. Both from
results/controllers (`python -m experiments.controllers`). The 32
people of the 4 x 4 grid x 2 seeds, question-type task, skips on, 300 ticks (start
decisions in a session, or chances to come back while away), true f and k hidden; V4,
gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s. Error
bars are 2 SE over the 32 people (bootstrap for RMSE).

    python -m paper fig8 figS5
"""
import numpy as np
import pandas as pd

from . import style as fs
from .common import OUT, RESULTS, se

CHECKPOINTS = (100, 200, 300)
ORDER = ("fixed", "random", "reward_scheduled", "reward", "information")
LABELS = {"fixed": "fixed", "random": "random", "reward_scheduled": "reward",
          "reward": "reward + redo", "information": "information"}


def fig8():
    fs.apply()
    df = pd.read_csv(RESULTS / "controllers" / "runs.csv")
    fig, (ax, cx) = fs.figure(1, 2, width=7.4, height=3.6)
    _draw(df, ax=ax, cx=cx)
    fig.subplots_adjust(wspace=0.4, top=0.8)
    fs.save(fig, OUT / "fig8_controllers")


def figS5():
    fs.apply()
    df = pd.read_csv(RESULTS / "controllers" / "runs.csv")
    fig, axes = fs.figure(2, 2, width=7.4, height=6.4)
    (ax, bx), (cx, dx) = axes
    _draw(df, ax=ax, bx=bx, cx=cx, dx=dx)
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.9)
    fs.save(fig, OUT / "figS5_controllers")


def _draw(df, ax, cx, bx=None, dx=None):
    """(a) k RMSE, (b) f RMSE, (c) spread of cost, (d) engagement; panels left as None are
    skipped and the rest lettered in order."""
    fig = ax.figure
    # Grey ink for the comparison arms; blues for the planners, dark to light: reward + redo,
    # reward (84 settings, diamonds), information (the app's controller).
    style = {"fixed": ("#8C8C8C", "--", "s"), "random": ("#8C8C8C", ":", "^"),
             "reward_scheduled": (fs.MID, "--", "D"), "reward": (fs.DARK, "-", "o"),
             "information": (fs.LIGHT, "-", "o")}
    # Error bars are 2 SE over the 32 people: bootstrap (2000 resamples, fixed seed) for
    # RMSE, the usual SE of the mean otherwise. Arms are dodged sideways so bars don't overlap.
    def line(axis, j, arm, ys, ses, label=None):
        col, ls, mk = style[arm]
        xs = [c + (j - 2) * 7 for c in CHECKPOINTS]
        axis.errorbar(xs, ys, yerr=[2 * e for e in ses], color=col, ls=ls, marker=mk, ms=6,
                      mec="black", mew=0.6, capsize=0, elinewidth=1.2, label=label)

    def rmse_se(err):
        rng = np.random.default_rng(0)
        boots = [np.sqrt((rng.choice(err, len(err)) ** 2).mean()) for _ in range(2000)]
        return float(np.std(boots))

    for axis, p in ((ax, "k"), (bx, "f")):
        if axis is None:
            continue
        for j, arm in enumerate(ORDER):
            d = df[df.arm == arm]
            errs = [(d[f"{p}_{n}"] - d[f"{p}_true"]).to_numpy() for n in CHECKPOINTS]
            line(axis, j, arm, [np.sqrt((e ** 2).mean()) for e in errs],
                 [rmse_se(e) for e in errs], LABELS[arm])
        axis.axhline(0.05, color="black", lw=1.0, ls=":")
        axis.set_xticks(CHECKPOINTS)
        axis.set_xlabel("Ticks")
        axis.set_ylabel(f"${p}$ RMSE")
        axis.set_ylim(bottom=0)
    # (c) why: how much cost varied in the data each arm collected
    for j, arm in enumerate(ORDER):
        d = df[df.arm == arm]
        line(cx, j, arm, [d[f"csd_{n}"].mean() for n in CHECKPOINTS],
             [se(d[f"csd_{n}"]) for n in CHECKPOINTS])
    cx.set_xticks(CHECKPOINTS)
    cx.set_xlabel("Ticks")
    cx.set_ylabel("Spread of cost (sd)")
    cx.set_ylim(bottom=0)
    # (d) what it cost: engagement in each block of 100 ticks
    if dx is not None:
        for j, arm in enumerate(ORDER):
            d = df[df.arm == arm]
            line(dx, j, arm, [d[f"pgo_{n}"].mean() for n in CHECKPOINTS],
                 [se(d[f"pgo_{n}"]) for n in CHECKPOINTS])
        dx.set_xticks(CHECKPOINTS)
        dx.set_xticklabels(["1-100", "101-200", "201-300"])
        dx.set_xlabel("Ticks (block)")
        dx.set_ylabel("P(start next set)")
        dx.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               fontsize=10, handlelength=2.8, columnspacing=1.0)
    for axis, letter in zip([a for a in (ax, bx, cx, dx) if a is not None], "abcd"):
        fs.panel_label(axis, letter)
