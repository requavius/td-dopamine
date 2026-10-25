"""Paper Figures 1 (the model), 4 (recovery from one person) and S2 (timed-out trials kept
vs dropped). Figures 4 and S2 read results/recovery_sweep/runs.csv
(`python -m experiments.recovery_sweep`): gamma 0.9, ACHIEVEMENT_SCALE 4, continuing
bootstrap, T_DUR 7.0 s, T0 pinned at 0, 1200 recorded decisions per fit split across four
learners, 4x4 grid x 3 reps.

    python -m paper fig1 fig4 figS2
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from temporal.model.accumulator import BOUND, SIGMA, T_DUR

from .common import OUT, RESULTS

LEVELS = (0.2, 0.6, 1.0, 1.5)

# Categorical slots 1 and 2 of the dataviz reference palette (CVD-validated pair).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#000000", "#3a3a3a", "#d9d9d9"

# Helvetica, with Arial Unicode MS for any glyph it lacks (Greek, math symbols).
RC = {
    "font.family": ["Helvetica", "Arial Unicode MS"], "font.size": 11,
    "axes.edgecolor": INK, "axes.linewidth": 1.2, "axes.labelcolor": INK,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.direction": "out",
    "ytick.direction": "out", "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
}


def _panel(ax, letter):
    ax.text(-0.2, 1.1, f"({letter})", transform=ax.transAxes, fontsize=15, color=INK,
            ha="left", va="bottom")


def _rmse(est, truth):
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def _identity(ax, lo=0.0, hi=1.8):
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks(LEVELS)


# Figure 4: true vs recovered, timed-out trials kept.
def fig_recovery(res):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    stats = {}
    for ax, p in zip(axes, ("f", "k")):
        truth, est = res[f"{p}_true"].to_numpy(), res[f"{p}_kept"].to_numpy()
        stats[p] = _rmse(est, truth)
        rng = np.random.default_rng(0)
        ax.scatter(truth + rng.uniform(-0.04, 0.04, len(truth)), est, s=22, color=BLUE,
                   alpha=0.55, edgecolors="white", linewidths=0.6, zorder=3)
        means = res.groupby(f"{p}_true")[f"{p}_kept"].mean()
        ax.plot(means.index, means.to_numpy(), color=INK, lw=0, marker="_", ms=18,
                mew=2, zorder=4)
        _identity(ax)
        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(f"recovered {p}")
        ax.text(1.75, 0.08, f"RMSE {stats[p]:.3f}", ha="right", va="bottom", color=INK)
        _panel(ax, "ab"[p == "k"])
    fig.tight_layout()
    fig.savefig(OUT / "fig4_recovery.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats


# Figure S2: the same datasets, timed-out trials kept vs dropped.
def fig_timeouts(res):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    stats = {}
    for ax, p in zip(axes, ("f", "k")):
        truth = res[f"{p}_true"].to_numpy()
        for col, color, dx, name in ((f"{p}_kept", BLUE, -0.035, "timeouts kept"),
                                     (f"{p}_drop", ORANGE, 0.035, "timeouts dropped")):
            g = res.groupby(f"{p}_true")[col]
            m, sd = g.mean(), g.std()
            ax.errorbar(m.index + dx, m.to_numpy(), yerr=sd.to_numpy(), fmt="o", ms=6,
                        color=color, ecolor=color, elinewidth=1.5, capsize=0,
                        mec="white", mew=1, zorder=3, label=name)
            est = res[col].to_numpy()
            stats[col] = dict(rmse=_rmse(est, truth),
                              bias_pct=float(np.mean((est - truth) / truth) * 100))
        _identity(ax)
        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(f"recovered {p}")
        _panel(ax, "ab"[p == "k"])
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "figS2_timeouts.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats


def _box(ax, x, y, w, h, text, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                fc="#e3eefb" if accent else "white", ec=INK, lw=2.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=13, color=INK,
            linespacing=1.3)


def _arrow(ax, a, b, rad=0.0, label=None, lpos=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                 color=INK, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=4, shrinkB=4))
    if label:
        ax.text(*lpos, label, ha="center", va="center", fontsize=12, color=INK)


# (a) the loop: a task, the decisions it draws out, the model, and the person's f and k
# feeding back into the task.
def _panel_loop(ax):
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    W, H, Y = 2.9, 1.5, 0.9
    xs = (0.3, 4.5, 8.7, 12.8)
    labels = ("Task", "Start or stop?\n+ reaction time", "Model\n(TD + DDM)", "f and k")
    for x, text in zip(xs, labels):
        _box(ax, x, Y, W, H, text, accent=text == "f and k")
    for x0, x1 in zip(xs, xs[1:]):
        _arrow(ax, (x0 + W, Y + H / 2), (x1, Y + H / 2))
    _arrow(ax, (xs[3] + W / 2, Y), (xs[0] + W / 2, Y), rad=-0.12, label="adjust the task",
           lpos=(8.0, 0.05))
    ax.text(-0.02, 1.0, "(a)", transform=ax.transAxes, fontsize=15, va="bottom")


# (b) the start decision for two people facing the same task: example accumulator paths.
def _panel_paths(ax):
    dt = 0.005
    rng = np.random.default_rng(3)
    people = ((1.2, 0.4, BLUE, "high f"), (0.4, 1.5, ORANGE, "high k"))
    value, cost = 1.0, 0.6
    for f, k, color, _ in people:
        drift = f * value - k * cost
        for i in range(4):
            x, xs = 0.0, [0.0]
            while abs(x) < BOUND and len(xs) * dt < T_DUR:
                x += drift * dt + SIGMA * np.sqrt(dt) * rng.standard_normal()
                xs.append(x)
            xs = np.clip(xs, -BOUND, BOUND)
            t = np.arange(len(xs)) * dt
            ax.plot(t, xs, color=color, lw=1.4, alpha=0.9)
            if abs(xs[-1]) >= BOUND:
                ax.plot(t[-1], xs[-1], "o", color=color, ms=6, mec="white", mew=1, zorder=4)
    ax.text(2.15, BOUND - 0.3, "high f", color=BLUE, fontsize=12, va="center")
    ax.text(5.15, -BOUND + 0.3, "high k", color=ORANGE, fontsize=12, va="center")
    for y, text in ((BOUND, "start"), (-BOUND, "stop")):
        ax.axhline(y, color=INK, lw=1.6)
        ax.text(-0.15, y, text, ha="right", va="center", fontsize=12)
    ax.axvline(T_DUR, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(T_DUR, BOUND + 0.45, "deadline", ha="center", va="bottom", fontsize=11, color=INK2)
    ax.set_xlim(0, T_DUR + 0.2)
    ax.set_ylim(-BOUND - 0.5, BOUND + 0.5)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.spines["left"].set_visible(False)
    ax.set_title("drift = f · value − k · cost", fontsize=13, pad=10)
    _panel(ax, "b")


# (c) the "just hard enough" curve: a right answer is worth 1 - p and comes with chance p.
def _panel_reward(ax):
    p = np.linspace(0, 1, 201)
    ax.plot(p, p * (1 - p), color=BLUE, lw=2.6)
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("chance right, p")
    ax.set_ylabel("expected reward")
    ax.set_title("reward = p(1 − p)", fontsize=13, pad=10)
    _panel(ax, "c")


# Figure 1: (a) the loop, (b) the decision, (c) what makes a task worth starting.
def fig_model():
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=(1, 1.5), width_ratios=(1.5, 1),
                          hspace=0.45, wspace=0.3)
    _panel_loop(fig.add_subplot(gs[0, :]))
    _panel_paths(fig.add_subplot(gs[1, 0]))
    _panel_reward(fig.add_subplot(gs[1, 1]))
    fig.savefig(OUT / "fig1_model.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _setup():
    plt.rcdefaults()
    plt.rcParams.update(RC)
    OUT.mkdir(parents=True, exist_ok=True)


def fig1():
    _setup()
    fig_model()


def fig4():
    _setup()
    print("Figure 4 RMSE:", fig_recovery(pd.read_csv(RESULTS / "recovery_sweep" / "runs.csv")))


def figS2():
    _setup()
    print("Figure S2:", fig_timeouts(pd.read_csv(RESULTS / "recovery_sweep" / "runs.csv")))
