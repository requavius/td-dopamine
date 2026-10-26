"""Paper Figures 7 (which difficulty keeps a person engaged, and what showing feedback does,
five people) and S4 (the same for f = k = 0.8 with 96 paired seeds), from
results/task_choice (`python -m experiments.task_choice`). Question-type task, one setting
held for each run of 300 ticks, review 0.25, skipped questions dropped, no controller; V4,
gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s. Error bars
are 2 SE over seeds.

    python -m paper fig7 figS4
"""
import pandas as pd

from . import style as fs
from .common import OUT, RESULTS, se

TARGETS = (0.9, 0.75, 0.6, 0.45, 0.3, 0.2)


def fig7():
    fs.apply()
    runs = pd.read_csv(RESULTS / "task_choice" / "runs.csv")
    five = runs[runs.part == "a"]

    fig, axes = fs.figure(2, 2, width=7.4, height=6.8)
    (a, b), (c, d) = axes

    # (a) engagement and (b) total reward by target, five people
    order = [(0.6, 1.2), (1.0, 1.0), (0.8, 0.8), (0.4, 0.4), (1.2, 0.4)]
    shade = dict(zip(order, [fs.DARK, fs.MID, fs.MID, fs.LIGHT, fs.LIGHT]))
    mark = dict(zip(order, ["o", "s", "o", "s", "o"]))
    xs = [t * 100 for t in TARGETS]
    for axis, col in ((a, "p_go"), (b, "reward")):
        for f, k in order:
            g = five[(five.f == f) & (five.k == k)].groupby("target")[col]
            m, s = g.mean(), g.agg(se)
            axis.errorbar(xs, [m[t] for t in TARGETS], yerr=[2 * s[t] for t in TARGETS],
                          color=shade[(f, k)], marker=mark[(f, k)], ms=6, mec="black",
                          mew=0.6, capsize=0, label=f"{f}, {k}")
    a.set_ylabel("P(start next set)")
    a.set_ylim(0, 1)
    b.set_ylabel("Total reward")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, title="$f$, $k$", loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, fontsize=10, title_fontsize=11, handlelength=1.0, columnspacing=1.0,
               handletextpad=0.3)

    # (c, d) feedback shown - hidden by target, the same five people, paired on seed
    hidden = runs[runs.part == "a_hidden"]
    for axis, col, label in ((c, "p_go", "P(start), shown $-$ hidden"),
                             (d, "reward", "Reward, shown $-$ hidden")):
        axis.axhline(0, color="black", lw=1.2, ls="--")
        for j, (f, k) in enumerate(order):
            on = five[(five.f == f) & (five.k == k)].set_index(["target", "seed"])[col]
            off = hidden[(hidden.f == f) & (hidden.k == k)].set_index(["target", "seed"])[col]
            diff = (on - off).groupby(level="target")
            m, s_ = diff.mean(), diff.agg(se)
            dx = (j - 2) * 1.2  # dodge, in target-percent units
            axis.errorbar([x + dx for x in xs], [m[t] for t in TARGETS],
                          yerr=[2 * s_[t] for t in TARGETS], color=shade[(f, k)],
                          marker=mark[(f, k)], ms=5, mec="black", mew=0.6, capsize=0, lw=1.4,
                          elinewidth=1.0)
        axis.set_ylabel(label)
    for axis in (a, b, c, d):
        axis.invert_xaxis()
        axis.set_xticks(xs)
        axis.set_xlabel("Target chance right (%)")

    for axis, letter in zip((a, b, c, d), "abcd"):
        fs.panel_label(axis, letter)
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.86)
    fs.save(fig, OUT / "fig7_task")


def _paired(df, target, col):
    w = df[df.target == target].pivot(index="seed", columns="feedback", values=col)
    d = w[True] - w[False]
    return d.mean(), se(d)


def figS4():
    fs.apply()
    df = pd.read_csv(RESULTS / "task_choice" / "runs.csv")
    fig, axes = fs.figure(2, 2, width=7.4, height=6.8)
    (ax, bx), (cx, dx) = axes
    xs = [t * 100 for t in TARGETS]
    a = df[df.part == "a"]
    b = df[df.part == "b"]
    # Five people, shaded by k (dark = effort-averse, light = not), shape within a shade.
    order = [(0.6, 1.2), (1.0, 1.0), (0.8, 0.8), (0.4, 0.4), (1.2, 0.4)]
    shades = dict(zip(order, [fs.DARK, fs.MID, fs.MID, fs.LIGHT, fs.LIGHT]))
    marks = dict(zip(order, ["o", "s", "o", "s", "o"]))

    # (a) engagement and (b) reward, by target, each person; open marker = their best
    for axis, col in ((ax, "p_go"), (bx, "reward")):
        for f, k in order:
            g = a[(a.f == f) & (a.k == k)].groupby("target")[col]
            m, s = g.mean(), g.agg(se)
            axis.errorbar(xs, [m[t] for t in TARGETS], yerr=[2 * s[t] for t in TARGETS],
                          color=shades[(f, k)], marker=marks[(f, k)], ms=6, mec="black",
                          mew=0.6, capsize=0, label=f"{f}, {k}")
            best = m.idxmax()
            axis.plot(best * 100, m[best], marker=marks[(f, k)], ms=11, mfc="none",
                      mec="black", mew=1.4)
        axis.invert_xaxis()
        axis.set_xticks(xs)
        axis.set_xlabel("Target chance right (%)")
    ax.set_ylabel("P(start next set)")
    ax.set_ylim(0, 1)
    bx.set_ylabel("Total reward")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="$f$, $k$", loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, fontsize=10, handlelength=1.0, columnspacing=1.0, handletextpad=0.3,
               title_fontsize=11)

    # (c, d) feedback shown - hidden, f = k = 0.8, paired seeds
    for axis, col, label in ((cx, "p_go", "P(start), shown $-$ hidden"),
                             (dx, "reward", "Reward, shown $-$ hidden")):
        pts = [_paired(b, t, col) for t in TARGETS]
        axis.axhline(0, color="black", lw=1.2, ls="--")
        axis.errorbar(xs, [m for m, _ in pts], yerr=[2 * e for _, e in pts], color=fs.MID,
                      marker="o", ms=6, mec="black", mew=0.6, capsize=0)
        axis.invert_xaxis()
        axis.set_xticks(xs)
        axis.set_xlabel("Target chance right (%)")
        axis.set_ylabel(label)
    for axis, letter in zip((ax, bx, cx, dx), "abcd"):
        fs.panel_label(axis, letter)
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.86)
    fs.save(fig, OUT / "figS4_task_reward")
