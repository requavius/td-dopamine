"""Paper Figures 7 (which difficulty keeps a person engaged, and what showing feedback does,
five people) and S4 (the same for f = k = 0.8 with 96 paired seeds), from
results/task_choice (`python -m experiments.task_choice`). Question-type task, one setting
held for each run of 300 ticks, review 0.25, skipped questions dropped, no controller; V4,
gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s. Error bars in
Figure 7 (a, b) and S4 are 2 SE over seeds. Figure 7 (c) is the feedback effect (shown -
hidden P(start)) by the person's γ and the target, from results/gamma_feedback: each cell
the mean over 32 paired seeds of the five people's mean, hatched where it is within 2 SE
(over seeds) of zero, an open circle on a row's largest cell if it clears 2 SE, and the
γ 0.9 row outlined (the condition (a) and (b) show, in separate runs).

    python -m paper fig7 figS4
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from . import style as fs
from .common import conditions, out, results, se

TARGETS = (0.9, 0.75, 0.6, 0.45, 0.3, 0.2)


# An experiment's people, most effort-averse first, shaded by k (dark = high), each with
# its own marker (the diamond is kept for averages); for the paper's five, the order and
# styles it shows.
def people(name):
    order = sorted((tuple(p) for p in conditions(name)["conditions"]["people"]),
                   key=lambda p: (-p[1], p[0]))
    shade = {p: fs.DARK if p[1] >= 1.1 else fs.MID if p[1] > 0.6 else fs.LIGHT for p in order}
    mark = {p: "os^vp<>h"[i] for i, p in enumerate(order)}
    return order, shade, mark


def fig7():
    fs.apply()
    runs = pd.read_csv(results("task_choice") / "runs.csv")
    five = runs[runs.part == "a"]

    fig, (a, b, d) = fs.figure(1, 3, width=11.0, height=4.2)

    # (a) engagement and (b) total reward by target, five people
    order, shade, mark = people("task_choice")
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

    # (c) feedback shown - hidden in P(start) by γ and target (experiments.gamma_feedback,
    # 32 paired seeds):
    # each seed's mean over the five people, then mean and SE over seeds; columns share
    # (b)'s x positions, rows are γ as categories, most patient at the top. A row's largest
    # cell gets a circle only if it clears 2 SE; where the peak sits is descriptive
    from experiments.gamma_feedback import paired
    g = paired(pd.read_csv(results("gamma_feedback") / "runs.csv"))
    GAMMAS = sorted(g.g.unique())
    w = g.groupby(["g", "target", "seed"]).p_go.mean()
    m = w.groupby(["g", "target"]).mean().unstack("target")
    s = w.groupby(["g", "target"]).agg(se).unstack("target")
    mids = [(x + y) / 2 for x, y in zip(xs, xs[1:])]
    xe = [xs[0] + (xs[0] - mids[0])] + mids + [xs[-1] - (mids[-1] - xs[-1])]
    lim = np.ceil(m.abs().values.max() * 100) / 100
    mesh = d.pcolormesh(xe, range(len(GAMMAS) + 1), m[list(TARGETS)].values, cmap="RdBu",
                        vmin=-lim, vmax=lim)
    for i, gv in enumerate(GAMMAS):
        for j, t in enumerate(TARGETS):
            v, x0, x1 = m.loc[gv, t], xe[j], xe[j + 1]
            if abs(v) <= 2 * s.loc[gv, t]:
                hatch = Rectangle((x0, i), x1 - x0, 1, fill=False, hatch="///",
                                  edgecolor="#B4B4B4", lw=0)
                hatch.set_hatch_linewidth(0.5)
                d.add_patch(hatch)
            label = "0" if round(v, 2) == 0 else f"{v:+.2f}".replace("0.", ".")
            d.text(xs[j], i + 0.6, label.replace("-", "\u2212"), ha="center", va="center",
                   fontsize=7, color="white" if abs(v) > 0.6 * lim else "black")
        best = m.loc[gv, list(TARGETS)].idxmax()
        if m.loc[gv, best] > 2 * s.loc[gv, best]:
            d.plot(xs[TARGETS.index(best)], i + 0.25, marker="o", ms=4, mfc="none",
                   mec="white" if m.loc[gv, best] > 0.6 * lim else "black", mew=1.0)
    row = GAMMAS.index(0.9)
    d.add_patch(Rectangle((xe[0], row), xe[-1] - xe[0], 1, fill=False, edgecolor="black",
                          lw=1.6, clip_on=False))
    d.text(xe[-1] - 1.5, row + 0.5, "condition\nof (a), (b)", ha="left", va="center",
           fontsize=8, linespacing=1.1)
    d.set_yticks([i + 0.5 for i in range(len(GAMMAS))], [f"{v:g}" for v in GAMMAS])
    d.set_ylabel(r"Discount factor $\gamma$")
    d.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        d.spines[side].set_visible(False)
    cax = d.inset_axes([0.1, 1.2, 0.8, 0.05])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-lim, 0, lim])
    bar.set_label("P(start), shown $-$ hidden", fontsize=10, labelpad=2)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    bar.ax.tick_params(labelsize=9)
    for axis in (a, b, d):
        axis.set_xlim(xe[0], xe[-1])
        axis.set_xticks(xs)
        axis.set_xlabel("Target chance right (%)")

    for axis, letter in zip((a, b, d), "abc"):
        fs.panel_label(axis, letter)
    fig.subplots_adjust(wspace=0.42, top=0.7, bottom=0.14, left=0.07, right=0.93)
    _centre(fig)
    # the legend over (a) and (b), level with the colorbar over (c)
    handles, labels = a.get_legend_handles_labels()
    left, right = a.get_position().x0, b.get_position().x1
    fig.legend(handles, labels, title="$f$, $k$", loc="lower center",
               bbox_to_anchor=((left + right) / 2, cax.get_position().y0 - 0.01),
               ncol=len(order), fontsize=10, title_fontsize=11, handlelength=1.0,
               columnspacing=1.0, handletextpad=0.3)
    fs.save(fig, out("fig7_task"))


# Shift every axes sideways so the drawn content has equal margins left and right.
def _centre(fig):
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    width = fig.get_figwidth()
    shift = ((width - box.x1) - box.x0) / 2 / width
    for ax in fig.axes:
        if ax.get_axes_locator() is None:
            pos = ax.get_position()
            ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])


def _paired(df, target, col):
    w = df[df.target == target].pivot(index="seed", columns="feedback", values=col)
    d = w[True] - w[False]
    return d.mean(), se(d)


def figS4():
    fs.apply()
    df = pd.read_csv(results("task_choice") / "runs.csv")
    fig, axes = fs.figure(2, 2, width=7.4, height=6.8)
    (ax, bx), (cx, dx) = axes
    xs = [t * 100 for t in TARGETS]
    a = df[df.part == "a"]
    b = df[df.part == "b"]
    # Five people, shaded by k (dark = effort-averse, light = not), shape within a shade.
    order, shades, marks = people("task_choice")

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
               ncol=len(order), fontsize=10, handlelength=1.0, columnspacing=1.0, handletextpad=0.3,
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
    fs.save(fig, out("figS4_task_reward"))




# γ as its horizon 1/(1 - γ) on a log axis, so 0.9, 0.95 and 0.99 are not crowded together.
def _horizon(g):
    return 1 / (1 - g)


def figS6():
    """Supplementary Figure S6: what showing feedback does, by γ, from
    results/gamma_feedback (`python -m experiments.gamma_feedback`). Task as Figure 7, 32
    seeds paired across γ, target and feedback; each point is shown - hidden averaged over
    the six targets, error bars 2 SE over seeds."""
    from experiments.gamma_feedback import paired
    from temporal.model.params import ModelState

    fs.apply()
    d = paired(pd.read_csv(results("gamma_feedback") / "runs.csv"))
    GAMMAS = sorted(d.g.unique())
    fig, axes = fs.figure(2, 2, width=7.4, height=6.8)
    (a, b), (c, e) = axes
    order, shade, mark = people("gamma_feedback")

    # (a) a set's achievement discounted to its first slot: one lump at the last slot
    # (hidden) against one share per slot (shown), equal totals
    n = ModelState.stage_amt  # the dataclass default
    gs = [0.5 + 0.49 * i / 400 for i in range(401)]
    worth = [n * g ** (n - 1) / sum(g ** s for s in range(n)) for g in gs]
    a.plot([_horizon(g) for g in gs], worth, color="black", lw=2)
    a.plot([_horizon(g) for g in GAMMAS],
           [n * g ** (n - 1) / sum(g ** s for s in range(n)) for g in GAMMAS],
           ls="none", marker="o", ms=6, color=fs.MID, mec="black", mew=0.6)
    a.axhline(1, color="black", lw=1.2, ls="--")
    a.set_ylim(0, 1.08)
    a.set_ylabel("Worth of a set, hidden / shown")

    # (b-d) shown - hidden by γ, the five people of Figure 7
    for axis, col, label in ((b, "p_go", "P(start), shown $-$ hidden"),
                             (c, "reward", "Reward, shown $-$ hidden"),
                             (e, "away", "Share of time away,\nshown $-$ hidden")):
        axis.axhline(0, color="black", lw=1.2, ls="--")
        for j, (f, k) in enumerate(order):
            w = d[(d.f == f) & (d.k == k)].groupby(["g", "seed"])[col].mean().unstack("g")
            dodge = 1.035 ** (j - 2)  # multiplicative, on the log axis
            axis.errorbar([_horizon(g) * dodge for g in GAMMAS], [w[g].mean() for g in GAMMAS],
                          yerr=[2 * se(w[g]) for g in GAMMAS], color=shade[(f, k)],
                          marker=mark[(f, k)], ms=6, mec="black", mew=0.6, capsize=0, lw=1.6,
                          elinewidth=1.0, label=f"{f}, {k}")
        axis.set_ylabel(label)

    for axis in (a, b, c, e):
        axis.set_xscale("log")
        axis.set_xticks([_horizon(g) for g in GAMMAS])
        axis.set_xticklabels(["" if g == 0.7 else f"{g:g}" for g in GAMMAS])  # 0.7 crowds 0.8
        axis.minorticks_off()
        axis.set_xlabel(r"Discount factor $\gamma$")
    handles, labels = b.get_legend_handles_labels()
    fig.legend(handles, labels, title="$f$, $k$", loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(order), fontsize=10, title_fontsize=11, handlelength=1.0, columnspacing=1.0,
               handletextpad=0.3)
    for axis, letter in zip((a, b, c, e), "abcd"):
        fs.panel_label(axis, letter)
    fig.subplots_adjust(hspace=0.45, wspace=0.42, top=0.86)
    fs.save(fig, out("figS6_gamma_feedback"))
