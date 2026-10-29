"""The controller figures. Paper Figure 8 is `fig8lhc` (four controllers on the hypercube
people, all five parameters fitted); `fig8` is the earlier grid version, where every
person has γ 0.9 and bound 1.5 and only f and k are fitted: how well each way of choosing
settings measures k, person by person, and why (the spread of cost each collects). Figure S5 has the RMSEs over time, f, and what each
costs in engagement. Both from
results/controllers (`python -m experiments.controllers`). The 32
people of the 4 x 4 grid x 2 seeds, question-type task, skips on, 300 ticks (start
decisions in a session, or chances to come back while away), true f and k hidden; V4,
gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s. Error
bars in S5 are 2 SE over the 32 people (bootstrap for RMSE).

`fig8` (a): each person's |k error| at 300 ticks under each arm; hollow points gave fewer
than 100 decisions to the fit; the bar is the median. (b): information minus each other
arm in k RMSE at 100 and 300 ticks, 95% percentile intervals from a bootstrap over people
that keeps each person's arms together (10,000 resamples, seed 0, as
experiments.intervals). (c): spread of cost, mean ± 2 SE.

    python -m paper fig8 figS5
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import style as fs
from .common import conditions, out, results, se



# The ticks each run was fitted at, as the experiment recorded them.
def checkpoints(name="controllers"):
    return tuple(conditions(name)["conditions"]["checkpoints"])
ORDER = ("fixed", "random", "reward_scheduled", "reward", "information")
LABELS = {"fixed": "fixed", "random": "random", "reward_scheduled": "reward",
          "reward": "reward + redo", "information": "information"}
# Grey ink for the comparison arms; blues for the planners, dark to light: reward + redo,
# reward (84 settings, diamonds), information (the app's controller).
STYLE = {"fixed": ("#8C8C8C", "--", "s"), "random": ("#8C8C8C", ":", "^"),
         "reward_scheduled": (fs.MID, "--", "D"), "reward": (fs.DARK, "-", "o"),
         "information": (fs.LIGHT, "-", "o")}


def fig8():
    fs.apply()
    df = pd.read_csv(results("controllers") / "runs.csv")
    key = ["f_true", "k_true", "seed"]
    wide = {arm: g.sort_values(key).reset_index(drop=True) for arm, g in df.groupby("arm")}
    fig, (ax, bx, cx) = fs.figure(1, 3, width=10.2, height=3.8,
                                  gridspec_kw={"width_ratios": [1.3, 1.1, 1.0]})
    xs = np.arange(len(ORDER))
    rng = np.random.default_rng(1)  # jitter only
    first, last = checkpoints()[0], checkpoints()[-1]

    # (a) each person's |k error| at the last checkpoint, by arm
    for j, arm in enumerate(ORDER):
        d = wide[arm]
        err = (d[f"k_{last}"] - d.k_true).abs().to_numpy()
        few = (d[f"rows_{last}"] < 100).to_numpy()
        jit = rng.uniform(-0.18, 0.18, len(err))
        col = STYLE[arm][0]
        ax.scatter(j + jit[~few], err[~few], s=14, color=col, edgecolors="black",
                   linewidths=0.4, zorder=3)
        ax.scatter(j + jit[few], err[few], s=14, facecolors="white", edgecolors=col,
                   linewidths=0.9, zorder=3)
        ax.plot([j - 0.3, j + 0.3], [np.median(err)] * 2, color="black", lw=2.2, zorder=4)
    ax.set_xticks(xs, [LABELS[a] for a in ORDER], fontsize=9, rotation=35,
                  ha="right", rotation_mode="anchor")
    ax.set_ylabel(f"|$k$ error|, {last} ticks")
    ax.set_ylim(bottom=0)

    # (b) information minus each other arm, k RMSE, bootstrap over people
    idx = np.random.default_rng(0).integers(0, len(wide["information"]), size=(10_000, len(wide["information"])))

    def rmse(arm, n, rows=None):
        e = ((wide[arm][f"k_{n}"] - wide[arm].k_true) ** 2).to_numpy()
        return np.sqrt(e.mean()) if rows is None else np.sqrt(e[rows].mean(axis=1))

    others = [a for a in ORDER if a != "information"]
    bx.axvline(0, color="black", lw=1.2, ls="--")
    for i, n in enumerate((first, last)):
        for j, arm in enumerate(others):
            dlt = rmse("information", n) - rmse(arm, n)
            lo, hi = np.percentile(rmse("information", n, idx) - rmse(arm, n, idx), [2.5, 97.5])
            y = j + (i - 0.5) * 0.32
            bx.errorbar(dlt, y, xerr=[[dlt - lo], [hi - dlt]], color=STYLE[arm][0],
                        marker="o" if n == first else "s", ms=6, mec="black", mew=0.6,
                        mfc=STYLE[arm][0] if n == last else "white", capsize=0, elinewidth=1.4,
                        label=f"{n} ticks" if j == 0 else None)
    bx.set_yticks(range(len(others)), [LABELS[a] for a in others], fontsize=9)
    bx.set_ylim(len(others) - 0.4, -0.6)
    bx.set_xlabel("$k$ RMSE, information $-$ other")
    bx.legend(loc="upper right", fontsize=9, handletextpad=0.2, borderaxespad=0.2)

    # (c) spread of cost in the data each arm collected, at the last checkpoint
    for j, arm in enumerate(ORDER):
        v = wide[arm][f"csd_{last}"]
        cx.errorbar(j, v.mean(), yerr=2 * se(v), color=STYLE[arm][0], marker=STYLE[arm][2],
                    ms=7, mec="black", mew=0.6, capsize=0, elinewidth=1.4)
    cx.set_xticks(xs, [LABELS[a] for a in ORDER], fontsize=9, rotation=35,
                  ha="right", rotation_mode="anchor")
    cx.set_ylabel(f"Spread of cost (sd), {last} ticks")
    cx.set_ylim(bottom=0)
    for axis, letter in zip((ax, bx, cx), "abc"):
        fs.panel_label(axis, letter)
    fig.subplots_adjust(wspace=0.55)
    fs.save(fig, out("fig8_controllers"))


def figS5():
    fs.apply()
    df = pd.read_csv(results("controllers") / "runs.csv")
    fig, axes = fs.figure(2, 2, width=7.4, height=6.4)
    (ax, bx), (cx, dx) = axes
    _draw(df, ax=ax, bx=bx, cx=cx, dx=dx)
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.9)
    fs.save(fig, out("figS5_controllers"))


def _draw(df, ax, cx, bx=None, dx=None):
    """(a) k RMSE, (b) f RMSE, (c) spread of cost, (d) engagement; panels left as None are
    skipped and the rest lettered in order."""
    fig = ax.figure
    style = STYLE
    CHECKPOINTS = checkpoints()
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
        dx.set_xticklabels([f"{lo + 1}-{n}" for lo, n in zip((0, *CHECKPOINTS), CHECKPOINTS)])
        dx.set_xlabel("Ticks (block)")
        dx.set_ylabel("P(start next set)")
        dx.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               fontsize=10, handlelength=2.8, columnspacing=1.0)
    for axis, letter in zip([a for a in (ax, bx, cx, dx) if a is not None], "abcd"):
        fs.panel_label(axis, letter)


def fig9():
    """Figure 9: the app's controller against one that fits and plans with γ, the bound and
    T0 as well, from results/controller_full (`python -m experiments.controller_full`).
    People from a Latin hypercube over f, k, γ, BOUND and T0, 300 ticks each under both,
    paired on seed. (a)-(c) RMSE of f, k and γ, all five parameters fitted by maximum
    likelihood to the data recorded by each checkpoint; error bars 2 SE from a bootstrap
    over people (2000 resamples, seed 0). (d) the start decisions each person gave in the
    300 ticks under each controller, and the mean paired difference with its 95% t interval."""
    import glob

    fs.apply()
    CHECKPOINTS = checkpoints("controller_full")
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(results("controller_full") / "runs_*.csv"))],
                   ignore_index=True)
    both = set(df[df.arm == "current"].person) & set(df[df.arm == "full"].person)
    df = df[df.person.isin(both)].sort_values(["arm", "person"])
    arms = {"current": ("app's controller ($\\gamma$ 0.9, bound 1.5)", fs.GREY, "--", "s"),
            "full": ("all five parameters", fs.MID, "-", "o")}
    fig, axes = fs.figure(2, 2, width=7.4, height=6.4)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(both), size=(2000, len(both)))
    for ax, (p, label) in zip(axes.ravel()[:3], (("f", "$f$"), ("k", "$k$"), ("g", r"$\gamma$"))):
        for j, (arm, (name, col, ls, mk)) in enumerate(arms.items()):
            d = df[df.arm == arm]
            ys, es = [], []
            for n in CHECKPOINTS:
                e2 = ((d[f"{p}_{n}"] - d[f"true_{p}"]) ** 2).to_numpy()
                ys.append(np.sqrt(np.nanmean(e2)))
                es.append(float(np.std(np.sqrt(np.nanmean(e2[idx], axis=1)))))
            ax.errorbar([c + (j - 0.5) * 8 for c in CHECKPOINTS], ys, yerr=[2 * e for e in es],
                        color=col, ls=ls, marker=mk, ms=6, mec="black", mew=0.6, capsize=0,
                        elinewidth=1.2, label=name)
        ax.set_xticks(CHECKPOINTS)
        ax.set_xlabel("Ticks")
        ax.set_ylabel(f"{label} RMSE")
        ax.set_ylim(bottom=0)
    # (d) start decisions each person gave in 300 ticks, under each controller
    ax = axes[1, 1]
    cur, full = (df[df.arm == arm].set_index("person").sort_index() for arm in ("current", "full"))
    ax.scatter(cur.decisions, full.decisions, s=14, color=fs.MID, edgecolors="black",
               linewidths=0.4, zorder=3)
    top = max(cur.decisions.max(), full.decisions.max()) * 1.05
    ax.plot([0, top], [0, top], color=fs.GREY, ls="--", lw=1.2, zorder=1)
    d = full.decisions - cur.decisions
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    ax.text(0.04, 0.96, f"all five $-$ app's:\n{d.mean():+.1f} [{d.mean() - half:+.1f}, "
            f"{d.mean() + half:+.1f}]", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("Start decisions, app's controller")
    ax.set_ylabel("Start decisions, all five")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               fontsize=10, handlelength=2.4)
    for ax, letter in zip(axes.ravel(), "abcd"):
        fs.panel_label(ax, letter)
    fig.subplots_adjust(hspace=0.45, wspace=0.4, top=0.9)
    fs.save(fig, out("fig9_controller_full"))


def fig8lhc():
    """Paper Figure 8: four ways of choosing settings on the hypercube people, from
    results/controller_full (`python -m experiments.controller_full`, its four-controller
    arms). 128 people from a Latin hypercube over f, k, γ, BOUND and T0, 300 ticks each
    under every arm, paired on seed: the fixed default, a random setting at each return,
    the engagement maximizer and information planning on f, k and γ, the last three over
    the same 384 settings, the two planners fitting all five parameters. All five fitted
    by maximum likelihood to the data recorded by each checkpoint. (a)-(b) RMSE of f and
    k; error bars 2 SE from a bootstrap over people (2000 resamples, seed 0). (c) γ RMSE at
    300 ticks by the start decisions a person gave in them (groups of fewer than 5 people
    left out). (d) each person's |k error| at 300 ticks; hollow points gave fewer than 100
    decisions to the fit; the bar is the median. (e)-(f) information minus each other arm,
    P(start next set) and the share of the 300 ticks spent in the task, mean and 95% t
    interval over people."""
    import glob

    fs.apply()
    marks = checkpoints("controller_full")
    last = marks[-1]
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(results("controller_full") / "runs_*.csv"))],
                   ignore_index=True)
    order = ("fixed", "random_return", "engage", "info_fkg")
    labels = {"fixed": "fixed default", "random_return": "random", "engage": "engagement",
              "info_fkg": "information"}
    style = {"fixed": (fs.GREY, "--", "s"), "random_return": (fs.GREY, ":", "^"),
             "engage": (fs.DARK, "-", "D"), "info_fkg": (fs.LIGHT, "-", "o")}
    keep = set.intersection(*(set(df[df.arm == a].person) for a in order))
    by = {a: df[(df.arm == a) & df.person.isin(keep)].drop_duplicates("person")
          .set_index("person").sort_index() for a in order}
    for d in by.values():
        d["in_task"] = 1 - d.away
    fig, axes = fs.figure(2, 3, width=10.2, height=6.6)
    idx = np.random.default_rng(0).integers(0, len(keep), size=(2000, len(keep)))

    # (a)-(b) RMSE over time
    for ax, (p, name) in zip(axes[0][:2], (("f", "$f$"), ("k", "$k$"))):
        for j, arm in enumerate(order):
            col, ls, mk = style[arm]
            ys, es = [], []
            for n in marks:
                e2 = ((by[arm][f"{p}_{n}"] - by[arm][f"true_{p}"]) ** 2).to_numpy()
                ys.append(np.sqrt(np.nanmean(e2)))
                es.append(float(np.std(np.sqrt(np.nanmean(e2[idx], axis=1)))))
            ax.errorbar([c + (j - 1.5) * 7 for c in marks], ys, yerr=[2 * e for e in es],
                        color=col, ls=ls, marker=mk, ms=6, mec="black", mew=0.6, capsize=0,
                        elinewidth=1.2, label=labels[arm])
        ax.set_xticks(marks)
        ax.set_xlabel("Ticks")
        ax.set_ylabel(f"{name} RMSE")
        ax.set_ylim(bottom=0)

    # (c) γ RMSE at the last checkpoint, by how many start decisions the person gave
    ax = axes[0, 2]
    bins = (0, 50, 100, 200, 10_000)
    names = ["<50", "50-99", "100-199", "200+"]
    for j, arm in enumerate(order):
        col, ls, mk = style[arm]
        d = by[arm]
        e2 = (d[f"g_{last}"] - d.true_g) ** 2
        cut = pd.cut(d.decisions, bins, right=False, labels=False)
        xs, ys = [], []
        for b in range(len(names)):
            sel = e2[(cut == b).to_numpy()].dropna()
            if len(sel) >= 5:
                xs.append(b + (j - 1.5) * 0.12)
                ys.append(np.sqrt(sel.mean()))
        ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=6, mec="black", mew=0.6)
    ax.set_xticks(range(len(names)), names, fontsize=10)
    ax.set_xlabel("Start decisions in 300 ticks")
    ax.set_ylabel(r"$\gamma$ RMSE")
    ax.set_ylim(bottom=0)

    # (d) each person's |k error| at the last checkpoint
    ax = axes[1, 0]
    rng = np.random.default_rng(1)  # jitter only
    for j, arm in enumerate(order):
        d = by[arm]
        err = (d[f"k_{last}"] - d.true_k).abs().to_numpy()
        few = (d[f"rows_{last}"] < 100).to_numpy()
        jit = rng.uniform(-0.2, 0.2, len(err))
        col = style[arm][0]
        ax.scatter(j + jit[~few], err[~few], s=12, color=col, edgecolors="black",
                   linewidths=0.4, zorder=3)
        ax.scatter(j + jit[few], err[few], s=12, facecolors="white", edgecolors=col,
                   linewidths=0.9, zorder=3)
        ax.plot([j - 0.32, j + 0.32], [np.nanmedian(err)] * 2, color="black", lw=2.2, zorder=4)
    ax.set_xticks(range(len(order)), [labels[a] for a in order], fontsize=10, rotation=30,
                  ha="right", rotation_mode="anchor")
    ax.set_ylabel(f"|$k$ error|, {last} ticks")
    ax.set_ylim(bottom=0)

    # (e)-(f) information minus each baseline, paired on person
    base = order[:-1]
    for ax, (col, name) in zip(axes[1, 1:], (("p_go", "P(start)"), ("in_task", "Time in task"))):
        ax.axvline(0, color="black", lw=1.2, ls="--")
        for j, arm in enumerate(base):
            d = (by["info_fkg"][col] - by[arm][col]).dropna()
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            c, _, mk = style[arm]
            ax.errorbar(d.mean(), j, xerr=half, color=c, marker=mk, ms=7, mec="black",
                        mew=0.6, capsize=0, elinewidth=1.6)
        ax.set_ylim(len(base) - 0.5, -0.5)
        ax.set_xlabel(f"{name}, difference")
        ax.set_yticks(range(len(base)), [labels[a] for a in base] if col == "p_go" else [])
    handles, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               fontsize=10, handlelength=2.8, columnspacing=1.2)
    for ax, letter in zip(axes.ravel(), "abcdef"):
        fs.panel_label(ax, letter)
    fig.subplots_adjust(hspace=0.55, wspace=0.45, top=0.9)
    fs.save(fig, out("fig8lhc_controllers"))


def figS10():
    """Supplementary Figure S10: feedback in blocks as a controller setting, with redos
    mid-set or on the end-of-set screen, from results/feedback_blocks (`python -m
    experiments.feedback_blocks`). The app's controller, the same with feedback shown and
    hidden in alternate blocks of 25 start decisions, and the same with the hidden blocks'
    redos moved to the end of the set; paired on person and seed. (a)-(b) RMSE of k and γ,
    all five parameters fitted by maximum likelihood at each checkpoint; error bars 2 SE from
    a bootstrap over people (2000 resamples, seed 0). (c) redo decisions per person, mid-set
    and at the end of the set. (d) each blocked arm minus the app's controller: P(start next
    set), share of ticks away and start decisions, mean and 95% t interval."""
    import glob

    fs.apply()
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(results("feedback_blocks") / "runs_*.csv"))],
                   ignore_index=True)
    marks = checkpoints("feedback_blocks")
    arms = {"current": ("app's controller", fs.GREY, "--", "s"),
            "blocks": ("blocks, redos mid-set", fs.DARK, "-", "o"),
            "blocks_end": ("blocks, redos at the end", fs.LIGHT, "-", "D")}
    keep = set.intersection(*(set(df[df.arm == a].person) for a in arms))
    df = df[df.person.isin(keep)].sort_values(["arm", "person"])
    fig, axes = fs.figure(2, 2, width=7.4, height=6.4)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keep), size=(2000, len(keep)))
    for ax, (p, label) in zip(axes[0], (("k", "$k$"), ("g", r"$\gamma$"))):
        for j, (arm, (name, col, ls, mk)) in enumerate(arms.items()):
            d = df[df.arm == arm]
            ys, es = [], []
            for n in marks:
                e2 = ((d[f"{p}_{n}"] - d[f"true_{p}"]) ** 2).to_numpy()
                ys.append(np.sqrt(np.nanmean(e2)))
                es.append(float(np.std(np.sqrt(np.nanmean(e2[idx], axis=1)))))
            ax.errorbar([c + (j - 1) * 8 for c in marks], ys, yerr=[2 * e for e in es], color=col,
                        ls=ls, marker=mk, ms=6, mec="black", mew=0.6, capsize=0, elinewidth=1.2,
                        label=name)
        ax.set_xticks(marks)
        ax.set_xlabel("Ticks")
        ax.set_ylabel(f"{label} RMSE")
        ax.set_ylim(bottom=0)
    ax = axes[1, 0]
    for j, (arm, (name, col, _, _)) in enumerate(arms.items()):
        d = df[df.arm == arm]
        mid, end = (d.redos - d.end_redos).mean(), d.end_redos.mean()
        ax.bar(j, mid, 0.6, color=col, edgecolor="black", lw=1.0)
        ax.bar(j, end, 0.6, bottom=mid, color="white", edgecolor=col, hatch="///", lw=1.0)
    ax.set_xticks(range(len(arms)), ["app's", "blocks,\nmid-set", "blocks,\nat the end"], fontsize=10)
    ax.set_ylabel("Redo decisions per person")
    ax.set_ylim(0, 1.3 * df.redos.groupby(df.arm).mean().max())
    ax.text(0.02, 0.98, "hatched: at the end of the set", transform=ax.transAxes, va="top",
            fontsize=9)
    ax = axes[1, 1]
    cur = df[df.arm == "current"].set_index("person").sort_index()
    rows = (("p_go", "P(start)", 1.0), ("away", "time away", 1.0), ("decisions", "starts ÷ 100", 0.01))
    ax.axvline(0, color="black", lw=1.2, ls="--")
    for j, arm in enumerate(("blocks", "blocks_end")):
        d = df[df.arm == arm].set_index("person").sort_index()
        for i, (col, _, scale) in enumerate(rows):
            x = (d[col] - cur[col]).dropna() * scale
            half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            name, color, _, mk = arms[arm]
            ax.errorbar(x.mean(), i + (j - 0.5) * 0.3, xerr=half, color=color, marker=mk, ms=6,
                        mec="black", mew=0.6, capsize=0, elinewidth=1.4)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=10)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel("blocked arm $-$ app's controller")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=10, handlelength=2.2)
    for ax, letter in zip(axes.ravel(), "abcd"):
        fs.panel_label(ax, letter)
    fig.subplots_adjust(hspace=0.45, wspace=0.45, top=0.9)
    fs.save(fig, out("figS10_feedback_blocks"))
