"""Paper Figures 6 (each shortcut in the fit on its own) and S3 (every combination of three
fit choices). Current model: gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0 s.

(a) timed-out trials kept vs dropped: results/recovery_sweep (experiments.recovery_sweep)
(b) non-decision time pinned at 0 vs fitted: results/pipelines/t0.csv
    (python -m temporal recovery --mode t0 --reps 3 --out results/pipelines)
(c) f and k from decisions, answer and skip times, both: results/pipelines/persistence.csv
    (python -m temporal recovery --mode persistence --reps 3 --out results/pipelines)
(d) expected SD of k from the Fisher information at the truth, 300 start decisions with
    and without their redo decisions, with f and k free (γ, BOUND, T0 at the truth) and
    with all five free: results/precision_full (experiments.precision_full, the 100
    people of experiments.recovery_lhc under InfoController); bars are medians, dots people
S3: results/fit_factorial (experiments.fit_factorial)

    python -m paper fig6 figS3
"""
import numpy as np
import pandas as pd

from . import style as fs
from .common import out, results

# The 8 fits, pipeline first, then one change, two, all three.
VERSIONS = [(("kept", "fitted", "no"), "pipeline"), (("dropped", "fitted", "no"), "drop\ntimeouts"),
            (("kept", "pinned", "no"), "pin T0"), (("kept", "fitted", "yes"), "add\ntimes"),
            (("dropped", "pinned", "no"), "drop +\npin"), (("dropped", "fitted", "yes"), "drop +\ntimes"),
            (("kept", "pinned", "yes"), "pin +\ntimes"), (("dropped", "pinned", "yes"), "all\nthree")]


def _bias_panel(ax, groups, labels, colors, ylabel):
    # groups: {label: {param: array of per-fit values}}; bars = mean, dots = each fit
    width = 0.36
    for j, (lab, col) in enumerate(zip(labels, colors)):
        for i, p in enumerate(("f", "k")):
            vals = groups[lab][p]
            x = i + (j - (len(labels) - 1) / 2) * width
            ax.bar(x, np.mean(vals), width * 0.9, color=col, edgecolor="black", lw=1.0,
                   label=lab if i == 0 else None, zorder=2)
            jitter = np.random.default_rng(i * 10 + j).uniform(-width * 0.25, width * 0.25, len(vals))
            ax.scatter(x + jitter, vals, s=8, color="black", alpha=0.45, lw=0, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$f$", "$k$"])
    ax.set_ylabel(ylabel)


def fig6():
    fs.apply()
    fig, axes = fs.figure(2, 2, width=7.4, height=6.6)
    fig.subplots_adjust(hspace=0.5, wspace=0.45)
    (a, b), (c, d) = axes

    # (a) timed-out trials
    r = pd.read_csv(results("recovery_sweep") / "runs.csv")
    groups = {lab: {p: 100 * (r[f"{p}_{s}"] - r[f"{p}_true"]) / r[f"{p}_true"] for p in ("f", "k")}
              for lab, s in (("kept", "kept"), ("dropped", "drop"))}
    _bias_panel(a, groups, ["kept", "dropped"], [fs.DARK, fs.LIGHT], "Bias (%)")
    a.axhline(0, color="black", lw=1.2, ls="--", zorder=1)
    a.set_ylim(-90, 160)
    a.legend(title="Timed out", loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.8)
    fs.panel_label(a, "a")

    # (b) non-decision time
    t = pd.read_csv(results("pipelines") / "t0.csv")
    g = t.groupby("t0_true")
    for col, mark, lab in (("pinned", "o", "pinned at 0"), ("free", "s", "fitted")):
        shade = fs.LIGHT if col == "pinned" else fs.DARK
        for p, dx, mfc in (("f", -0.006, shade), ("k", 0.006, "white")):
            m, s = g[f"{p}_{col}"].mean(), g[f"{p}_{col}"].std()
            b.errorbar(m.index + dx, m, yerr=s, color=shade, marker=mark, ms=6, mfc=mfc,
                       mec="black", mew=0.8, capsize=0, lw=2,
                       label=f"{lab}, ${p}$")
    b.axhline(0.8, color="black", lw=1.2, ls="--")
    b.set_xlabel("True non-decision time (s)")
    b.set_ylabel("Recovered value")
    b.set_xticks(sorted(t.t0_true.unique()))
    b.legend(loc="lower left", fontsize=9, handlelength=1.2, ncol=2, columnspacing=0.6)
    fs.panel_label(b, "b")

    # (c) data source
    pr = pd.read_csv(results("pipelines") / "persistence.csv")
    sources = (("decisions", "decisions"), ("persistence", "times"), ("joint", "both"))
    xs = np.arange(len(sources))
    width = 0.36
    for i, (p, shade) in enumerate((("f", fs.DARK), ("k", fs.LIGHT))):
        rm = [np.sqrt(((pr[f"{p}_{s}"] - pr[f"{p}_true"]) ** 2).mean()) for s, _ in sources]
        x = xs + (i - 0.5) * width
        c.bar(x, rm, width * 0.9, color=shade, edgecolor="black", lw=1.0, label=f"${p}$", zorder=2)
        for xi, (s, _) in zip(x, sources):
            err = np.abs(pr[f"{p}_{s}"] - pr[f"{p}_true"])
            jit = np.random.default_rng(abs(int(xi * 100))).uniform(-width * 0.25, width * 0.25, len(err))
            c.scatter(xi + jit, err, s=8, color="black", alpha=0.45, lw=0, zorder=3)
    c.set_xticks(xs)
    c.set_xticklabels([lab for _, lab in sources])
    c.set_ylabel("RMSE")
    c.set_xlabel("Fitted from")
    c.legend(loc="upper right", handlelength=1.0)
    fs.panel_label(c, "c")

    # (d) how precisely k can be measured, by the kind of decisions in the data
    _precision(d)
    fs.panel_label(d, "d")

    fs.save(fig, out("fig6_shortcuts"))


# Expected SD of k by the decisions in the data and how many parameters are free
# (experiments.precision_full).
DESIGNS = (("starts", "starts"), ("both", "starts\n+ redos"))


def _precision(d):
    df = pd.read_csv(results("precision_full") / "runs.csv")
    xs = np.arange(len(DESIGNS))
    width = 0.38
    for i, (free, shade, label) in enumerate((("2", fs.DARK, "$f$, $k$ free"),
                                              ("5", fs.LIGHT, "all five free"))):
        x = xs + (i - 0.5) * width
        for xi, (name, _) in zip(x, DESIGNS):
            vals = df[f"sd_k_{name}_{free}"].dropna()
            d.bar(xi, np.median(vals), width * 0.9, color=shade, edgecolor="black", lw=1.0,
                  label=label if xi == x[0] else None, zorder=2)
            jit = np.random.default_rng(int(abs(xi) * 100) + i).uniform(-width * 0.25, width * 0.25, len(vals))
            d.scatter(xi + jit, vals, s=6, color="black", alpha=0.45, lw=0, zorder=3)
    d.set_xticks(xs)
    d.set_xticklabels([lab for _, lab in DESIGNS], fontsize=10)
    d.set_xlabel("Decisions in the data")
    d.set_ylabel("Expected SD of $k$")
    d.set_yscale("log")
    d.set_yticks([0.03, 0.1, 0.3, 1.0], ["0.03", "0.1", "0.3", "1"])
    d.minorticks_off()
    d.legend(loc="upper right", handlelength=1.0, fontsize=9)


def figS3():
    fs.apply()
    df = pd.read_csv(results("fit_factorial") / "runs.csv")
    fig = fs.plt.figure(figsize=(7.4, 6.8))
    gs = fig.add_gridspec(2, 1, hspace=0.6)
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1])
    xs = np.arange(len(VERSIONS))
    width = 0.38
    for i, (p, shade) in enumerate((("f", fs.DARK), ("k", fs.LIGHT))):
        rm, bias = [], []
        for key, _ in VERSIONS:
            g = df[(df.timeouts == key[0]) & (df.t0 == key[1]) & (df.times == key[2])]
            rm.append(np.sqrt(((g[f"{p}_hat"] - g[f"{p}_true"]) ** 2).mean()))
            bias.append(100 * ((g[f"{p}_hat"] - g[f"{p}_true"]) / g[f"{p}_true"]).mean())
        x = xs + (i - 0.5) * width
        a.bar(x, rm, width * 0.9, color=shade, edgecolor="black", lw=1.0, label=f"${p}$", zorder=2)
        b.bar(x, bias, width * 0.9, color=shade, edgecolor="black", lw=1.0, zorder=2)
    a.axvspan(-0.5, 0.5, color=fs.BOX_FILL, zorder=0)
    a.set_xticks(xs)
    a.set_xticklabels([lab for _, lab in VERSIONS], fontsize=10)
    a.set_xlim(-0.5, len(VERSIONS) - 0.5)
    a.set_ylabel("RMSE")
    a.legend(loc="upper right", handlelength=1.0)
    b.axhline(0, color="black", lw=1.2)
    b.set_xticks(xs)
    b.set_xticklabels([str(i + 1) for i in xs], fontsize=10)
    b.set_xlabel("Fit version (as in a)")
    b.set_ylabel("Mean bias (%)")
    fs.panel_label(a, "a", x=-0.08)
    fs.panel_label(b, "b", x=-0.08)
    fs.save(fig, out("figS3_factorial"))
