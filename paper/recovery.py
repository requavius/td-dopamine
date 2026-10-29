"""Paper Figure 4 (full model): recovery of f, k, γ, BOUND and T0 fitted together, from
results/recovery_lhc/fits.csv (`python -m experiments.recovery_lhc fit`).

People are a Latin hypercube over f, k in 0.2 to 1.5, γ in 0.6 to 0.97, BOUND in 1.0 to
2.0 and T0 in 0 to 0.4 s. Each plays the app's task under InfoController for 300 start
decisions plus their redo decisions; all five parameters are fitted by maximum
likelihood with the closed-form likelihood (temporal.fastlik). α 0.05, SIGMA 0.7,
ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0.

(a)-(e) true against recovered, with RMSE, bias and r; hollow points are fits that sat at a
parameter bound. (f) the error bars checked, one parameter at a time: the share of fits
away from a parameter bound whose truth lies within ±1.96 SE, SE from the observed
information (filled) and from the Fisher information at the truth (hollow), and the share
of all fits, those at a bound included, whose profile-likelihood 95% interval contains the
truth (diamonds; results/lhc_profile, `python -m experiments.lhc_profile`). The grey band
is where the share among the n fits away from a bound falls 95% of the time if the
intervals are honest (binomial, p = 0.95); for all 100 it is 91 to 99%.

    python -m paper fig4lhc
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import style as fs
from .common import out, results

PARAMS = [("f", "f"), ("k", "k"), ("g", r"$\gamma$"), ("bound", "bound"), ("t0", r"$T_0$ (s)")]


def fig4lhc():
    fs.apply()
    df = pd.read_csv(results("recovery_lhc") / "fits.csv")
    fig, axes = fs.figure(2, 3, width=7.6, height=5.4)
    axes = axes.ravel()
    for ax, (p, label) in zip(axes, PARAMS):
        truth, est = df[f"true_{p}"].to_numpy(), df[f"est_{p}"].to_numpy()
        err = est - truth
        lo = min(truth.min(), est.min())
        hi = max(truth.max(), est.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=fs.GREY, ls="--", lw=1.2, zorder=1)
        edge = df[[f"at_bound_{q}" for q, _ in PARAMS]].any(axis=1).to_numpy()
        ax.scatter(truth[~edge], est[~edge], s=16, color=fs.MID, edgecolors="black",
                   linewidths=0.4, zorder=3)
        ax.scatter(truth[edge], est[edge], s=18, facecolors="white", edgecolors="black",
                   linewidths=0.8, zorder=4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"recovered {label}")
        ax.text(0.04, 0.96, f"RMSE {np.sqrt(np.mean(err ** 2)):.3f}\nbias {err.mean():+.3f}\n"
                f"r {np.corrcoef(truth, est)[0, 1]:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9)
    # (f) coverage of ±1.96 SE by parameter, observed and Fisher information
    ax = axes[5]
    away = ~df[[f"at_bound_{p}" for p, _ in PARAMS]].any(axis=1)
    n = int(away.sum())
    lo, hi = (100 * stats.binom.ppf(q, n, 0.95) / n for q in (0.025, 0.975))
    xs = np.arange(len(PARAMS))
    ax.axhspan(lo, hi, color=fs.GREY, alpha=0.25, lw=0, zorder=0)
    ax.axhline(95, color=fs.GREY, ls="--", lw=1.2, zorder=1)
    for dx, src, face in ((-0.22, "obs", fs.MID), (0.0, "fisher", "white")):
        cover = [100 * (np.abs((df[f"est_{p}"] - df[f"true_{p}"]) / df[f"se_{src}_{p}"])[away]
                        < 1.96).mean() for p, _ in PARAMS]
        ax.scatter(xs + dx, cover, s=40, facecolors=face, edgecolors="black", linewidths=0.8,
                   zorder=3, label={"obs": "observed", "fisher": "Fisher"}[src])
    # profile-likelihood intervals, every fit (experiments.lhc_profile)
    prof = pd.read_csv(results("lhc_profile") / "intervals.csv").set_index("person")
    truth = df.set_index("person")
    cover = [100 * ((prof[f"lo_{p}"] <= truth[f"true_{p}"]) & (truth[f"true_{p}"] <= prof[f"hi_{p}"])).mean()
             for p, _ in PARAMS]
    ax.scatter(xs + 0.22, cover, s=44, marker="D", color=fs.DARK, edgecolors="black",
               linewidths=0.8, zorder=3, label=f"profile, all {len(prof)}")
    ax.set_xticks(xs, [label.replace(" (s)", "") for _, label in PARAMS], fontsize=10)
    ax.set_xlim(-0.6, len(PARAMS) - 0.4)
    ax.set_ylim(80, 100)
    ax.set_ylabel("truth inside 95% interval (%)")
    ax.legend(loc="lower left", fontsize=9, handletextpad=0.2, borderaxespad=0.2)
    ax.text(0.97, 0.04, f"n = {n}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    for ax, letter in zip(axes, "abcdef"):
        fs.panel_label(ax, letter, x=-0.32)
    fig.tight_layout()
    fs.save(fig, out("fig4_recovery_full"))


def figS8():
    """Supplementary Figure S8: the five-parameter fit from part of the data, each person
    alone (maximum likelihood) and pooled (empirical Bayes, a normal population on an
    unbounded scale fitted by EM with a Laplace approximation), from results/lhc_size
    (`python -m experiments.lhc_size`). The 100 histories of Figure 4, cut at N start
    decisions with the same share of their redo decisions. (a)-(e) RMSE against N, error
    bars 2 SE from a bootstrap over people (2000 resamples, seed 0). (f) the share of people
    whose pooled 95% interval contains the truth; grey band as in Figure 4 (f), for 100."""
    fs.apply()
    df = pd.read_csv(results("lhc_size") / "fits.csv")
    sizes = sorted(df.n.unique())
    fig, axes = fs.figure(2, 3, width=7.6, height=5.4)
    axes = axes.ravel()
    rng = np.random.default_rng(0)
    for ax, (p, label) in zip(axes, PARAMS):
        for j, (col, name, color, ls, mk) in enumerate(
                (("ml", "alone (ML)", fs.GREY, "--", "s"), ("po", "pooled", fs.MID, "-", "o"))):
            ys, es = [], []
            for n in sizes:
                e2 = ((df[df.n == n][f"{col}_{p}"] - df[df.n == n][f"true_{p}"]) ** 2).to_numpy()
                idx = rng.integers(0, len(e2), size=(2000, len(e2)))
                ys.append(np.sqrt(e2.mean()))
                es.append(float(np.std(np.sqrt(e2[idx].mean(axis=1)))))
            ax.errorbar(np.array(sizes) * (1 + 0.04 * (j - 0.5)), ys, yerr=[2 * e for e in es],
                        color=color, ls=ls, marker=mk, ms=6, mec="black", mew=0.6, capsize=0,
                        elinewidth=1.2, label=name)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Start decisions")
        ax.set_ylabel(f"{label} RMSE")
        ax.set_ylim(bottom=0)
    ax = axes[5]
    lo, hi = (100 * stats.binom.ppf(q, 100, 0.95) / 100 for q in (0.025, 0.975))
    ax.axhspan(lo, hi, color=fs.GREY, alpha=0.25, lw=0, zorder=0)
    ax.axhline(95, color=fs.GREY, ls="--", lw=1.2)
    marks = dict(zip([p for p, _ in PARAMS], ["o", "s", "D", "^", "v"]))
    shades = dict(zip([p for p, _ in PARAMS], [fs.DARK, fs.MID, fs.LIGHT, fs.DARK, fs.MID]))
    for p, label in PARAMS:
        cover = [100 * ((df[df.n == n][f"lo_{p}"] <= df[df.n == n][f"true_{p}"])
                        & (df[df.n == n][f"true_{p}"] <= df[df.n == n][f"hi_{p}"])).mean()
                 for n in sizes]
        ax.plot(sizes, cover, color=shades[p], marker=marks[p], ms=6, mec="black", mew=0.6,
                lw=1.4, label=label.replace(" (s)", ""))
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Start decisions")
    ax.set_ylabel("pooled 95% interval\ncontains truth (%)")
    ax.set_ylim(72, 100)
    ax.legend(fontsize=8, ncol=3, loc="lower center", handletextpad=0.2, columnspacing=0.6,
              handlelength=1.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=10)
    for ax, letter in zip(axes, "abcdef"):
        fs.panel_label(ax, letter, x=-0.32)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fs.save(fig, out("figS8_size_pooling"))


def figS11():
    """Supplementary Figure S11: a countdown on the redo prompt, from results/redo_pressure
    (`python -m experiments.redo_pressure`). Each person plays twice on one seed: with no
    countdown, and with a 3 s countdown on the redo prompt in alternate blocks of 25 start
    decisions. Under it their bound is λ × BOUND. (a) λ recovered against the truth, with
    the six parameters fitted together. (b) RMSE of each of the five shared parameters with
    no countdown and with it in blocks, error bars 2 SE from a bootstrap over people (2000
    resamples, seed 0). (c) share of countdown prompts that timed out (closed without a
    choice) and that ended in a redo, against λ. (d) bias when the fit knows the countdown's
    deadline but not λ (λ held at 1), mean and 95% t interval."""
    import glob

    fs.apply()
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(results("redo_pressure") / "runs_*.csv"))],
                   ignore_index=True)
    none = df[df.arm == "none"].set_index("person").sort_index()
    blk = df[df.arm == "blocks"].set_index("person").sort_index()
    both = none.index.intersection(blk.index)
    none, blk = none.loc[both], blk.loc[both]
    fig, axes = fs.figure(2, 2, width=7.4, height=6.6)
    (a, b), (c, d) = axes

    a.plot([0.3, 1.1], [0.3, 1.1], color=fs.GREY, ls="--", lw=1.2, zorder=1)
    a.scatter(blk.true_lam, blk.est_lam, s=16, color=fs.MID, edgecolors="black", linewidths=0.4,
              zorder=3)
    err = blk.est_lam - blk.true_lam
    a.text(0.04, 0.96, f"RMSE {np.sqrt(np.mean(err ** 2)):.3f}\nbias {err.mean():+.3f}\n"
           f"r {np.corrcoef(blk.true_lam, blk.est_lam)[0, 1]:.3f}", transform=a.transAxes,
           va="top", fontsize=9)
    a.set_xlabel(r"true $\lambda$ (bound kept under the countdown)")
    a.set_ylabel(r"recovered $\lambda$")

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(both), size=(2000, len(both)))
    xs = np.arange(len(PARAMS))
    for j, (frame, name, col) in enumerate(((none, "no countdown", fs.GREY),
                                            (blk, "countdown in blocks", fs.MID))):
        ys, es = [], []
        for p, _ in PARAMS:
            e2 = ((frame[f"est_{p}"] - frame[f"true_{p}"]) ** 2).to_numpy()
            ys.append(np.sqrt(e2.mean()))
            es.append(float(np.std(np.sqrt(e2[idx].mean(axis=1)))))
        b.bar(xs + (j - 0.5) * 0.38, ys, 0.36, yerr=[2 * e for e in es], color=col,
              edgecolor="black", lw=1.0, error_kw=dict(lw=1.2), label=name)
    b.set_xticks(xs, [label.replace(" (s)", "") for _, label in PARAMS], fontsize=10)
    b.set_ylabel("RMSE")
    b.legend(fontsize=9, loc="upper right")

    c.scatter(blk.true_lam, blk.timed_out, s=16, color=fs.DARK, edgecolors="black",
              linewidths=0.4, label="timed out")
    c.scatter(blk.true_lam, blk.taken_pressured, s=16, marker="s", color=fs.LIGHT,
              edgecolors="black", linewidths=0.4, label="redo taken")
    c.set_xlabel(r"true $\lambda$")
    c.set_ylabel("Share of countdown prompts")
    c.set_ylim(-0.03, 1.03)
    c.legend(fontsize=9, loc="upper right")

    d.axhline(0, color="black", lw=1.2)
    for i, (p, label) in enumerate(PARAMS[:4]):
        rel = 100 * (blk[f"ign_{p}"] - blk[f"true_{p}"]) / blk[f"true_{p}"]
        half = stats.t.ppf(0.975, len(rel) - 1) * rel.std(ddof=1) / np.sqrt(len(rel))
        d.errorbar(i, rel.mean(), yerr=half, color=fs.MID, marker="o", ms=7, mec="black",
                   mew=0.6, capsize=0, elinewidth=1.4)
    d.set_xticks(range(4), [label for _, label in PARAMS[:4]], fontsize=10)
    d.set_ylabel(r"Bias with $\lambda$ ignored (%)")
    for ax, letter in zip(axes.ravel(), "abcd"):
        fs.panel_label(ax, letter)
    fig.subplots_adjust(hspace=0.5, wspace=0.42)
    fs.save(fig, out("figS11_redo_pressure"))
