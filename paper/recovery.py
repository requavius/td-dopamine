"""Paper Figure 4 (full model): recovery of f, k, γ, BOUND and T0 fitted together, from
results/recovery_lhc/fits.csv (`python -m experiments.recovery_lhc fit`).

People are a Latin hypercube over f, k in 0.2 to 1.5, γ in 0.6 to 0.97, BOUND in 1.0 to
2.0 and T0 in 0 to 0.4 s. Each plays the app's task under InfoController for 300 start
decisions plus their redo decisions; all five parameters are fitted by maximum
likelihood with the closed-form likelihood (temporal.fastlik). α 0.05, SIGMA 0.7,
ACHIEVEMENT_SCALE 4, continuing bootstrap, T_DUR 7.0.

(a)-(e) true against recovered, with RMSE, bias and r. (f) the error bars checked: each
fit's standardized error (recovered − true) / SE, SE from the observed information,
pooled over the five parameters for fits away from a parameter bound, against the
standard normal they should follow if the SEs are honest.

    python -m paper fig4lhc
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import style as fs
from .common import OUT, RESULTS

PARAMS = [("f", "f"), ("k", "k"), ("g", r"$\gamma$"), ("bound", "bound"), ("t0", r"$T_0$ (s)")]


def fig4lhc():
    fs.apply()
    df = pd.read_csv(RESULTS / "recovery_lhc" / "fits.csv")
    fig, axes = fs.figure(2, 3, width=7.6, height=5.4)
    axes = axes.ravel()
    for ax, (p, label) in zip(axes, PARAMS):
        truth, est = df[f"true_{p}"].to_numpy(), df[f"est_{p}"].to_numpy()
        err = est - truth
        lo = min(truth.min(), est.min())
        hi = max(truth.max(), est.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=fs.GREY, ls="--", lw=1.2, zorder=1)
        ax.scatter(truth, est, s=16, color=fs.MID, edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"recovered {label}")
        ax.text(0.04, 0.96, f"RMSE {np.sqrt(np.mean(err ** 2)):.3f}\nbias {err.mean():+.3f}\n"
                f"r {np.corrcoef(truth, est)[0, 1]:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9)
    ax = axes[5]
    away = ~df[[f"at_bound_{p}" for p, _ in PARAMS]].any(axis=1)
    z = np.concatenate([((df[f"est_{p}"] - df[f"true_{p}"]) / df[f"se_obs_{p}"])[away].dropna()
                        for p, _ in PARAMS])
    bins = np.linspace(-4, 4, 33)
    ax.hist(np.clip(z, -4, 4), bins=bins, density=True, color=fs.LIGHT, edgecolor="black", lw=0.5)
    x = np.linspace(-4, 4, 200)
    ax.plot(x, stats.norm.pdf(x), color=fs.DARK, lw=fs.LINE)
    ax.set_ylim(0, 0.62)  # room for the text above the bars
    ax.set_xlabel("(recovered − true) / SE")
    ax.set_ylabel("density")
    ax.text(0.04, 0.96, f"SD {z.std():.2f}\ninside ±1.96: {(np.abs(z) < 1.96).mean():.0%}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9)
    for ax, letter in zip(axes, "abcdef"):
        fs.panel_label(ax, letter, x=-0.32)
    fig.tight_layout()
    fs.save(fig, OUT / "fig4_recovery_full")
