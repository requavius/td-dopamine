"""Recovery plots."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"

PANELS = [
    ("f", "f_true", "f_hat", "reward sensitivity", "#2196F3"),
    ("k", "k_true", "k_hat", "effort aversion", "#FF5722"),
]


def plot_recovery(res, out_path, title="Initiation DDM parameter recovery", reps=5):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (name, tcol, hcol, label, color) in zip(axes, PANELS):
        truth, est = res[tcol].to_numpy(), res[hcol].to_numpy()
        lo = min(truth.min(), est.min())
        hi = max(truth.max(), est.max())
        pad = 0.05 * (hi - lo)
        lims = [lo - pad, hi + pad]

        ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")
        # Jitter x only so the reps at each grid point are distinguishable
        jitter = (np.arange(len(truth)) % reps - (reps - 1) / 2) * 0.012
        ax.scatter(truth + jitter, est, s=28, alpha=0.65, color=color,
                   edgecolors="none", label="fits")

        means = res.groupby(tcol)[hcol].mean()
        ax.plot(means.index, means.to_numpy(), "o-", color="black",
                markersize=5, linewidth=1.2, alpha=0.8, label="mean per level")

        r = float(np.corrcoef(truth, est)[0, 1])
        rmse = float(np.sqrt(np.mean((est - truth) ** 2)))
        stats[name] = {"r": r, "rmse": rmse}

        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"recovered {name}")
        ax.set_title(f"{name} ({label})\nr = {r:.3f}   RMSE = {rmse:.3f}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return stats


# Why one decision cannot separate f and k, and two with different mixes can.
# Each decision fixes only its drift, f*V - k*C = mu, so on the (f, k) plane it is a
# band of slope V/C (width set by how well mu is known). People inside every band fit.
IDENT_TRUE = (0.8, 0.8)
IDENT_EPS = 0.03
IDENT_PANELS = [
    ("(a) One decision", [(0.25, 0.30)]),
    ("(b) Two decisions, different mixes", [(0.25, 0.30), (0.25, 0.75)]),
    ("(c) Two decisions, same mix", [(0.25, 0.30), (0.50, 0.60)]),
]
IDENT_COLORS = ["#2196F3", "#FF5722"]


def plot_identifiability(out_path=ASSETS / "identifiability.png", eps=IDENT_EPS):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f_true, k_true = IDENT_TRUE
    grid = np.linspace(0, 2, 600)
    F, K = np.meshgrid(grid, grid)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, (title, decisions) in zip(axes, IDENT_PANELS):
        fits = np.ones_like(F, dtype=bool)
        for i, (v, c) in enumerate(decisions):
            mu = f_true * v - k_true * c
            fits &= np.abs(F * v - K * c - mu) <= eps
            color = IDENT_COLORS[i]
            lo, hi = (grid * v - mu - eps) / c, (grid * v - mu + eps) / c
            ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(grid, (grid * v - mu) / c, color=color, linewidth=1.6,
                    label=f"V = {v:.2f}, C = {c:.2f}  (slope V/C = {v / c:.2f})")
        ax.contourf(F, K, fits, levels=[0.5, 1.5], colors=["#1d6b58"], alpha=0.55)
        ax.plot(f_true, k_true, "o", color="black", markersize=5)
        ax.plot(f_true, k_true, "o", markerfacecolor="none", markeredgecolor="black",
                markersize=12, markeredgewidth=1.2, label="true person (0.8, 0.8)")
        ax.set_title(title)
        ax.set_xlabel("f (reward sensitivity)")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    axes[0].set_ylabel("k (effort aversion)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
