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


# The same argument in 3D, with drift as height. A decision is a tilted plane (a roof)
# mu = V*f - C*k through the origin. Measuring its drift floods the space to that height;
# the shoreline is every (f, k) that fits. Standing the shoreline up gives a wall, and
# two walls with different slopes meet in one vertical line: one (f, k) at every height.
IDENT_Z = (-1.0, 0.6)


def _shoreline(v, c, mu, n=200):
    f = np.linspace(0, 2, n)
    k = (f * v - mu) / c
    keep = (k >= 0) & (k <= 2)
    return f[keep], k[keep]


def _wall(ax, f, k, color, zlim=IDENT_Z, alpha=0.22):
    Z = np.vstack([np.full_like(f, zlim[0]), np.full_like(f, zlim[1])])
    ax.plot_surface(np.vstack([f, f]), np.vstack([k, k]), Z, color=color, alpha=alpha,
                    linewidth=0, shade=False)


def _box(ax, title, zlim=IDENT_Z):
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(*zlim)
    ax.set_xlabel("f (reward sensitivity)", labelpad=4)
    ax.set_ylabel("k (effort aversion)", labelpad=4)
    ax.set_zlabel("drift  μ", labelpad=2)
    ax.set_title(title, fontsize=11, pad=0)
    ax.view_init(elev=22, azim=-125)
    ax.tick_params(labelsize=7)


def plot_identifiability_3d(out_path=ASSETS / "identifiability_3d.png"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f_true, k_true = IDENT_TRUE
    zlo, zhi = IDENT_Z
    g = np.linspace(0, 2, 30)
    F, K = np.meshgrid(g, g)
    (v1, c1), (v2, c2), (v3, c3) = (0.25, 0.30), (0.25, 0.75), (0.50, 0.60)
    mu1, mu2, mu3 = (f_true * v - k_true * c for v, c in ((v1, c1), (v2, c2), (v3, c3)))
    blue, orange = IDENT_COLORS

    fig = plt.figure(figsize=(12, 10.5))

    # (a) the roof: what drift every possible person would produce on this decision
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot_surface(F, K, F * v1 - K * c1, cmap="Blues", alpha=0.75, linewidth=0)
    ax.plot([f_true], [k_true], [mu1], "o", color="black", markersize=5)
    ax.text(f_true, k_true, mu1 + 0.12, "true person", fontsize=8)
    _box(ax, f"(a) One decision is a tilted plane\nμ = {v1}·f − {c1}·k")

    # (b) flood to the measured drift; the shoreline is everyone who fits
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    ax.plot_surface(F, K, F * v1 - K * c1, cmap="Blues", alpha=0.45, linewidth=0)
    ax.plot_surface(F, K, np.full_like(F, mu1), color="#7fb7d9", alpha=0.25, linewidth=0,
                    shade=False)
    f, k = _shoreline(v1, c1, mu1)
    ax.plot(f, k, np.full_like(f, mu1), color=blue, linewidth=3, label="shoreline")
    ax.plot(f, k, np.full_like(f, zlo), color=blue, linewidth=1.5, linestyle="--",
            label="its shadow on the floor")
    ax.plot([f_true], [k_true], [mu1], "o", color="black", markersize=5)
    ax.legend(loc="upper left", fontsize=8)
    _box(ax, f"(b) Measured drift = {mu1:.2f}: flood to that height.\n"
             "Every (f, k) on the shoreline fits")

    # (c) two decisions with different mixes: their walls meet in one vertical line
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    for v, c, mu, col, name in ((v1, c1, mu1, blue, "decision 1"),
                                (v2, c2, mu2, orange, "decision 2")):
        f, k = _shoreline(v, c, mu)
        _wall(ax, f, k, col)
        ax.plot(f, k, np.full_like(f, mu), color=col, linewidth=2.5,
                label=f"{name}: V/C = {v / c:.2f}")
        ax.plot(f, k, np.full_like(f, zlo), color=col, linewidth=1.2, linestyle="--")
    ax.plot([f_true] * 2, [k_true] * 2, [zlo, zhi], color="black", linewidth=2.5,
            label="walls meet: f = 0.8, k = 0.8\nat every height")
    ax.legend(loc="upper left", fontsize=8)
    _box(ax, "(c) Two decisions, different mixes:\nthe walls cross in one vertical line")

    # (d) same mix: the second wall is the first one again
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    for v, c, mu, col, lw, name in ((v1, c1, mu1, blue, 4.5, "decision 1"),
                                    (v3, c3, mu3, orange, 1.8, "decision 2")):
        f, k = _shoreline(v, c, mu)
        _wall(ax, f, k, col, alpha=0.18)
        ax.plot(f, k, np.full_like(f, zlo), color=col, linewidth=lw,
                label=f"{name}: V/C = {v / c:.2f}")
    ax.plot([f_true], [k_true], [zlo], "o", color="black", markersize=5)
    ax.legend(loc="upper left", fontsize=8)
    _box(ax, "(d) Two decisions, same mix:\none wall, so no single crossing")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
