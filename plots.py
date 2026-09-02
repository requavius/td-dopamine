from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).parent / "assets"

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
