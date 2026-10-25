"""Why one decision cannot separate f and k, and two with different value/cost mixes can.

Each decision fixes only its drift, f·V − k·C = μ, so on the (f, k) plane it is a band of
slope V/C, as wide as μ is uncertain (±0.03 here). The people inside every band fit all
the decisions. The person is f = k = 0.8; decisions 1, 2 and 3 are those of Figure 2
(V 0.25, C 0.30; V 0.25, C 0.75; V 0.50, C 0.60). No simulation: drifts are exact.

    python -m paper identifiability   # identifiability (2D) and identifiability_3d
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from temporal.model.accumulator import initiation_drift

from . import explainers
from . import style as fs
from .common import OUT, ROOT

TRUE = explainers.PLANE_TRUE
DECISIONS = dict(enumerate(explainers.PLANE_DECISIONS, start=1))
COLOURS = {1: fs.DARK, 2: fs.LIGHT, 3: fs.LIGHT}  # as in Figure 2
EPS = 0.03
PANELS = (("a", "Decision 1", (1,)), ("b", "Decisions 1 and 2", (1, 2)),
          ("c", "Decisions 1 and 3", (1, 3)))


def fig2d():
    fs.apply()
    grid = np.linspace(0, 2, 801)
    F, K = np.meshgrid(grid, grid)
    fig, axes = fs.figure(1, 3, width=7.4, height=3.2, sharey=True)
    fig.subplots_adjust(wspace=0.25, top=0.86)
    for ax, (letter, title, used) in zip(axes, PANELS):
        fits = np.ones_like(F, dtype=bool)
        for n in used:
            v, c = DECISIONS[n]
            mu = initiation_drift(v, c, *TRUE)
            fits &= np.abs(initiation_drift(v, c, F, K) - mu) <= EPS
            ax.fill_between(grid, (v * grid - mu - EPS) / c, (v * grid - mu + EPS) / c,
                            color=COLOURS[n], alpha=0.3, lw=0, zorder=1)
            ax.plot(grid, (v * grid - mu) / c, color=COLOURS[n],
                    lw=fs.LINE * (2 if n == 1 and 3 in used else 1), zorder=2)
        ax.contourf(F, K, fits, levels=[0.5, 1.5], colors=[fs.MID], alpha=0.8, zorder=3)
        ax.contour(F, K, fits, levels=[0.5], colors=fs.INK, linewidths=0.8,
                   linestyles="solid", zorder=3.1)
        ax.plot(*TRUE, marker="+", color=fs.INK, ms=9, mew=1.8, zorder=5)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect("equal")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xlabel("$f$")
        ax.set_title(title, fontsize=12, pad=6)
        fs.panel_label(ax, letter, x=-0.2, y=1.04)
    axes[0].set_ylabel("$k$")
    handles = [Line2D([], [], color=COLOURS[n], lw=fs.LINE,
                      label=f"{n}: V/C = {DECISIONS[n][0] / DECISIONS[n][1]:.2f}")
               for n in (1, 2, 3)]
    handles += [Patch(facecolor=fs.MID, edgecolor=fs.INK, lw=1.0, label="fits all"),
                Line2D([], [], marker="+", color=fs.INK, ls="none", ms=9, mew=1.8,
                       label="$f = k = 0.8$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
    fs.save(fig, OUT / "identifiability")


# The 3D version is Figure 2's drift planes: the same decisions, drift as height.
def fig3d():
    explainers.draw("planes")
    fs.copy(explainers.OUT / "6_drift_planes", OUT / "identifiability_3d")


def draw():
    fig2d()
    fig3d()
    # The copies kept in assets/ for the repo.
    for name in ("identifiability", "identifiability_3d"):
        fs.copy(OUT / name, ROOT / "assets" / name, suffixes=(".png",))
