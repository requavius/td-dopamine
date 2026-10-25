"""Draw the paper's figures from results/ into paper/figures/.

    python -m paper                 # every figure
    python -m paper fig7 fig8       # some of them
    python -m paper explainers      # the explanatory figures (paper/figures/explainers/)
"""
import sys

from . import controllers, explainers, identifiability, misspecification, model, shortcuts, task
from .common import OUT

FIGURES = {
    "fig1": model.fig1,               # the model
    "fig2": explainers.fig2,          # drift planes
    "fig3": explainers.fig3,          # patience (needs a recorded session)
    "fig4": model.fig4,               # recovery from one person
    "fig5": misspecification.fig5,    # the person's constants differ from the fit's
    "fig6": shortcuts.fig6,           # shortcuts in the fit
    "fig7": task.fig7,                # difficulty and feedback
    "fig8": controllers.fig8,         # controllers over time
    "figS2": model.figS2,             # timed-out trials kept vs dropped
    "figS3": shortcuts.figS3,         # the fit factorial
    "figS4": task.figS4,              # reward by difficulty, one-person feedback test
    "figS5": controllers.figS5,       # controllers over time, all four panels
    "identifiability": identifiability.draw,  # 2D and 3D identifiability
}
EXTRA = {"explainers": explainers.draw}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        if name not in FIGURES and name not in EXTRA:
            sys.exit(f"unknown figure {name!r}; choose from {', '.join([*FIGURES, *EXTRA])}")
    for name in names:
        print(f"drawing {name}", flush=True)
        (FIGURES.get(name) or EXTRA[name])()
    print(f"wrote {OUT}")
