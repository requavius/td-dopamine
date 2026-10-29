"""Draw the paper's figures from results/ into paper/figures/.

    python -m paper                 # every figure
    python -m paper fig7 fig8       # some of them
    python -m paper explainers      # the explanatory figures (paper/figures/explainers/)
    python -m paper --captions      # print what each figure's data was generated under

Drawn from other results, for example a run under changed conditions
(`python -m experiments.controllers --set reps=4 --out controllers_long`):

    python -m paper fig8 --results controllers=controllers_long

draws into paper/figures/controllers_long/ (or --out NAME), so the paper's figures are kept.
Every drawing also writes conditions.md beside the figures: what each figure's data was
generated under, from the conditions.json of the results it read (paper/captions.py).
"""
import argparse
import sys

from . import (captions, controllers, explainers, identifiability, misspecification, model,
               recovery, shortcuts, task)
from .common import out, use

FIGURES = {
    "fig1": model.fig1,               # the model
    "fig2": explainers.fig2,          # drift planes
    "fig3": explainers.fig3,          # patience (needs a recorded session)
    "fig4": model.fig4,               # recovery from one person
    "fig4lhc": recovery.fig4lhc,      # recovery of f, k, γ, BOUND, T0 together
    "fig5": misspecification.fig5,    # what stays biased once γ is fitted
    "fig6": shortcuts.fig6,           # shortcuts in the fit
    "fig7": task.fig7,                # difficulty and feedback
    "fig8": controllers.fig8,         # controllers on the grid (γ 0.9, bound 1.5)
    "fig8lhc": controllers.fig8lhc,   # controllers on the hypercube people, all five fitted
    "fig9": controllers.fig9,         # planning with γ, the bound and T0
    "figS2": model.figS2,             # timed-out trials kept vs dropped
    "figS3": shortcuts.figS3,         # the fit factorial
    "figS4": task.figS4,              # reward by difficulty, one-person feedback test
    "figS5": controllers.figS5,       # controllers over time, all four panels
    "figS6": task.figS6,              # feedback shown - hidden, by γ
    "figS7": misspecification.figS7,  # constants differ, γ held at 0.9 (the old Figure 5)
    "figS8": recovery.figS8,          # the fit from less data, alone and pooled
    "figS9": misspecification.figS9,  # α and the reward, measured
    "figS10": controllers.figS10,     # feedback in blocks, redos mid-set or at the end
    "figS11": recovery.figS11,        # a countdown on the redo prompt
    "identifiability": identifiability.draw,  # 2D and 3D identifiability
}
EXTRA = {"explainers": explainers.draw}


def source(text):
    name, _, folder = text.partition("=")
    if not folder:
        raise argparse.ArgumentTypeError(f"{text!r}: expected experiment=folder")
    return name, folder


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="figures to draw (default: all)")
    p.add_argument("--results", type=source, action="append", default=[],
                   metavar="EXPERIMENT=FOLDER", help="read EXPERIMENT from results/FOLDER/")
    p.add_argument("--out", help="draw into paper/figures/OUT/ (default: the folders' names)")
    p.add_argument("--captions", action="store_true", help="print the captions, draw nothing")
    a = p.parse_args()
    names = a.names or list(FIGURES)
    for name in names:
        if name not in FIGURES and name not in EXTRA:
            sys.exit(f"unknown figure {name!r}; choose from {', '.join([*FIGURES, *EXTRA])}")
    sources = dict(a.results)
    unused = set(sources) - {e for n in names for e in captions.READS.get(n, ())}
    if unused:
        sys.exit(f"--results {', '.join(sorted(unused))}: no figure drawn here reads it")
    use(sources, a.out)
    lines = [f"## {n}\n\n{captions.CAPTIONS[n]()}\n" for n in names if n in captions.CAPTIONS]
    if a.captions:
        print("\n".join(lines))
        sys.exit()
    for name in names:
        print(f"drawing {name}", flush=True)
        (FIGURES.get(name) or EXTRA[name])()
    if lines:
        head = "# What each figure's data was generated under\n\n"
        if sources:
            head += "Drawn from " + ", ".join(f"results/{f}/ for {e}" for e, f in sources.items()) + ".\n\n"
        out().mkdir(parents=True, exist_ok=True)
        (out() / "conditions.md").write_text(head + "\n".join(lines))
    print(f"wrote {out()}")
