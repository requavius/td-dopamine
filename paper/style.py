"""House style for the figures: Helvetica, thick black axes, no gridlines, ordinal blues,
square light-blue diagram boxes, (a)/(b) panel labels.

    from paper import style as fs
    fs.apply()
    fig, axes = fs.figure(2, 2)
    ...
    fs.save(fig, OUT / "name")   # writes name.png and name.pdf
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# Ordinal blues, light -> dark (validated: monotone lightness, light end clears white).
LIGHT = "#56B0F2"
MID = "#2F6FE0"
DARK = "#0B1F8F"
BLUES = (DARK, MID, LIGHT)  # low, medium, high -- the reference figure's order
BOX_FILL = "#DCEBFA"        # diagram boxes
INK = "black"
GREY = "#8C8C8C"            # reference marks only (bounds, deadlines, rugs) -- never data series

FONT = 12
LINE = 2.0      # data lines
AXIS = 1.6      # spines and ticks
BOX_EDGE = 2.5  # diagram box borders


def apply():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": FONT,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": AXIS,
        "xtick.major.width": AXIS,
        "ytick.major.width": AXIS,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titleweight": "normal",
        "lines.linewidth": LINE,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
    })


def figure(rows=1, cols=1, width=7.2, height=None, **kw):
    height = height or 2.8 * rows
    return plt.subplots(rows, cols, figsize=(width, height), **kw)


# "(a)" outside the top-left corner of an axes, like the reference.
def panel_label(ax, letter, x=-0.18, y=1.08):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=13, va="bottom", ha="left")


# Square-cornered box with a thick black border and light blue fill; returns its centre.
def box(ax, x, y, w, h, text, fill=BOX_FILL, fontsize=12, **kw):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK,
                           linewidth=BOX_EDGE, joinstyle="miter", **kw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return x + w / 2, y + h / 2


def arrow(ax, start, end, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                 linewidth=AXIS, color=INK, shrinkA=0, shrinkB=0, **kw))


# An axes used as a blank canvas for a diagram, in 0..1 coordinates.
def canvas(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


# A saved figure's PNG and PDF under a second name.
def copy(stem, to, suffixes=(".png", ".pdf")):
    Path(to).parent.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        shutil.copyfile(Path(stem).with_suffix(suffix), Path(to).with_suffix(suffix))
