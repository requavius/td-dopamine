"""Figures explaining how f, k and patience are recovered, paper Figures 2 and 3 among them.
All: SIGMA 0.7, BOUND 1.5, T_DUR 7 s, gamma 0.9, ACHIEVEMENT_SCALE 4, continuing bootstrap,
T0 0; simulated people have k = 0.8. Written to paper/figures/explainers/ (about 4 min).

    python -m paper explainers   # every figure below
    python -m paper fig2 fig3    # the two the paper uses

Figure 3 ("patience") and "joint fit pull" are drawn from a recorded session, the
decisions.jsonl named by $TEMPORAL_RECORD (default data/decisions.jsonl, the desktop app's).
Without one they are skipped. The rest need no data.
"""
import json
import math
import os
from pathlib import Path

import numpy as np

from temporal.model import accumulator as A
from temporal.model import persistence as P

from . import style as fs
from .common import OUT as PAPER_OUT
from .common import ROOT

OUT = PAPER_OUT / "explainers"
DATA = Path(os.environ.get("TEMPORAL_RECORD", ROOT / "data" / "decisions.jsonl"))
K_TRUE = 0.8
F_PEOPLE = (0.4, 0.8, 1.6)  # low, medium, high
F_GRID = np.round(np.arange(0.0, 3.0001, 0.05), 4)
K_GRID = np.round(np.arange(0.0, 3.0001, 0.1), 4)
K_FINE = np.round(np.arange(0.0, 3.0001, 0.025), 4)  # likelihood surfaces: the coupled ridge is narrow
F_FINE = np.round(np.arange(0.0, 2.6001, 0.02), 4)


# ---------------------------------------------------------------- model helpers

_models = {}


def solve(f, k, value, cost):
    key = (round(f, 4), round(k, 4))
    if key not in _models:
        _models[key] = A.build_model({"f_val": key[0], "k_val": key[1]})
    return _models[key].solve(conditions={"value": value, "cost": cost})


def p_go(f, k, value, cost):
    return solve(f, k, value, cost).prob("_top")


# Expected log-likelihood of n decisions per condition from `truth`, under (f, k).
def expected_ll(truth, design, n, f, k):
    total = 0.0
    for (v, c), sol_true in zip(design, truth):
        sol = solve(f, k, v, c)
        for side in ("_top", "_bottom"):
            q = sol_true.pdf(side) * A.DT
            total += n * np.sum(q * np.log(np.maximum(sol.pdf(side), 1e-300)))
        pu = sol_true.prob_undecided()
        total += n * pu * math.log(max(sol.prob_undecided(), 1e-300))
    return total


# Log-likelihood of observed decisions [(value, cost, GO, RT)] under (f, k).
def sample_ll(rows, f, k):
    total = 0.0
    sols = {}
    for v, c, go, rt in rows:
        if (v, c) not in sols:
            sols[v, c] = solve(f, k, v, c)
        sol = sols[v, c]
        if go is None or rt is None or rt > A.T_DUR:
            total += math.log(max(sol.prob_undecided(), 1e-300))
            continue
        i = min(int(round(rt / A.DT)), len(sol.t_domain) - 1)
        total += math.log(max(sol.pdf("_top" if go else "_bottom")[i], 1e-300))
    return total


def simulate(f, k, design, n, seed):
    np.random.seed(seed)
    rows = []
    for v, c in design:
        for _ in range(n):
            go, rt = A._draw(f, k, A.snap_value(v), A.snap_cost(c), 0.0)
            rows.append((v, c, go, rt))
    return rows


def surface(fn):
    return np.array([[fn(f, k) for f in F_GRID] for k in K_FINE])  # rows k, cols f


# ------------------------------------------------------------------------ data


def records():
    return [json.loads(line) for line in DATA.open()]


def human_decisions(recs):
    return [(r["value"], r["cost"], r["GO"], r["RT"]) for r in recs if r["node"] in ("start", "stage")]


def human_attempts(recs):
    # As Session._load: answered = kept going that long; skipped or left = gave up.
    raw = [{"p": r.get("p"), "rt": r.get("rt"), "skipped": r["node"] != "answer",
            "stage_amt": r.get("stage_amt") or 4}
           for r in recs if r["node"] in ("answer", "skip")
           or (r["node"] == "abandon" and r.get("screen") == "question" and r.get("rt") is not None)]
    return P.attempts_to_frame(raw)


# --------------------------------------------------------------------- figures

INDEPENDENT = [(v, c) for v in (-0.25, 0.25, 0.75, 1.25) for c in (0.15, 0.5, 0.85)]  # 12 conditions
COUPLED = [(v, 0.7 * v) for v in (0.1, 0.4, 0.7, 1.0, 1.3, 1.6)]                     # 6 conditions
N_DECISIONS = 300


def fig_recovery():
    fig = fs.plt.figure(figsize=(7.6, 6.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.5, 1.0), hspace=0.4, wspace=0.42)

    # (a) the pipeline
    ax = fs.canvas(fig.add_subplot(grid[0, :]))
    fs.panel_label(ax, "a", x=-0.02, y=0.98)
    h, y = 0.5, 0.42
    boxes = [(0.00, 0.15, "Task state"),
             (0.19, 0.16, "value, cost"),
             (0.39, 0.25, "Accumulator\n$f\\cdot$value $-$ $k\\cdot$cost"),
             (0.68, 0.13, "GO, RT"),
             (0.85, 0.15, "$\\hat{f}$, $\\hat{k}$")]
    for i, (x, w, text) in enumerate(boxes):
        fs.box(ax, x, y, w, h, text, fontsize=11.5)
        if i + 1 < len(boxes):
            fs.arrow(ax, (x + w + 0.008, y + h / 2), (boxes[i + 1][0] - 0.008, y + h / 2))
    # the fit inverts the accumulator
    fs.arrow(ax, (0.925, y - 0.02), (0.515, y - 0.02), connectionstyle="arc3,rad=-0.2")
    ax.text(0.72, 0.05, "maximum likelihood", ha="center", va="center", fontsize=11)

    # (b) the accumulator: paths, bounds and the deadline
    ax = fig.add_subplot(grid[1, 0])
    fs.panel_label(ax, "b")
    rng = np.random.default_rng(4)
    dt, drift = 0.01, 0.25
    colours = {"GO": fs.DARK, "STAY": fs.MID, "undecided": fs.LIGHT}
    shown = {"GO": 0, "STAY": 0, "undecided": 0}
    want = {"GO": 3, "STAY": 2, "undecided": 2}
    while shown != want:
        x = np.concatenate([[0], np.cumsum(drift * dt + A.SIGMA * math.sqrt(dt) * rng.standard_normal(int(A.T_DUR / dt)))])
        hit = np.flatnonzero(np.abs(x) >= A.BOUND)
        outcome = "undecided" if not len(hit) else ("GO" if x[hit[0]] > 0 else "STAY")
        if shown[outcome] >= want[outcome]:
            continue
        shown[outcome] += 1
        end = hit[0] + 1 if len(hit) else len(x)
        ax.plot(np.arange(end) * dt, np.clip(x[:end], -A.BOUND, A.BOUND), color=colours[outcome],
                lw=1.4, label=outcome if shown[outcome] == 1 else None)
    for b in (A.BOUND, -A.BOUND):
        ax.axhline(b, color=fs.INK, lw=fs.AXIS)
    ax.axvline(A.T_DUR, color=fs.INK, lw=fs.AXIS, ls="--")
    ax.text(A.T_DUR, A.BOUND + 0.3, "$T_{DUR}$", ha="center", fontsize=11)
    ax.set_xlim(0, A.T_DUR + 0.3)
    ax.set_ylim(-A.BOUND - 0.4, A.BOUND + 0.8)
    ax.set_yticks([-A.BOUND, 0, A.BOUND])
    ax.set_yticklabels(["STAY", "0", "GO"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Evidence")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(n) for n in ("GO", "STAY", "undecided")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, handlelength=1.0,
              columnspacing=0.8, handletextpad=0.4)

    # (c) f is the slope of the choice curve
    ax = fig.add_subplot(grid[1, 1])
    fs.panel_label(ax, "c")
    values = np.round(np.arange(-1.0, 2.0001, 0.05), 4)
    for f, colour, name in zip(F_PEOPLE, fs.BLUES, ("Low", "Medium", "High")):
        ax.plot(values, [p_go(f, K_TRUE, v, 0.5) for v in values], color=colour, label=f"{name} ({f})")
    ax.set_xlabel("Value")
    ax.set_ylabel("P(GO)")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(title="$f$", loc="lower right", handlelength=1.2)
    fs.save(fig, OUT / "1_f_recovery_pipeline")

    # standalone (d): per-person estimates, like the reference's ridge plot
    fig, ax = fs.figure(1, 1, width=4.2, height=3.6)
    for row, (f, colour) in enumerate(zip(F_PEOPLE, fs.BLUES)):
        rows = simulate(f, K_TRUE, INDEPENDENT, N_DECISIONS // len(INDEPENDENT), seed=10 + row)
        ll = np.array([[sample_ll(rows, fv, kv) for fv in F_FINE] for kv in K_GRID])
        post = np.exp(ll - ll.max()).sum(axis=0)  # flat prior, k marginalised
        post = 0.85 * post / post.max()
        ax.fill_between(F_FINE, row, row + post, color=colour, lw=0, zorder=3 - row)
        ax.plot(F_FINE, row + post, color=fs.INK, lw=1.0, zorder=3 - row)
        ax.plot([f, f], [row, row + 0.95], color=fs.INK, lw=1.4, ls="--", zorder=4)
        _models.clear()
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Person {i + 1}" for i in range(3)])
    ax.set_xlabel("$f$")
    ax.set_xlim(0, 2.6)
    ax.set_ylim(-0.05, 3.0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fs.save(fig, OUT / "1d_f_per_person")


def fig_identifiability():
    fig, axes = fs.figure(2, 2, width=7.4, height=6.6)
    fig.subplots_adjust(hspace=0.48, wspace=0.42)
    (a, b), (c, d) = axes

    # (a) the two designs
    fs.panel_label(a, "a")
    a.scatter(*zip(*INDEPENDENT), s=60, marker="s", color=fs.DARK, label="Independent", zorder=3)
    a.scatter(*zip(*COUPLED), s=60, marker="o", color=fs.LIGHT, edgecolor=fs.INK, lw=1, label="Coupled", zorder=3)
    a.set_xlabel("Value")
    a.set_ylabel("Cost")
    a.legend(loc="upper left", handletextpad=0.2)
    a.set_ylim(0, 1.3)

    # (b) at a fixed cost, f is the slope; cost shifts the curve by k * cost
    fs.panel_label(b, "b")
    values = np.round(np.arange(-1.0, 2.0001, 0.05), 4)
    for cc, colour in zip((0.15, 0.5, 0.85), fs.BLUES):
        b.plot(values, [p_go(0.8, K_TRUE, v, cc) for v in values], color=colour, label=f"{cc}")
        rows = simulate(0.8, K_TRUE, [(v, cc) for v in (-0.75, -0.25, 0.25, 0.75, 1.25, 1.75)], 40, seed=3)
        for v in sorted({r[0] for r in rows}):
            gos = [r[2] == 1 for r in rows if r[0] == v]
            b.plot(v, np.mean(gos), "o", color=colour, mec=fs.INK, mew=0.8, ms=6)
    b.set_xlabel("Value")
    b.set_ylabel("P(GO)")
    b.set_ylim(0, 1)
    b.set_yticks([0, 0.5, 1])
    b.legend(title="Cost", loc="lower right", handlelength=1.2)

    # (c, d) likelihood over (f, k) for each design; contours at 1, 2, 3 SD (2 df)
    levels = [-5.9, -3.1, -1.15, 0]
    for ax, design, letter in ((c, INDEPENDENT, "c"), (d, COUPLED, "d")):
        fs.panel_label(ax, letter)
        truth = [solve(0.8, K_TRUE, v, cc) for v, cc in design]
        n = N_DECISIONS // len(design)
        ll = surface(lambda fv, kv: expected_ll(truth, design, n, fv, kv))
        ax.contourf(F_GRID, K_FINE, ll - ll.max(), levels=levels, colors=[fs.BOX_FILL, fs.LIGHT, fs.MID])
        ax.contour(F_GRID, K_FINE, ll - ll.max(), levels=levels[:-1], colors=fs.INK, linewidths=1.0,
                   linestyles="solid")
        ax.plot(0.8, K_TRUE, marker="+", color=fs.INK, ms=14, mew=2.2)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 2.5)
        ax.set_aspect("equal")
        ax.set_xlabel("$f$")
        ax.set_ylabel("$k$")
        _models.clear()
    fs.save(fig, OUT / "2_identifiability")


def fig_patience():
    recs = records()
    att = human_attempts(recs)
    decisions = human_decisions(recs)
    f0 = k0 = 0.8  # the controller's prior, which it uses below 40 decided rows
    pats = np.exp(np.linspace(math.log(0.5), math.log(40), 120))
    nll = np.array([P.neg_log_likelihood(f0, k0, att, p) for p in pats])
    p_fit = P.fit_patience(f0, k0, att)

    fig = fs.plt.figure(figsize=(7.4, 6.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.6, 1.0), hspace=0.42, wspace=0.42)

    # (a) the race on every question
    ax = fs.canvas(fig.add_subplot(grid[0, :]))
    fs.panel_label(ax, "a", x=-0.02, y=0.98)
    fs.box(ax, 0.0, 0.33, 0.2, 0.34, "Question")
    fs.box(ax, 0.32, 0.6, 0.34, 0.34, "Solve time")
    fs.box(ax, 0.32, 0.06, 0.34, 0.34, "Give-up accumulator\n$k\\cdot$cost $-$ $f\\cdot$value")
    fs.box(ax, 0.8, 0.6, 0.18, 0.34, "Answer")
    fs.box(ax, 0.8, 0.06, 0.18, 0.34, "Skip")
    fs.arrow(ax, (0.2, 0.55), (0.315, 0.77))
    fs.arrow(ax, (0.2, 0.45), (0.315, 0.23))
    fs.arrow(ax, (0.665, 0.77), (0.795, 0.77))
    fs.arrow(ax, (0.665, 0.23), (0.795, 0.23))
    ax.text(0.73, 0.15, "× patience", ha="center", va="top", fontsize=11)
    ax.text(0.89, 0.5, "first to finish", ha="center", va="center", fontsize=11)

    # (b) give-up time in seconds for this person's typical skipped question, two clocks
    ax = fig.add_subplot(grid[1, 0])
    fs.panel_label(ax, "b")
    skipped = att[att["skipped"]]
    mu = float(np.median(P.quit_drift(f0, k0, skipped["value"].to_numpy(), skipped["cost"].to_numpy())))
    secs = np.linspace(0.05, 35, 700)
    for pat, colour, ls in ((p_fit, fs.DARK, "-"), (P.PATIENCE, fs.LIGHT, "-")):
        dens = np.exp(P.log_density(secs / pat, mu)) / pat
        ax.plot(secs, dens, color=colour, ls=ls, label=f"{pat:.1f} s")
    top = ax.get_ylim()[1]
    for s in skipped["seconds"]:
        ax.plot([s, s], [0, -0.06 * top], color=fs.INK, lw=1.4, clip_on=False)
    ax.set_ylim(-0.08 * top, top)
    ax.spines["bottom"].set_position(("data", -0.08 * top))
    ax.set_xlim(0, 35)
    ax.set_xlabel("Seconds on question")
    ax.set_ylabel("Give-up density")
    ax.set_yticks([0, round(top * 0.9, 2)])
    ax.legend(title="Patience", loc="upper right", handlelength=1.2)

    # (c) -log L against patience
    ax = fig.add_subplot(grid[1, 1])
    fs.panel_label(ax, "c")
    ax.plot(pats, nll, color=fs.DARK)
    ax.plot(p_fit, P.neg_log_likelihood(f0, k0, att, p_fit), "o", color=fs.DARK, ms=9, mec=fs.INK)
    ax.plot(P.PATIENCE, P.neg_log_likelihood(f0, k0, att, P.PATIENCE), "o", color=fs.LIGHT, ms=9, mec=fs.INK)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xlabel("Patience (s)")
    ax.set_ylabel("$-$log L")
    fs.save(fig, OUT / "3_patience")

    # 4: where the minimum sits, decisions alone and decisions + times at a fixed 10 s
    fig, axes = fs.figure(1, 2, width=7.4, height=3.6)
    fig.subplots_adjust(wspace=0.35)
    dec = surface(lambda f, k: -sample_ll(decisions, f, k))
    times = np.array([[P.neg_log_likelihood(f, k, att, P.PATIENCE) for f in F_GRID] for k in K_FINE])
    _models.clear()
    levels = [0, 1.15, 3.1, 5.9]
    minima = []
    for nll in (dec, dec + times):
        i, j = np.unravel_index(np.argmin(nll), nll.shape)
        minima.append((F_GRID[j], K_FINE[i]))
    for ax, nll, letter, here in ((axes[0], dec, "a", 0), (axes[1], dec + times, "b", 1)):
        fs.panel_label(ax, letter)
        rel = nll - nll.min()
        ax.contourf(F_GRID, K_FINE, rel, levels=levels, colors=[fs.MID, fs.LIGHT, fs.BOX_FILL])
        ax.contour(F_GRID, K_FINE, rel, levels=levels[1:], colors=fs.INK, linewidths=1.0)
        ax.plot(*minima[here], "o", color=fs.DARK, mec=fs.INK, ms=10, zorder=5)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect("equal")
        ax.set_xlabel("$f$")
        ax.set_ylabel("$k$")
    axes[1].plot(*minima[0], "o", mfc="none", mec=fs.INK, mew=1.6, ms=10, zorder=5)
    fs.arrow(axes[1], (minima[0][0] - 0.08, minima[0][1]), (minima[1][0] + 0.1, minima[1][1]))
    axes[0].set_title("Decisions", fontsize=12)
    axes[1].set_title("+ answer and skip times, 10 s", fontsize=12)
    fs.save(fig, OUT / "4_joint_fit_pull")


def fig_clocks():
    fig, (a, b) = fs.figure(1, 2, width=7.4, height=3.2)
    fig.subplots_adjust(wspace=0.4)

    # (a) decisions: the deadline does not move with the person
    fs.panel_label(a, "a")
    for f, colour in zip(F_PEOPLE, fs.BLUES):
        sol = solve(f, K_TRUE, 0.5, 0.5)
        a.plot(sol.t_domain, sol.pdf("_top"), color=colour, label=f"{f}")
    a.axvline(A.T_DUR, color=fs.INK, lw=fs.AXIS, ls="--")
    a.text(A.T_DUR, a.get_ylim()[1], "$T_{DUR}$", ha="center", va="bottom", fontsize=11)
    a.set_xlim(0, 9)
    a.set_xlabel("Decision time (s)")
    a.set_ylabel("GO density")
    a.legend(title="$f$", loc="center", bbox_to_anchor=(0.6, 0.6), handlelength=1.2)

    # (b) giving up: patience stretches the clock, and there is no deadline
    fs.panel_label(b, "b")
    secs = np.linspace(0.05, 40, 800)
    mu = P.quit_drift(0.8, K_TRUE, *[float(x) for x in P.conditions(0.35, 4)])
    for pat, colour in zip((2.0, 10.0, 20.0), fs.BLUES):
        b.plot(secs, np.exp(P.log_density(secs / pat, mu)) / pat, color=colour, label=f"{pat:g} s")
    b.set_xlim(0, 40)
    b.set_xlabel("Seconds on question")
    b.set_ylabel("Give-up density")
    b.legend(title="Patience", loc="upper right", handlelength=1.2)
    fs.save(fig, OUT / "5_deadline_vs_patience")


# ---------------------------------------------------------------- drift planes

# A decision's (V, C) makes drift a plane over (f, k); its measured drift floods that
# plane to one height, and the shoreline is every person who fits. Two decisions with
# different V/C cross in one (f, k); two with the same V/C lie on one line.
PLANE_TRUE = (0.8, 0.8)
PLANE_DECISIONS = ((0.25, 0.30), (0.25, 0.75), (0.50, 0.60))  # (V, C): 1, 2, 3 (same mix as 1)
PLANE_FLOOR, PLANE_TOP = -1.0, 0.6


def _plane_axes(fig, pos, letter, title):
    ax = fig.add_subplot(2, 2, pos, projection="3d", computed_zorder=False)
    ax.view_init(elev=22, azim=-125)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(PLANE_FLOOR, PLANE_TOP)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticks([0.5, 1, 1.5, 2])  # 0 is shared with f at the near corner
    ax.set_zticks([-1.0, -0.6, -0.2, 0.2, 0.6])
    ax.set_xlabel("$f$", labelpad=2)
    ax.set_ylabel("$k$", labelpad=2)
    ax.tick_params(pad=0)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(fs.INK)
        axis.line.set_linewidth(1.0)
    ax.set_box_aspect((1, 1, 0.8), zoom=0.9)  # room for the legend under the title
    # z label beside its own ticks (matplotlib puts it on the far edge at this azimuth)
    ax.text2D(-0.07, 0.55, "Drift $\\mu$", transform=ax.transAxes, rotation=90, va="center")
    ax.text2D(-0.02, 0.98, f"({letter})", transform=ax.transAxes, fontsize=13, va="top")
    ax.text2D(0.5, 0.98, title, transform=ax.transAxes, fontsize=12, va="top", ha="center")
    return ax


def _shoreline(v, c, mu, n=2):
    f = np.linspace(0, 2, n)
    k = (v * f - mu) / c  # f·V − k·C = μ
    keep = (k >= 0) & (k <= 2)
    return f[keep], k[keep]


def _wall(ax, v, c, mu, color, alpha):
    f, k = _shoreline(v, c, mu)
    F = np.array([f, f])
    K = np.array([k, k])
    Z = np.array([[PLANE_FLOOR] * len(f), [PLANE_TOP] * len(f)])
    ax.plot_surface(F, K, Z, color=color, alpha=alpha, linewidth=0, shade=False)


def fig_drift_planes():
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    f_true, k_true = PLANE_TRUE
    mus = [A.initiation_drift(v, c, f_true, k_true) for v, c in PLANE_DECISIONS]
    (v1, c1), (v2, c2), (v3, c3) = PLANE_DECISIONS
    c_one, c_two = fs.DARK, fs.LIGHT
    roof_map = LinearSegmentedColormap.from_list("roof", [fs.DARK, fs.MID, fs.LIGHT])

    fg, kg = np.meshgrid(np.linspace(0, 2, 41), np.linspace(0, 2, 41))
    roof = A.initiation_drift(v1, c1, fg, kg)
    shade = roof_map((roof - roof.min()) / (roof.max() - roof.min()))

    fig = fs.plt.figure(figsize=(7.2, 7.0))

    # (a) one decision: drift over every (f, k)
    a = _plane_axes(fig, 1, "a", "$\\mu = 0.25f - 0.30k$")
    a.plot_surface(fg, kg, roof, facecolors=shade, alpha=0.75, linewidth=0, shade=False)
    a.scatter([f_true], [k_true], [mus[0]], color=fs.INK, s=36, depthshade=False, zorder=10)
    a.text(f_true, k_true, mus[0] + 0.14, "true person", fontsize=11, ha="center")

    # (b) the measured drift floods the plane; the shoreline is everyone who fits
    b = _plane_axes(fig, 2, "b", f"Measured $\\mu = {mus[0]:.2f}$")
    b.plot_surface(fg, kg, roof, facecolors=shade, alpha=0.45, linewidth=0, shade=False)
    b.plot_surface(fg, kg, np.full_like(fg, mus[0]), color=fs.BOX_FILL, alpha=0.45,
                   linewidth=0, shade=False)
    f, k = _shoreline(v1, c1, mus[0])
    b.plot(f, k, [mus[0]] * len(f), color=c_one, lw=3, zorder=10)
    b.plot(f, k, [PLANE_FLOOR] * len(f), color=c_one, lw=1.6, ls="--")
    b.scatter([f_true], [k_true], [mus[0]], color=fs.INK, s=36, depthshade=False, zorder=11)
    b.legend(handles=[Line2D([], [], color=c_one, lw=3, label="shoreline"),
                      Line2D([], [], color=c_one, lw=1.6, ls="--", label="on the floor")],
             loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, columnspacing=1.0, fontsize=11, handlelength=1.6)

    # (c) two mixes: two walls, one crossing
    cc = _plane_axes(fig, 3, "c", "Decisions 1 and 2")
    for (v, c), mu, col in (((v1, c1), mus[0], c_one), ((v2, c2), mus[1], c_two)):
        _wall(cc, v, c, mu, col, 0.22)
        f, k = _shoreline(v, c, mu)
        cc.plot(f, k, [mu] * len(f), color=col, lw=2.5)
        cc.plot(f, k, [PLANE_FLOOR] * len(f), color=col, lw=1.4, ls="--")
    cc.plot([f_true] * 2, [k_true] * 2, [PLANE_FLOOR, PLANE_TOP], color=fs.INK, lw=2.5)
    cc.legend(handles=[Line2D([], [], color=c_one, lw=2.5, label=f"1: V/C = {v1 / c1:.2f}"),
                       Line2D([], [], color=c_two, lw=2.5, label=f"2: V/C = {v2 / c2:.2f}"),
                       Line2D([], [], color=fs.INK, lw=2.5, label="$f = k = 0.8$")],
              loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, columnspacing=0.8,
              handlelength=1.1, handletextpad=0.4, fontsize=11)

    # (d) the same mix twice: one wall, no crossing
    d = _plane_axes(fig, 4, "d", "Decisions 1 and 3")
    _wall(d, v1, c1, mus[0], c_one, 0.18)
    _wall(d, v3, c3, mus[2], c_two, 0.18)
    f, k = _shoreline(v1, c1, mus[0])
    d.plot(f, k, [PLANE_FLOOR] * len(f), color=c_one, lw=4.5)
    f, k = _shoreline(v3, c3, mus[2])
    d.plot(f, k, [PLANE_FLOOR] * len(f), color=c_two, lw=1.8)
    d.scatter([f_true], [k_true], [PLANE_FLOOR], color=fs.INK, s=36, depthshade=False, zorder=10)
    d.legend(handles=[Line2D([], [], color=c_one, lw=4.5, label=f"1: V/C = {v1 / c1:.2f}"),
                      Line2D([], [], color=c_two, lw=1.8, label=f"3: V/C = {v3 / c3:.2f}")],
             loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, columnspacing=1.0, fontsize=11, handlelength=1.6)

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.97, wspace=0.08, hspace=0.02)
    fs.save(fig, OUT / "6_drift_planes")


FIGURES = {"recovery": fig_recovery, "identifiability": fig_identifiability,
           "patience": fig_patience, "clocks": fig_clocks, "planes": fig_drift_planes}


def draw(*names):
    fs.apply()
    for name in names or FIGURES:
        if name == "patience" and not DATA.exists():
            print(f"skipped patience and joint fit pull: no record at {DATA}")
            continue
        FIGURES[name]()


# The paper's copies: Figure 2 is the drift planes, Figure 3 the patience figure.
def fig2():
    draw("planes")
    fs.copy(OUT / "6_drift_planes", PAPER_OUT / "fig2_drift_planes")


def fig3():
    if not DATA.exists():
        print(f"skipped Figure 3: no record at {DATA} (set TEMPORAL_RECORD)")
        return
    draw("patience")
    fs.copy(OUT / "3_patience", PAPER_OUT / "fig3_patience")
