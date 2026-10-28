"""Recovery of the learning rate α with f, k, γ, BOUND and T0, from the data the app records.

γ is fitted from shadow TD learners, one per γ, run beside the person at α 0.05
(experiments.recovery_lhc). A second axis of shadows, one per (γ, α), would multiply
what each history stores, so this experiment logs the reward stream once and replays it
at fit time:

    the log      every TD step (slot, model reward), every set's end and whether it was
                 abandoned, and every recorded decision's place among them (EventLog)
    the replay   the TD learner of temporal.model.value run on that log at every (γ, α)
                 of a grid at once, giving each decision's value at each grid point in
                 two parts, as GridShadow's parts_by_g: V(0) or the slot's share, and the
                 carried error before its clip at ±1 (an abandoned set carries 0)
    the fit      both parts as tensor-product cubic splines over (γ, log α), the error
                 clipped after interpolating (Values): the clip puts kinks in value that
                 a spline through clipped values misses by up to 0.008, while the
                 unclipped parts are smooth and interpolate to about 1e-5; then
                 recovery_lhc's likelihood, optimizer, Hessian and Fisher information
                 with log α as a sixth parameter

People are a Latin hypercube over recovery_lhc's five ranges and log α:

    f      0.2 to 1.5       k      0.2 to 1.5       γ    0.6 to 0.97
    BOUND  1.0 to 2.0       T0     0 to 0.4 s       α    0.01 to 0.2, uniform in log α

N_PEOPLE of them from a seeded hypercube (SEED, not recovery_lhc's), so every machine
draws the same people and `--people` only says which to run. The task is recovery_lhc's:
the app's task, InfoController planning with γ 0.9 and α 0.05 as the app does, 300 start
decisions plus their redo decisions. SIGMA 0.7, ACHIEVEMENT_SCALE 4, continuing
bootstrap, T_DUR 7.0. As there, the person's draws used value rounded to 0.05 and the
fit uses exact value.

Three fits of the same data:
    a  all six free: f, k, γ, log α, BOUND, T0
    b  α pinned at 0.05, the other five free (recovery_lhc's method)
    c  the current app method: γ 0.9, BOUND 1.5, α 0.05 pinned; f, k, T0 free

Claims, fixed before any simulation (2026-10-28):
    A1  RMSE, bias and r for each of the six parameters of fit a, α on the log scale and
        as α; no threshold, the numbers are the result
    A2  calibration with SE from the Fisher information at the truth: over fits with no
        parameter within two finite-difference steps of its bound (those are excluded and
        counted), the standardized errors have SD within 0.8 to 1.25 and the truth is
        inside ±1.96 SE in 90% to 98% of fits, for every parameter
    A3  what fitting α buys: the slope of f's relative error, (f̂ − f) / f, on the true
        log α is within 2 SE of zero under fit a, and positive beyond 2 SE under fit b
        (a person who learns faster than 0.05 is read as more reward sensitive); also
        reported, no threshold: f's relative bias by tercile of true α under each fit
    A4  none: the median correlations of log α with γ and with f from the inverse Fisher
        information, and the median Fisher SE of log α. If that SE is log 2 (0.69) or
        wider, α is identified too weakly to fit from one person's app data, and it stays
        assumed, with gamma_robustness's numbers as the cost

    python -m experiments.recovery_alpha check                 # validation, one person
    python -m experiments.recovery_alpha simulate --people 0-49 --tag server
    python -m experiments.recovery_alpha fit
    python -m experiments.recovery_alpha --replot
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from temporal import fastlik
from temporal.inference import T0_BOUNDS
from temporal.model import accumulator as acc
from temporal.model.conditions import DELTA_CLIP
from temporal.model.value import CONTINUING

from .common import Quiet, out_dir, run_pool, se
from .gamma_profile import APP_G, GridShadow, run_app
from .misspecification import PersonConstants
from .recovery_lhc import G_FINE, span, standard_errors

OUT = out_dir("recovery_alpha")
HIST = out_dir("recovery_alpha/histories")
N_PEOPLE = 100
RANGES = {"f": (0.2, 1.5), "k": (0.2, 1.5), "g": (0.6, 0.97), "log_a": (np.log(0.01), np.log(0.2)),
          "bound": (1.0, 2.0), "t0": (0.0, 0.4)}
SEED = 20261030
APP_A = 0.05  # the app's learning rate, and the shadows'
# log α nodes: 0.05 · 10^(j/12), j = -12 to 10, so 0.005 to about 0.34 with 0.05 a node.
A_GRID = tuple(APP_A * 10 ** (j / 12) for j in range(-12, 11))
STEP, SET_END, SET_ABANDONED, START, REDO = range(5)  # event kinds


class EventLog(GridShadow):
    """GridShadow that also logs what a replay needs: each TD step's slot and the reward the
    shadows were paid, each set's end (and whether it was abandoned), and each recorded
    decision, whose record gets `event`, its index in the log. Logging draws nothing, so
    the person, the shadows and the controller behave exactly as under GridShadow."""

    def __init__(self, params, grid, **kw):
        super().__init__(params, grid, **kw)
        self.log = []  # (kind, slot, reward)

    def step_stage(self, s, reward=0.0):
        delta = super().step_stage(s, reward)
        self.log.append((STEP, s, self.model_reward))
        return delta

    def run_episode(self):
        abandoned = self.state.abandoned
        out = super().run_episode()
        self.log.append((SET_ABANDONED if self.state.abandoned > abandoned else SET_END, 0, 0.0))
        return out

    def initiate(self):
        self.log.append((START, 0, 0.0))
        out = super().initiate()
        self.state.initiation_passages[-1]["event"] = len(self.log) - 1
        return out

    def _redoes(self, s, item):
        self.log.append((REDO, s, 0.0))
        out = super()._redoes(s, item)
        self.state.first_passage[-1]["event"] = len(self.log) - 1
        return out

    def arrays(self):
        kind, slot, reward = zip(*self.log)
        return (np.array(kind, dtype=np.int8), np.array(slot, dtype=np.int8),
                np.array(reward, dtype=float))


# Every decision's value, at every (γ, α) of the grids: the TD learner of
# temporal.model.value (features [1, s/3], TD(0), continuing bootstrap) replayed on the
# log, vectorized over the grid. Returns (decisions, γ, α, 2) in the order of `events`,
# the log indices of the decisions: V(0) or the slot's share, and the carried error
# before its clip. Value is the first plus the clipped second (value_of).
def replay(kind, slot, reward, events, gammas, alphas, n_slots=4):
    g = np.asarray(gammas, float)[:, None]
    a = np.asarray(alphas, float)[None, :]
    t0 = np.zeros((len(gammas), len(alphas)))  # theta, intercept and slope
    t1 = np.zeros_like(t0)
    slot_err = np.zeros_like(t0)
    set_sum = np.zeros_like(t0)
    set_err = np.zeros_like(t0)
    at = {e: i for i, e in enumerate(events)}
    out = np.empty((len(events), len(gammas), len(alphas), 2))
    last = n_slots - 1
    for e, (kd, s, r) in enumerate(zip(kind, slot, reward)):
        if kd == STEP:
            x = s / last
            v_s = t0 + t1 * x
            v_next = t0 + t1 * ((s + 1) / last) if s < last else (t0 if CONTINUING else 0.0)
            delta = r + g * v_next - v_s
            step = a * delta
            t0 = t0 + step
            t1 = t1 + step * x
            slot_err = delta
            set_sum = set_sum + delta
        elif kd == SET_END:
            set_err, set_sum = set_sum, np.zeros_like(t0)
        elif kd == SET_ABANDONED:
            set_err, set_sum = np.zeros_like(t0), np.zeros_like(t0)
        elif e in at:
            if kd == START:
                out[at[e], ..., 0], out[at[e], ..., 1] = t0, set_err
            else:
                out[at[e], ..., 0] = (1.0 / n_slots) * g ** (last - s)
                out[at[e], ..., 1] = slot_err
    return out


def value_of(parts):
    return parts[..., 0] + np.clip(parts[..., 1], -DELTA_CLIP, DELTA_CLIP)


class Values:
    """Value as a function of (γ, log α): each part a cubic spline in γ on G_FINE, then one
    in log α on A_GRID, through every replayed node; the error is clipped after."""

    def __init__(self, parts, gammas=G_FINE, alphas=A_GRID):
        n, G, A, _ = parts.shape
        self.shape = (A, n, 2)
        self.log_a = np.log(alphas)
        self.in_g = CubicSpline(gammas, parts.transpose(1, 2, 0, 3).reshape(G, -1), axis=0)

    def __call__(self, g, log_a):
        at_g = self.in_g(g).reshape(self.shape)
        return value_of(CubicSpline(self.log_a, at_g, axis=0)(log_a))


# ---------------------------------------------------------------- stage 1: simulate

def people():
    unit = qmc.LatinHypercube(d=len(RANGES), seed=SEED).random(N_PEOPLE)
    lo = np.array([r[0] for r in RANGES.values()])
    hi = np.array([r[1] for r in RANGES.values()])
    return [dict(zip(RANGES, row)) for row in lo + unit * (hi - lo)]


def history_path(i):
    return HIST / f"p{i:03d}.npz"


# One person's history: what the app records, and the event log.
def play(truth, seed, n=300, grid=(APP_G,)):
    sims = []
    data = run_app(truth["f"], truth["k"], truth["g"], seed=seed, n=n, a=float(np.exp(truth["log_a"])),
                   person=PersonConstants(acc.SIGMA, truth["bound"]), grid=grid, t0=truth["t0"],
                   cls=EventLog, sim_out=sims)
    return data, sims[0]


def save_history(path, data, sim, truth):
    kind, slot, reward = sim.arrays()
    np.savez_compressed(
        path, kind=kind, slot=slot, reward=reward,
        event=np.array([p["event"] for p in data]),
        node=np.array([p.get("node") == "stage" for p in data], dtype=np.int8),
        go=np.array([-1 if p["GO"] is None else p["GO"] for p in data], dtype=np.int8),
        rt=np.array([np.nan if p["RT"] is None else p["RT"] for p in data]),
        cost=np.array([p["cost"] for p in data]),
        value_rec=np.array([p["value"] for p in data]),
        feedback=np.array([p.get("feedback", 1) for p in data], dtype=np.int8),
        **{f"true_{k}": v for k, v in truth.items()})


def _simulate(i):
    truth = people()[i]
    acc.clear_caches()
    with Quiet():
        data, sim = play(truth, seed=SEED + i)
    save_history(history_path(i), data, sim, truth)
    return dict(person=i, decisions=len(data), redos=sum(p.get("node") == "stage" for p in data),
                steps=int((sim.arrays()[0] == STEP).sum()), **truth)


# ---------------------------------------------------------------- stage 2: fit

NAMES = ("f", "k", "g", "log_a", "bound", "t0")
BOUNDS = [(0.0, 3.0), (0.0, 3.0), (0.5, 0.99), (np.log(0.005), np.log(0.3)), (0.5, 3.0), T0_BOUNDS]
STEPS = np.array([0.01, 0.01, 0.005, 0.02, 0.01, 0.005])  # finite-difference steps
FREE = {"a": (0, 1, 2, 3, 4, 5), "b": (0, 1, 2, 4, 5), "c": (0, 1, 5)}
PINNED = {"b": {3: np.log(APP_A)}, "c": {2: APP_G, 3: np.log(APP_A), 4: acc.BOUND}}


class Person:
    """One saved history: its decisions, and value as a smooth function of (γ, log α)."""

    def __init__(self, i, path=None):
        h = np.load(path or history_path(i))
        self.i = i
        self.go, self.rt, self.cost, self.node = h["go"], h["rt"], h["cost"], h["node"]
        self.value = Values(replay(h["kind"], h["slot"], h["reward"], h["event"], G_FINE, A_GRID))
        self.truth = np.array([float(h[f"true_{n}"]) for n in NAMES])

    def mu(self, th):
        return th[0] * self.value(th[2], th[3]) - th[1] * self.cost

    def nll(self, th):
        return fastlik.neg_log_likelihood(self.go, self.rt, self.mu(th), th[4], th[5])

    def fit(self, free=FREE["a"], fixed=None):
        th = np.array(fixed if fixed is not None else self.truth, float)
        free = list(free)

        def loss(x):
            th[free] = x
            return self.nll(th)
        res = differential_evolution(loss, [BOUNDS[j] for j in free], seed=self.i, tol=1e-8,
                                     maxiter=2000, polish=False)
        res = minimize(loss, res.x, method="L-BFGS-B", bounds=[BOUNDS[j] for j in free])
        th[free] = res.x
        return th.copy(), float(res.fun)

    def hessian(self, th, steps=STEPS):
        n = len(th)
        H = np.zeros((n, n))
        f0 = self.nll(th)
        e = np.eye(n) * steps
        for i in range(n):
            H[i, i] = (self.nll(th + e[i]) - 2 * f0 + self.nll(th - e[i])) / steps[i] ** 2
            for j in range(i):
                H[i, j] = H[j, i] = (self.nll(th + e[i] + e[j]) - self.nll(th + e[i] - e[j])
                                     - self.nll(th - e[i] + e[j]) + self.nll(th - e[i] - e[j])
                                     ) / (4 * steps[i] * steps[j])
        return H

    def fisher(self, th, steps=STEPS):
        def probs(x):
            up, lo, und = fastlik.outcome_probs(self.mu(x), x[4], x[5])
            return np.concatenate([up, lo, und[:, None]], axis=1)
        w = 1.0 / np.maximum(probs(th), 1e-300)
        e = np.eye(len(th)) * steps
        G = np.stack([(probs(th + e[i]) - probs(th - e[i])) / (2 * steps[i])
                      for i in range(len(th))])
        return np.einsum("ino,jno,no->ij", G, G, w)


def correlations(M):
    try:
        cov = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.full((len(M), len(M)), np.nan)
    d = np.sqrt(np.abs(np.diag(cov)))
    return cov / np.outer(d, d)


def _fit(i):
    p = Person(i)
    row = dict(person=i, decisions=len(p.go), redos=int(p.node.sum()), undecided=int((p.go < 0).sum()))
    for name, free in FREE.items():
        start = p.truth.copy()
        for j, v in PINNED.get(name, {}).items():
            start[j] = v
        est, nll = p.fit(free=free, fixed=start)
        row[f"nll_{name}"] = nll
        for j, n in enumerate(NAMES):
            row[f"{name}_{n}"] = est[j]
        if name == "a":
            se_obs, _ = standard_errors(p.hessian(est))
            M = p.fisher(p.truth)
            se_fis, _ = standard_errors(M)
            C = correlations(M)
            row.update(corr_a_g=C[3, 2], corr_a_f=C[3, 0], corr_g_f=C[2, 0])
            for j, n in enumerate(NAMES):
                lo, hi = BOUNDS[j]
                row.update({f"true_{n}": p.truth[j], f"se_obs_{n}": se_obs[j], f"se_fisher_{n}": se_fis[j],
                            f"at_bound_{n}": bool(min(est[j] - lo, hi - est[j]) < 2 * STEPS[j])})
    return row


def slope(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = resid @ resid / (len(y) - 2) * np.linalg.inv(X.T @ X)
    return beta[1], np.sqrt(cov[1, 1])


def summarize(df):
    print(f"\n== recovery of f, k, γ, α, BOUND, T0: {len(df)} people, "
          f"{df.decisions.mean():.0f} decisions each ({df.redos.mean():.0f} redos) ==")
    ok = ~df[[f"at_bound_{m}" for m in NAMES]].any(axis=1)
    print(f"fits with a parameter at its bound (excluded from calibration): {(~ok).sum()}")
    print(f"{'':>6} {'RMSE':>7} {'bias':>16} {'r':>6}   {'SD z obs':>8} {'cover obs':>9} "
          f"{'SD z Fisher':>11} {'cover Fisher':>12} {'n':>4}")
    a2 = True
    for n in NAMES:
        err = df[f"a_{n}"] - df[f"true_{n}"]
        r = np.corrcoef(df[f"a_{n}"], df[f"true_{n}"])[0, 1]
        use = ok & df[f"se_fisher_{n}"].notna()
        z_obs = (err / df[f"se_obs_{n}"])[use]
        z_fis = (err / df[f"se_fisher_{n}"])[use]
        a2 &= 0.8 <= z_fis.std() <= 1.25 and 0.90 <= (z_fis.abs() < 1.96).mean() <= 0.98
        print(f"{n:>6} {np.sqrt(np.mean(err ** 2)):7.3f} {err.mean():+8.3f} ± {se(err):.3f} {r:6.3f}   "
              f"{z_obs.std():8.2f} {(z_obs.abs() < 1.96).mean():9.0%} "
              f"{z_fis.std():11.2f} {(z_fis.abs() < 1.96).mean():12.0%} {use.sum():4d}")
    a_err = np.exp(df.a_log_a) - np.exp(df.true_log_a)
    print(f"{'α':>6} {np.sqrt(np.mean(a_err ** 2)):7.3f} {a_err.mean():+8.3f} ± {se(a_err):.3f} "
          f"{np.corrcoef(np.exp(df.a_log_a), np.exp(df.true_log_a))[0, 1]:6.3f}")
    print(f"A2 (Fisher calibration, every parameter): {'pass' if a2 else 'fail'}")

    print("\nA3: f's relative error (f̂ − f) / f against the true log α")
    third = pd.qcut(df.true_log_a, 3, labels=["low α", "mid α", "high α"])
    verdict = {}
    for name, label in (("a", "a, α free"), ("b", "b, α 0.05"), ("c", "c, app")):
        rel = (df[f"{name}_f"] - df.true_f) / df.true_f
        b, b_se = slope(df.true_log_a.to_numpy(), rel.to_numpy())
        verdict[name] = (b, b_se)
        terc = "  ".join(f"{t} {rel[third == t].mean():+.1%} ± {se(rel[third == t]):.1%}"
                         for t in third.cat.categories)
        print(f"  {label:>11}: slope {b:+.3f} ± {b_se:.3f} per unit log α   {terc}")
    (ba, sa), (bb, sb) = verdict["a"], verdict["b"]
    print(f"A3: {'pass' if abs(ba) <= 2 * sa and bb > 2 * sb else 'fail'} "
          f"(fit a slope within 2 SE of 0: {abs(ba) <= 2 * sa}; fit b slope > 2 SE: {bb > 2 * sb})")

    print("\nA4 (no claim): from the inverse Fisher information at the truth")
    for col, name in (("corr_a_g", "corr(log α, γ)"), ("corr_a_f", "corr(log α, f)"),
                      ("corr_g_f", "corr(γ, f)")):
        c = df[col].dropna()
        print(f"  {name}: median {c.median():+.2f}, 5th to 95th percentile "
              f"{c.quantile(0.05):+.2f} to {c.quantile(0.95):+.2f}")
    s = df.se_fisher_log_a.dropna()
    print(f"  Fisher SE of log α: median {s.median():.2f} (a factor of {np.exp(s.median()):.2f} on α); "
          f"{'weakly identified (≥ log 2)' if s.median() >= np.log(2) else 'below log 2'}")

    print("\nf, k, γ under each fit (RMSE, bias ± SE):")
    for n in ("f", "k", "g"):
        cells = []
        for name in FREE:
            e = df[f"{name}_{n}"] - df[f"true_{n}"]
            cells.append(f"{name}: {np.sqrt(np.mean(e ** 2)):.3f}, {e.mean():+.3f} ± {se(e):.3f}")
        print(f"  {n:>2}  " + "   ".join(cells))


# ---------------------------------------------------------------- validation

# Before any full run, on one simulated person (people()[0], 40 starts):
#   1  the replay at α 0.05 equals the shadows' parts_by_g on G_FINE
#   2  off-grid, the interpolated value equals a direct replay there
#   3  a smoke fit of all six parameters runs
def check(n=40):
    truth = people()[0]
    t = time.time()
    with Quiet():
        data, sim = play(truth, seed=SEED, n=n, grid=G_FINE)
    print(f"simulated {len(data)} decisions in {time.time() - t:.0f} s")
    kind, slot, reward = sim.arrays()
    events = [p["event"] for p in data]
    t = time.time()
    parts = replay(kind, slot, reward, events, G_FINE, A_GRID)
    print(f"replay of {int((kind == STEP).sum())} steps over {len(G_FINE)} × {len(A_GRID)} "
          f"(γ, α) in {time.time() - t:.2f} s")
    j = A_GRID.index(APP_A)
    shadow = np.array([[sum(p["parts_by_g"][g]) for g in G_FINE] for p in data])
    print(f"1  max |replay − shadows| at α 0.05: {np.abs(value_of(parts[:, :, j]) - shadow).max():.2e}")

    surface = Values(parts)
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0.5, 0.99, 200), rng.uniform(np.log(0.005), np.log(0.3), 200)])
    err = [np.abs(surface(g, la) - value_of(replay(kind, slot, reward, events, [g], [np.exp(la)])[:, 0, 0])).max()
           for g, la in pts]
    print(f"2  interpolation error at 200 off-grid (γ, α): max {max(err):.2e}, "
          f"median {np.median(err):.2e} (value is rounded to 0.05 steps)")

    path = OUT / "check.npz"
    save_history(path, data, sim, truth)
    p = Person(0, path)
    t = time.time()
    est, nll = p.fit()
    print(f"3  smoke fit in {time.time() - t:.0f} s, -log L {nll:.3f}")
    for n, tr, e in zip(NAMES, p.truth, est):
        print(f"     {n:>6} true {tr:+.3f}  fitted {e:+.3f}")
    path.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("check", "simulate", "fit"), default=None)
    ap.add_argument("--replot", action="store_true", help="summarize the saved fits")
    ap.add_argument("--people", type=span, default=list(range(N_PEOPLE)), help="for example 0-49")
    ap.add_argument("--tag", default="local", help="names this machine's file")
    a = ap.parse_args()
    if a.stage == "check":
        check()
    elif a.stage == "simulate":
        todo = [i for i in a.people if not history_path(i).exists()]
        run_pool(_simulate, todo, OUT / f"simulated_{a.tag}.csv", fresh=True,
                 label=lambda r: f"person {r['person']}: {r['decisions']} decisions")
    elif a.stage == "fit":
        todo = [i for i in a.people if history_path(i).exists()]
        df = run_pool(_fit, todo, OUT / "fits.csv", label=lambda r: f"person {r['person']}")
        df.sort_values("person").to_csv(OUT / "fits.csv", index=False)
    if a.stage == "fit" or a.replot:
        summarize(pd.read_csv(OUT / "fits.csv"))
