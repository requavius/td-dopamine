"""The task in a desktop window: python -m temporal.app.desktop

You answer questions, choose whether to redo a miss, and whether to start the next set.
The record is kept in data/ between sessions.
"""
import signal
import threading
import time
import tkinter as tk
from tkinter import ttk

import pandas as pd

from ..model.persistence import attempts_to_frame
from .live import Paths, model_view
from .session import MIN_FIT_ROWS, Session, fit_summary

BIG = ("Helvetica", 26)
MED = ("Helvetica", 15)
SMALL = ("Helvetica", 11)
MONO = ("Menlo", 11)


class DesktopApp(Session):
    def __init__(self, root, **kw):
        root.title("initiation model -- human version")
        root.geometry("880x800")
        root.minsize(880, 800)
        self.live = None  # the live model window, when open
        super().__init__(root, **kw)

    def make_var(self, value):
        kind = {bool: tk.BooleanVar, int: tk.IntVar, float: tk.DoubleVar, str: tk.StringVar}
        return kind[type(value)](value=value)

    # ------------------------------------------------------------ layout

    def build(self):
        self.left = ttk.Frame(self.root, padding=24)
        self.left.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.left.columnconfigure(0, weight=1)
        self.left.rowconfigure(9, weight=1)  # keeps the task area from jumping

        right = ttk.Frame(self.root, padding=16)
        right.grid(row=0, column=1, sticky="ns")
        self._build_settings(right)
        ttk.Separator(right).pack(fill="x", pady=10)
        self._build_panel(right)
        self.refresh_stats()

    def _build_settings(self, right):
        ttk.Label(right, text="settings (live)", font=MED).pack(anchor="w")
        ttk.Label(right, text="target chance of getting it right",
                  font=SMALL).pack(anchor="w", pady=(10, 0))
        ttk.Scale(right, from_=0.3, to=0.95, variable=self.v_target, length=200).pack()
        ttk.Label(right, text="stages per set", font=SMALL).pack(anchor="w", pady=(8, 0))
        ttk.Spinbox(right, from_=2, to=8, textvariable=self.v_stages, width=6).pack(anchor="w")
        ttk.Label(right, text="cold-start cost", font=SMALL).pack(anchor="w", pady=(8, 0))
        ttk.Scale(right, from_=0.05, to=1.5, variable=self.v_cold, length=200).pack()
        ttk.Label(right, text="review rate (owed questions return)",
                  font=SMALL).pack(anchor="w", pady=(8, 0))
        ttk.Scale(right, from_=0.0, to=1.0, variable=self.v_review, length=200).pack()
        ttk.Checkbutton(right, text="skipped questions come back too",
                        variable=self.v_return_skipped).pack(anchor="w")
        ttk.Checkbutton(right, text="controller sets target, review, feedback, skipped, redo prompts",
                        variable=self.v_controller).pack(anchor="w", pady=(6, 0))
        for text, var in [
            ("show stage count", self.v_show_count),
            ("show score so far", self.v_show_score),
            ("feedback (from next set)", self.v_feedback),
            ("offer repeat-or-continue", self.v_show_choice),
        ]:
            ttk.Checkbutton(right, text=text, variable=var).pack(anchor="w", pady=1)

        ttk.Label(right, text="when not offered, it:", font=SMALL).pack(anchor="w", pady=(6, 0))
        forced = ttk.Frame(right)
        forced.pack(anchor="w")
        ttk.Radiobutton(forced, text="continues", value="continue",
                        variable=self.v_forced).pack(side="left")
        ttk.Radiobutton(forced, text="repeats", value="repeat",
                        variable=self.v_forced).pack(side="left")

    def _build_panel(self, right):
        ttk.Label(right, text="behind the scenes", font=MED).pack(anchor="w")
        self.stats = ttk.Label(right, text="", font=MONO, justify="left")
        self.stats.pack(anchor="w", pady=(6, 0))
        self.now_label = ttk.Label(right, text="", font=SMALL, justify="left", wraplength=280)
        self.now_label.pack(anchor="w", pady=(4, 0))
        self.ctrl_label = ttk.Label(right, text="", font=SMALL, justify="left", wraplength=280)
        self.ctrl_label.pack(anchor="w", pady=(4, 0))

        ttk.Separator(right).pack(fill="x", pady=10)
        self.fit_label = ttk.Label(right, text=self.fit_result, font=MONO, justify="left")
        self.fit_label.pack(anchor="w")
        self.fit_button = ttk.Button(right, text="fit my parameters", command=self.run_fit)
        self.fit_button.pack(anchor="w", pady=6)
        ttk.Button(right, text="export all decisions to csv", command=self.export).pack(anchor="w")
        ttk.Button(right, text="live model view", command=self.open_live).pack(anchor="w", pady=6)

    def refresh_stats(self):
        self.stats.config(text=self.stats_text())
        self.now_label.config(text=self.now_text())
        self.ctrl_label.config(text=f"controller: {self.controller_status}")

    # ------------------------------------------------------------ screens

    def clear(self):
        if self._timer:
            self.root.after_cancel(self._timer)
            self._timer = None
        for key in ("1", "2"):
            self.root.unbind(key)
        for w in self.left.winfo_children():
            w.destroy()

    def ask_decision(self, prompt, yes_label, no_label, on_done):
        self.clear()
        ttk.Label(self.left, text=prompt, font=BIG).grid(row=0, column=0, pady=(60, 20))
        row = ttk.Frame(self.left)
        row.grid(row=1, column=0, pady=20)

        start = time.perf_counter()
        self.prompt_at = start
        done = []

        def finish(go):
            if done:
                return
            done.append(True)
            on_done(go, time.perf_counter() - start)
            if self.live and not self.closed:
                self.live.draw()

        ttk.Button(row, text=f"1  {yes_label}", command=lambda: finish(1)).pack(side="left", padx=8)
        ttk.Button(row, text=f"2  {no_label}", command=lambda: finish(0)).pack(side="left", padx=8)
        self.root.bind("1", lambda e: finish(1))
        self.root.bind("2", lambda e: finish(0))
        if self.live:
            self.live.follow(start)

    def render_problem(self, text, header):
        ttk.Label(self.left, text="   ".join(header), font=SMALL).grid(row=0, column=0, pady=(20, 0))
        ttk.Label(self.left, text=f"{text} = ?", font=BIG).grid(row=1, column=0, pady=30)
        entry = ttk.Entry(self.left, font=BIG, width=10, justify="center")
        entry.grid(row=2, column=0)
        entry.focus_set()
        entry.bind("<Return>", lambda e: self.on_answer(entry.get()))
        entry.bind("<Escape>", lambda e: self.on_skip())
        ttk.Label(self.left, text="type the answer and press enter -- or give up on it",
                  font=SMALL).grid(row=3, column=0, pady=12)
        ttk.Button(self.left, text="skip  (esc)", command=self.on_skip).grid(row=4, column=0)
        self.refresh_stats()

    def flash(self, text, then):
        self.clear()
        ttk.Label(self.left, text=text, font=BIG).grid(row=0, column=0, pady=90)
        self._timer = self.root.after(600, then)

    # ------------------------------------------------------------ panel buttons

    def run_fit(self):
        df = self.frame()
        if len(df) < MIN_FIT_ROWS:
            self.fit_label.config(text=f"only {len(df)} decided rows\nkeep going")
            return

        attempts = attempts_to_frame(self.state.persistence)
        self.fit_button.config(state="disabled")
        self.fit_label.config(text=f"fitting {len(df)} rows, {len(attempts)} questions...")

        def work():
            try:
                text = fit_summary(df, attempts)
            except Exception as exc:
                text = f"fit failed: {exc}"

            def show():
                self.fit_label.config(text=text)
                self.fit_button.config(state="normal")
            self.root.after(0, show)

        threading.Thread(target=work, daemon=True).start()

    def open_live(self):
        if self.live:
            self.live.top.lift()
            return
        self.live = LiveView(self)
        if self.screen in ("start", "redo"):
            self.live.follow(self.prompt_at)
        else:
            self.live.draw()

    def export(self):
        events = self.store.events()
        path = self.store.root / "decisions.csv"
        pd.json_normalize(events).to_csv(path, index=False)
        self.fit_label.config(text=f"wrote data/{path.name}\n{len(events)} events")


# The accumulator for the decision on screen, in its own window: bounds, mean drift,
# predicted RT densities, and sample paths run on the person's clock.
class LiveView:
    W, H = 640, 400
    L, R, STRIP = 46, 14, 56
    GO, NO, INK, MUTED, CLOCK = "#2e7d4f", "#b23a3a", "#1d1d1b", "#9b9a93", "#2f5d8a"

    def __init__(self, app):
        self.app = app
        self.top = tk.Toplevel(app.root)
        self.top.title("the accumulator, live")
        self.text = ttk.Label(self.top, font=MONO, justify="left", padding=(10, 8))
        self.text.pack(anchor="w")
        self.canvas = tk.Canvas(self.top, width=self.W, height=self.H, bg="white",
                                highlightthickness=0)
        self.canvas.pack(padx=10, pady=(0, 10))
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.decision = self.view = self.paths = None
        self.started = None
        self._timer = None

    def close(self):
        if self._timer:
            self.app.root.after_cancel(self._timer)
        self.top.destroy()
        self.app.live = None

    # A decision has just been put on screen at `start` (perf_counter): run its paths.
    def follow(self, start):
        if self._timer:
            self.app.root.after_cancel(self._timer)
        self.started = start
        self._new(self.app.decision_now)
        self.tick()

    def _new(self, decision):
        self.decision = decision
        self.view = model_view(self.app, decision)
        self.paths = Paths(self.view)

    def tick(self):
        self._timer = None
        self.draw()
        now = time.perf_counter() - self.started
        if self.app.decision_now.get("go") is None and now < self.view["t_dur"]:
            self._timer = self.app.root.after(33, self.tick)

    def draw(self):
        d = self.app.decision_now
        if d is None:
            self.text.config(text="no decision yet")
            return
        if d is not self.decision:
            self.started = None
            self._new(d)
        v = {**self.view, "go": d["go"], "rt": d["rt"]}
        now = v["rt"] if v["go"] is not None else (
            time.perf_counter() - self.started if self.started else 0.0)
        now = min(now, v["t_dur"])
        self.paths.step_to(now)
        self._text(v)
        self._plot(v, now)

    def _text(self, v):
        pct = lambda x: f"{x:.0%}"  # noqa: E731
        label = "start the next set" if v["node"] == "start" else "redo the miss"
        you = ("deciding..." if v["go"] is None else
               f"you: {v['yes'] if v['go'] else v['no']} at {v['rt']:.2f}s"
               + (" (past the deadline: undecided to the fit)" if v["rt"] > v["t_dur"] else ""))
        self.text.config(text="\n".join([
            f"{label}:  value {v['value']:.2f}  cost {v['cost']:.2f}",
            f"f {v['f']:.2f}  k {v['k']:.2f}  T0 {v['t0']:.2f}s  ->  drift {v['drift']:.2f}",
            f"predicted: {v['yes']} {pct(v['p_go'])}  {v['no']} {pct(v['p_no'])}  "
            f"undecided by {v['t_dur']:.0f}s {pct(v['p_undecided'])}",
            you, v["note"]]))

    def _plot(self, v, now):
        c = self.canvas
        c.delete("all")
        W, H, L, R, strip = self.W, self.H, self.L, self.R, self.STRIP
        top, bot = 6 + strip, H - 22 - strip
        b, T = v["bound"], v["t_dur"]
        X = lambda t: L + (W - L - R) * min(t, T) / T  # noqa: E731
        Y = lambda x: top + (bot - top) * (b - x) / (2 * b)  # noqa: E731

        c.create_rectangle(X(0), top, X(v["t0"]), bot, fill="#eeede8", width=0)

        peak = max(max(v["top"]), max(v["bottom"]), 1e-9)
        for pdf, y0, sign, colour in ((v["top"], Y(b), 1, self.GO), (v["bottom"], Y(-b), -1, self.NO)):
            pts = [X(0), y0]
            for i, p in enumerate(pdf):
                pts += [X((i + 0.5) * v["bin"]), y0 - sign * (strip - 8) * p / peak]
            pts += [X(T), y0]
            c.create_polygon(pts, fill=colour, stipple="gray25", outline=colour, width=1.5)

        for trace, end in zip(self.paths.trace, self.paths.ended):
            pts = [xy for t, x in trace if t <= now for xy in (X(t), Y(x))]
            if len(pts) >= 4:
                c.create_line(pts, fill=self.MUTED)
            if end is not None and end[0] <= now:
                x, y = X(end[0]), Y(end[1] * b)
                c.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, width=0,
                              fill=self.GO if end[1] > 0 else self.NO)

        t_hit = v["t0"] + b / abs(v["drift"]) if v["drift"] else float("inf")
        t_end = min(t_hit, T)
        c.create_line(X(v["t0"]), Y(0), X(t_end), Y(v["drift"] * (t_end - v["t0"])),
                      fill=self.INK, dash=(6, 5), width=1.5)

        for y in (b, -b):
            c.create_line(X(0), Y(y), X(T), Y(y), fill=self.INK, width=2)
        c.create_line(X(0), Y(0), X(T), Y(0), fill=self.INK, dash=(2, 4))
        for t in range(int(T) + 1):
            c.create_text(X(t), H - 8, text=f"{t}s", fill=self.MUTED, font=SMALL)
        for y, s in ((b, f"+{b}"), (0, "0"), (-b, f"\u2212{b}")):
            c.create_text(L - 6, Y(y), text=s, anchor="e", fill=self.MUTED, font=SMALL)
        c.create_text(X(0) + 4, Y(b) - strip + 10, anchor="w", fill=self.GO, font=SMALL,
                      text=f"{v['yes']}  {v['p_go']:.0%}")
        c.create_text(X(0) + 4, Y(-b) + strip - 8, anchor="w", fill=self.NO, font=SMALL,
                      text=f"{v['no']}  {v['p_no']:.0%}")
        go, no = self.paths.decided()
        c.create_text(X(T), Y(0) - 8, anchor="e", fill=self.MUTED, font=SMALL,
                      text=f"paths decided: {go} / {no} of {self.paths.n}")

        c.create_line(X(now), top - strip + 4, X(now), bot + strip - 4, fill=self.CLOCK, width=2)
        if v["go"] is not None:
            x = X(v["rt"])
            y = Y(0) if v["rt"] > T else Y(b if v["go"] else -b)
            c.create_oval(x - 6, y - 6, x + 6, y + 6, fill=self.CLOCK, width=0)
            right = v["rt"] > T / 2
            c.create_text(x + (-10 if right else 10), y - 10, text="you", fill=self.CLOCK,
                          anchor="e" if right else "w", font=SMALL)


def main():
    root = tk.Tk()
    app = DesktopApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Ctrl-C, a closed terminal or `kill` close the app the same way as the window's
    # close button, so the exit is logged with what was on screen.
    def stop(*_):
        app.on_close()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)

    def poll():
        # Python only runs a signal handler when it gets control back from Tk.
        if not app.closed:
            root.after(250, poll)

    poll()
    root.mainloop()


if __name__ == "__main__":
    main()
