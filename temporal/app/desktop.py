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

        ttk.Button(row, text=f"1  {yes_label}", command=lambda: finish(1)).pack(side="left", padx=8)
        ttk.Button(row, text=f"2  {no_label}", command=lambda: finish(0)).pack(side="left", padx=8)
        self.root.bind("1", lambda e: finish(1))
        self.root.bind("2", lambda e: finish(0))

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

    def export(self):
        events = self.store.events()
        path = self.store.root / "decisions.csv"
        pd.json_normalize(events).to_csv(path, index=False)
        self.fit_label.config(text=f"wrote data/{path.name}\n{len(events)} events")


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
