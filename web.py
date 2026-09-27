# The playable task as a website: python web.py. The browser only draws screens and times
# them; every rule is interface.App's, run here, one App per open page.
# Run a single worker process: the live sessions are kept in memory.
#
# Nothing is kept from anyone but the researchers unless TEMPORAL_COLLECT=1, which waits on
# approval for research with human participants. Until then other people can only play
# as guests, whose records live in a temporary folder deleted when they leave.

import argparse
import io
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from interface import DEFAULTS, MIN_FIT_ROWS, App, fit_summary
from persistence import attempts_to_frame

DATA_DIR = Path(os.environ.get("TEMPORAL_DATA", Path(__file__).parent / "data" / "web"))
COLLECT = os.environ.get("TEMPORAL_COLLECT", "0") == "1" # keep records of consenting players
# Whose records are always kept, and who always sees dev stuff: the researchers themselves.
RESEARCHERS = {e.strip().lower() for e in os.environ.get("TEMPORAL_RESEARCHERS", "").split(",")
               if e.strip()}
PANEL = os.environ.get("TEMPORAL_PANEL", "0") == "1" # dev stuff for every player
CONSENT_VERSION = 1 # bump when the consent text in web/index.html changes
IDLE_SECONDS = 2 * 3600 # a page silent this long is taken as closed
PAGE = Path(__file__).parent / "web" / "index.html"
TOKEN = re.compile(r"[0-9a-f]{32}")
EMAIL = re.compile(r"[a-z0-9_%+-][a-z0-9._%+-]*@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}")

# Range each setting may take, as the desktop app's widgets allow.
LIMITS = {"target": (0.3, 0.95), "stages": (2, 8), "cold": (0.05, 1.5), "review": (0.0, 1.0)}
FORCED = ("continue", "repeat")

_pool = None
_pool_lock = threading.Lock()


# One process pool for every player's plans and fits.
def shared_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))
        return _pool


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


# Stands in for Tk's root: timers run on a thread, holding the player's lock.
class Root:
    def __init__(self, lock):
        self.lock = lock

    def after(self, ms, fn):
        def run():
            with self.lock:
                fn()
        timer = threading.Timer(ms / 1000, run)
        timer.daemon = True
        timer.start()
        return timer

    def after_cancel(self, timer):
        timer.cancel()

    # Window calls with nothing to do in a browser.
    def title(self, *_):
        pass

    geometry = minsize = destroy = title


# interface.App with the screens sent to a browser. Opening the page is launching the app,
# and closing or leaving it is closing the window: the same records, the same sessions.
class WebApp(App):
    def __init__(self, root, data_dir, panel=False, ephemeral=False):
        self.panel = panel
        self.ephemeral = ephemeral # a guest's record, deleted when they leave
        self.view = {"screen": None}
        self.seq = 0
        self.flashes = []  # shown in order before the next screen
        self._decide = None
        self._fit_future = None
        self.client_elapsed = None
        self.last_seen = time.time()
        super().__init__(root, data_dir)

    make_var = staticmethod(Var)

    def _build(self):
        pass

    def clear(self):
        self._decide = None

    def refresh_stats(self):
        pass

    def show(self, **view):
        self.seq += 1
        self.view = {**view, "seq": self.seq}

    def render_problem(self, text, header):
        self.show(screen="question", text=f"{text} = ?", header="   ".join(header))

    # RT is timed in the browser, from the moment the buttons are drawn.
    def ask_decision(self, prompt, yes_label, no_label, on_done):
        self.clear()
        self.prompt_at = time.perf_counter()
        self._decide = on_done
        self.show(screen="decision", prompt=prompt, yes=yes_label, no=no_label)

    # The browser holds the flash before drawing what comes next, and starts the clock after.
    def flash(self, text, then):
        self.flashes.append(text)
        then()

    def decide(self, go, rt):
        on_done, self._decide = self._decide, None
        on_done(go, rt)

    # Times from the browser where it sent them: they exclude the network and the flash.
    def exit_record(self):
        rec = super().exit_record()
        if self.client_elapsed is not None:
            if "rt" in rec:
                rec["rt"] = self.client_elapsed
            if "prompt_seconds" in rec:
                rec["prompt_seconds"] = self.client_elapsed
        return rec

    def quit(self, abandon=False):
        super().quit(abandon)
        self.show(screen="closed", abandoned=abandon)
        if self.ephemeral:
            shutil.rmtree(self.store.root, ignore_errors=True)

    def executor(self):
        return shared_pool()

    # The pool is shared; only this player's work is dropped.
    def shutdown(self):
        for fut in (self._plan_future, self._fit_future):
            if fut is not None:
                fut.cancel()

    def run_fit(self):
        if self._fit_future is not None:
            return
        df = self.frame()
        if len(df) < MIN_FIT_ROWS:
            self.fit_result = f"only {len(df)} decided rows\nkeep going"
            return
        attempts = attempts_to_frame(self.state.persistence)
        self.fit_result = f"fitting {len(df)} rows, {len(attempts)} questions..."
        self._fit_future = self.executor().submit(fit_summary, df, attempts)

    def poll_fit(self):
        fut = self._fit_future
        if fut is None or not fut.done():
            return
        self._fit_future = None
        try:
            self.fit_result = fut.result()
        except Exception as exc:
            self.fit_result = f"fit failed: {exc}"

    def payload(self):
        out = {"view": self.view, "flashes": self.flashes, "panel": self.panel,
               "saved": not self.ephemeral}
        self.flashes = []
        if self.panel:
            self.poll_fit()
            out.update(settings=self.settings(), stats=self.stats_text(), now=self.now_text(),
                       controller=self.controller_status, fit=self.fit_result,
                       fitting=self._fit_future is not None)
        return out


# ---------------------------------------------------------------- players

# A guest, or one email address. Their sessions follow one another, never overlap.
class Player:
    def __init__(self):
        self.lock = threading.RLock()
        self.app = None


_players = {}
_sessions = {} # token -> (player, the app that page opened)
_players_lock = threading.Lock()


def session(token):
    if not token or not TOKEN.fullmatch(token):
        raise HTTPException(400, "bad session token")
    with _players_lock:
        found = _sessions.get(token)
    if found is None:
        raise HTTPException(409, "this session has ended")
    return found


# The page's app, if it is still the live one and the request is about the screen it is
# showing now.
def live(pl, app, seq=None):
    if app is not pl.app or app.closed:
        raise HTTPException(409, "this session has ended")
    if seq is not None and seq != app.seq:
        raise HTTPException(409, "that screen has moved on")
    app.last_seen = time.time()
    return app


def settings_allowed(app):
    if not app.panel:
        raise HTTPException(403, "dev stuff is off for this player")


# Pages left open and silent: closed as if the window were. Tokens of closed pages go.
def reap():
    while True:
        time.sleep(60)
        with _players_lock:
            players = list(_players.values())
        for pl in players:
            with pl.lock:
                app = pl.app
                if app is not None and not app.closed and time.time() - app.last_seen > IDLE_SECONDS:
                    app.on_close()
        with _players_lock:
            for token, (_, app) in list(_sessions.items()):
                if app.closed:
                    del _sessions[token]
            for key, pl in list(_players.items()):
                if key.startswith("guest-") and (pl.app is None or pl.app.closed):
                    del _players[key]


# ---------------------------------------------------------------- routes

@asynccontextmanager
async def lifespan(_):
    threading.Thread(target=reap, daemon=True).start()
    yield


api = FastAPI(lifespan=lifespan)


class Open(BaseModel):
    email: str | None = None # None plays as a guest
    consent: bool = False


class Token(BaseModel):
    token: str


class Answer(Token):
    seq: int
    raw: str
    rt: float


class Timed(Token):
    seq: int
    rt: float | None = None


class Decide(Token):
    seq: int
    go: int
    rt: float


class Settings(Token):
    settings: dict


@api.get("/")
def index():
    return FileResponse(PAGE)


@api.get("/api/config")
def config():
    return {"collect": COLLECT, "consent_version": CONSENT_VERSION}


# Starting is launching the app. One still open for this player (another tab or device,
# or a reload whose close never arrived) is closed first, as an abandon.
@api.post("/api/open")
def open_session(body: Open):
    if body.email is None:
        email, researcher = None, False
        key = "guest-" + uuid.uuid4().hex
    else:
        email = body.email.strip().lower()
        if len(email) > 254 or not EMAIL.fullmatch(email):
            raise HTTPException(400, "that doesn't look like an email address")
        researcher = email in RESEARCHERS
        if not (COLLECT or researcher):
            raise HTTPException(403, "Data collection isn't enabled yet, so only guest play "
                                     "is open. Nothing a guest does is saved.")
        if not (body.consent or researcher):
            raise HTTPException(400, "please read and agree to the consent form first")
        key = email
    token = uuid.uuid4().hex
    with _players_lock:
        pl = _players.setdefault(key, Player())
    with pl.lock:
        if pl.app is not None and not pl.app.closed:
            pl.app.on_close()
        if email is None:
            data_dir, ephemeral = Path(tempfile.mkdtemp(prefix="temporal-guest-")), True
        else:
            data_dir, ephemeral = DATA_DIR / email, False
        pl.app = WebApp(Root(pl.lock), data_dir, panel=PANEL or researcher, ephemeral=ephemeral)
        pl.app.store.append({"node": "consent", "session": pl.app.session, "email": email,
                             "consented": body.consent, "researcher": researcher,
                             "consent_version": CONSENT_VERSION, "collect": COLLECT})
        with _players_lock:
            _sessions[token] = (pl, pl.app)
        return {"token": token, "email": email, **pl.app.payload()}


@api.post("/api/answer")
def answer(body: Answer):
    pl, app = session(body.token)
    with pl.lock:
        live(pl, app, body.seq)
        if app.screen != "question":
            raise HTTPException(409, "no question on screen")
        app.on_answer(body.raw[:32], max(body.rt, 0.0))
        return app.payload()


@api.post("/api/skip")
def skip(body: Timed):
    pl, app = session(body.token)
    with pl.lock:
        live(pl, app, body.seq)
        app.on_skip(None if body.rt is None else max(body.rt, 0.0))
        return app.payload()


@api.post("/api/decide")
def decide(body: Decide):
    pl, app = session(body.token)
    with pl.lock:
        live(pl, app, body.seq)
        if app._decide is None or body.go not in (0, 1):
            raise HTTPException(409, "no decision on screen")
        app.decide(body.go, max(body.rt, 0.0))
        return app.payload()


# The page is going away: log it the way closing the window is logged.
@api.post("/api/close")
def close(body: Timed):
    try:
        pl, app = session(body.token)
    except HTTPException:
        return {"ok": False}
    with pl.lock:
        if app is not pl.app or app.closed or body.seq != app.seq:
            return {"ok": False}
        app.client_elapsed = body.rt
        app.on_close()
        return {"ok": True}


@api.get("/api/state")
def state(token: str):
    pl, app = session(token)
    with pl.lock:
        live(pl, app)
        return app.payload()


@api.post("/api/settings")
def settings(body: Settings):
    pl, app = session(body.token)
    with pl.lock:
        live(pl, app)
        settings_allowed(app)
        for name, value in body.settings.items():
            if name not in DEFAULTS:
                continue
            kind = type(DEFAULTS[name])
            try:
                value = kind(value)
            except (TypeError, ValueError):
                continue
            if name in LIMITS:
                lo, hi = LIMITS[name]
                value = min(max(value, lo), hi)
            if name == "forced" and value not in FORCED:
                continue
            getattr(app, "v_" + name).set(value)
        app.persist()
        return app.payload()


@api.post("/api/fit")
def run_fit(body: Token):
    pl, app = session(body.token)
    with pl.lock:
        live(pl, app)
        settings_allowed(app)
        app.run_fit()
        return app.payload()


@api.get("/api/export")
def export(token: str):
    pl, app = session(token)
    with pl.lock:
        live(pl, app)
        settings_allowed(app)
        buf = io.StringIO()
        pd.json_normalize(app.store.events()).to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="decisions.csv"'})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Serve the playable task.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = ap.parse_args()
    uvicorn.run(api, host=args.host, port=args.port, workers=1)
