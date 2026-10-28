"""Taskboard: start, watch, stop, split and move long runs across computers.

Run from the repo root:  python3 -m taskboard   (then open http://127.0.0.1:8765)

Nothing about the tasks is written down here. The page offers whatever the chosen
checkout (the main repo or any git worktree of it) can run: every module with a
`__main__` block, its usage lines from the docstring and its argparse options, read
from the source without importing it. A run can be split along any list option
(`--people 0-99`, `--reps 0,1`) into one piece per computer or into N pieces that the
first free computer takes; `{host}` in a command becomes that computer's tag
(hosts.json), made unique within the split, so `--tag {host}` gives runs_mac.csv and
runs_server.csv. Jobs started by hand in a checkout (or in a remote copy) show up too,
with progress when their output goes to a file, and can be stopped or moved.

Every task runs detached under its own process group, so it survives this server
restarting; its log, pid and exit code live in ~/.taskboard/runs/ on that computer.
The server polls each computer every few seconds over one shared ssh connection and
reads progress from the log: "[i/N]", pytest's "[ 45%]" or a tqdm bar. A remote copy
is synced from the task's checkout first (only when its code differs, and never while
something else is running in it), and the result files a remote run wrote are copied
back into that checkout. Standard library only.
"""
import argparse
import ast
import fcntl
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATE = HERE / "state.json"  # --state changes it
RUNS = "$HOME/.taskboard/runs"
POLL = 3.0
GRACE = 15.0  # seconds after launch before a missing pid counts as dead; and TERM -> KILL
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "-o", "ControlMaster=auto",
       "-o", "ControlPath=/tmp/taskboard-%C", "-o", "ControlPersist=120"]
# What a code sync leaves out: environments, local data, outputs (a remote copy may be newer).
SYNC_EXCLUDE = [".venv", ".git", "data", "private", "results", "logs", "__pycache__", "taskboard",
                "paper/figures", ".ruff_cache", ".pytest_cache", ".DS_Store", "CLAUDE.md", ".claude"]
LIVE = ("syncing", "starting", "running", "stopping")  # holding slots on a computer
POLLED = ("starting", "running", "stopping")  # a process exists (or is about to)
SCAN = ("experiments", "temporal", "paper")  # where runnable modules are looked for
# Prints the last progress line of a log, then its last lines.
TAIL = r"""tb_tail() { if [ -r "$1" ]; then
  echo "@@PROG $(tail -c 3000000 "$1" | tr '\r' '\n' | grep -aE '\[[0-9]+/[0-9]+\]|\[ *[0-9]+%\]|[0-9]+%\|' | tail -n 1)"
  tail -c 8000 "$1" | tr '\r' '\n' | grep -av '^[[:space:]]*$' | tail -n 15; fi; }
"""


def progress(lines):
    for line in reversed(lines):
        if m := re.search(r"\[(\d+)/(\d+)\]", line) or re.search(r"(\d+)/(\d+) \[", line):
            if int(m[2]) > 0:
                return [int(m[1]), int(m[2])]
        if m := re.search(r"\[\s*(\d{1,3})%\]", line) or re.search(r"(\d{1,3})%\|", line):
            return [int(m[1]), 100]
    return None


def seconds(etime):  # ps etime: [[dd-]hh:]mm:ss
    days, _, rest = etime.rpartition("-")
    total = 0
    for part in rest.split(":"):
        total = total * 60 + int(part)
    return total + 86400 * int(days or 0)


def sections(text):
    out, cur = [], None
    for line in text.split("\n"):
        if line.startswith("@@") and not line.startswith("@@PROG"):
            cur = (line[2:], [])
            out.append(cur)
        elif cur:
            cur[1].append(line)
    return out


def checkouts():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "worktree", "list", "--porcelain"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    found = []
    for block in out.strip().split("\n\n"):
        f = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        if "worktree" in f:
            branch = f.get("branch", "").removeprefix("refs/heads/") or f"detached {f.get('HEAD', '')[:7]}"
            found.append({"path": f["worktree"], "branch": branch})
    return found or [{"path": str(ROOT), "branch": "?"}]


# ---- the catalog: what a checkout can run, read from its source without importing it

class _Range:  # range() for evaluating defaults, refusing anything large
    def __call__(self, *a):
        r = range(*a)
        if len(r) > 100_000:
            raise ValueError("too long")
        return list(r)


SAFE = {"range": _Range(), "list": list, "tuple": tuple, "len": len, "str": str, "int": int,
        "float": float, "sorted": sorted, "round": round, "min": min, "max": max}


def _eval(node, env):
    return eval(compile(ast.Expression(node), "<default>", "eval"), {"__builtins__": {}, **SAFE}, env)


def _consts(tree):
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = _eval(node.value, env)
            except Exception:
                pass
    return env


def _shown(v):
    if isinstance(v, (list, tuple)):
        return ",".join(f"{x:g}" if isinstance(x, float) else str(x) for x in v)
    return None if v is None or isinstance(v, bool) else str(v)


def _options(scope, src, env):
    subs = {}
    for node in ast.walk(scope):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", "") == "add_parser" and node.value.args
                and isinstance(node.value.args[0], ast.Constant) and isinstance(node.targets[0], ast.Name)):
            subs[node.targets[0].id] = node.value.args[0].value
    opts = []
    for node in ast.walk(scope):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        o = {"flags": [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)],
             "sub": subs.get(getattr(node.func.value, "id", None))}
        for kw in node.keywords:
            if kw.arg not in ("help", "default", "choices", "action"):
                continue
            try:
                v = _eval(kw.value, env)
            except Exception:
                v = ast.get_source_segment(src, kw.value)
            if kw.arg == "default":
                o["default"] = _shown(v) if not isinstance(v, str) else v
            elif kw.arg == "choices":
                o["choices"] = [str(x) for x in v] if isinstance(v, (list, tuple)) else None
            else:
                o[kw.arg] = v if isinstance(v, str) else str(v)
        if o["flags"]:
            opts.append(o)
    return opts


def _parse(path):
    src = path.read_text()
    tree = ast.parse(src)
    return src, tree, _consts(tree)


def _module_entry(repo, path):
    src, tree, env = _parse(path)
    rel = path.relative_to(repo).with_suffix("")
    module = ".".join(rel.parts[:-1] if rel.name == "__main__" else rel.parts)
    doc = ast.get_docstring(tree) or ""
    opts = _options(tree, src, env)
    # Options added by a helper imported from the same repo: `args(__doc__)`, `cli.main()`.
    delegates = []
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        base = path.parent if node.level else repo
        for _ in range(max(0, node.level - 1)):
            base = base.parent
        other = base.joinpath(*(node.module or "").split(".")).with_suffix(".py")
        names = {a.name for a in node.names} & called
        if not names or not other.is_file():
            continue
        osrc, otree, oenv = _parse(other)
        fns = {fn.name: fn for fn in otree.body if isinstance(fn, ast.FunctionDef)}
        todo, done = list(names & set(fns)), set()
        while todo:  # the imported function, and what it calls in its own module
            fn = fns[todo.pop()]
            done.add(fn.name)
            opts += _options(fn, osrc, oenv)
            todo += [n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id in fns and n.func.id not in done and n.func.id not in todo]
        if rel.name == "__main__" and names & set(fns):
            doc = doc or ast.get_docstring(otree) or ""
            delegates.append(".".join(other.relative_to(repo).with_suffix("").parts))
    usage = []
    for line in doc.splitlines():
        if m := re.match(r"\s*(?:uv run\s+)?(python3?\s+-m\s+\S.*?)\s*$", line):
            usage.append("uv run " + re.sub(r"^python3?", "python", re.sub(r"\s+#.*$", "", m[1])))
    usage = list(dict.fromkeys(usage)) or [f"uv run python -m {module}"]
    summary = next((line.strip() for line in doc.splitlines() if line.strip()), "")
    return {"module": module, "summary": summary, "usage": usage, "options": opts, "delegates": delegates}


_CATALOG = {}


def catalog(repo):
    repo = Path(repo)
    files = [p for d in SCAN if (repo / d).is_dir() for p in sorted((repo / d).glob("*.py"))]
    stamp = max((p.stat().st_mtime for p in files), default=0)
    if _CATALOG.get(repo, (None,))[0] == stamp:
        return _CATALOG[repo][1]
    entries = []
    for p in files:
        try:
            if p.name == "__main__.py" or re.search(r"__name__\s*==\s*['\"]__main__['\"]", p.read_text()):
                entries.append(_module_entry(repo, p))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
    hidden = {d for e in entries for d in e.pop("delegates")}  # run through a package's __main__
    entries = [e for e in entries if e["module"] not in hidden]
    if (repo / "tests").is_dir():
        entries.append({"module": "pytest", "summary": "the tests in tests/", "usage": ["uv run pytest"],
                        "options": []})
    entries.sort(key=lambda e: (not e["module"].startswith("experiments"), e["module"]))
    _CATALOG[repo] = (stamp, entries)
    return entries


# ---- the board

def load_hosts(path):
    """The computers in hosts.json (personal, gitignored; see hosts.example.json), or
    this computer alone, tagged "local" like the experiments' own --tag default."""
    if Path(path).exists():
        return json.loads(Path(path).read_text())["hosts"]
    return [{"name": "this computer", "label": os.uname().nodename, "ssh": None, "dir": None, "tag": "local",
             "slots": max(1, (os.cpu_count() or 2) - 1), "env": "export PATH=$HOME/.local/bin:$PATH"}]


def commit_of(repo):
    """The commit a run starts from, and the code changes it doesn't include (results/
    and logs/ are outputs, so changes there don't count)."""
    def git(*a):
        return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, timeout=10).stdout
    try:
        head = git("rev-parse", "--short", "HEAD").strip()
        changed = [line[3:] for line in git("status", "--porcelain").splitlines()
                   if not line[3:].startswith(("results/", "logs/"))]
    except (OSError, subprocess.TimeoutExpired):
        return None, []
    return head or None, changed


class Board:
    def __init__(self, hosts_file):
        self.lock = threading.RLock()
        self.hosts = {h["name"]: h for h in load_hosts(hosts_file)}
        self.live = {n: {"online": None, "error": "", "stats": {}, "sync": "", "home": ""} for n in self.hosts}
        self.tasks, self.groups = {}, {}
        self.settings = {n: {} for n in self.hosts}
        self.synced = {}  # host -> checkout its remote copy was last synced from
        self.ext, self.ext_info, self.ext_stop = {}, {}, {}  # jobs started by hand
        self.procs = []
        self.sync_locks = {n: threading.Lock() for n in self.hosts}  # one sync per computer at a time
        self._checkouts = (0, [])
        if STATE.exists():
            saved = json.loads(STATE.read_text())
            self.tasks, self.groups = saved.get("tasks", {}), saved.get("groups", {})
            self.synced = saved.get("synced", {})
            for n, s in saved.get("settings", {}).items():
                if n in self.settings:
                    self.settings[n] = s
        for t in self.tasks.values():  # a sync cut off by a restart never launched
            if t["status"] == "syncing":
                t["status"] = "queued"

    # ---- persistence and small helpers

    def save(self):
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"tasks": self.tasks, "groups": self.groups, "settings": self.settings,
                                   "synced": self.synced}, indent=1))
        tmp.replace(STATE)

    def sh(self, name, script, timeout=20):
        host = self.hosts[name]
        cmd = [*SSH, host["ssh"], "bash -s"] if host.get("ssh") else ["bash", "-s"]
        return subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout, cwd=ROOT)

    def checkouts(self):
        if time.time() - self._checkouts[0] > 30:
            self._checkouts = (time.time(), checkouts())
        return self._checkouts[1]

    def capacity(self, name):
        return self.settings[name].get("capacity", self.hosts[name]["slots"])

    def used(self, name):
        mine = sum(t["eff"] for t in self.tasks.values() if t["status"] in LIVE and t.get("assigned") == name)
        hand = sum(x["workers"] for x in self.ext.values() if x["host"] == name and not x.get("ended"))
        return mine + min(hand, self.capacity(name))

    def workdir(self, name, repo):
        d = self.hosts[name].get("dir") or repo
        return d if d.startswith("~") else shlex.quote(d)

    def remote_home_dir(self, name):
        d = self.hosts[name].get("dir") or ""
        return self.live[name]["home"] + d[1:] if d.startswith("~") else d

    # ---- polling

    def loop(self):
        pool = ThreadPoolExecutor(len(self.hosts))
        while True:
            self.procs = [p for p in self.procs if p.poll() is None]  # reap local children
            list(pool.map(self.poll_host, list(self.hosts)))
            self.schedule()
            time.sleep(POLL)

    def poll_host(self, name):
        with self.lock:
            rids = [t["rid"] for t in self.tasks.values() if t.get("assigned") == name and t["status"] in POLLED]
            logs = {k: x["log"] for k, x in self.ext.items() if x["host"] == name and x.get("log")}
        script = TAIL + f"R={RUNS}\n" + "".join(f"""echo "@@T {rid}"
p=$(cat $R/{rid}.pid 2>/dev/null); echo "${{p:-none}}"
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo alive; else echo dead; fi
cat $R/{rid}.exit 2>/dev/null || echo none
tb_tail $R/{rid}.log
""" for rid in rids) + "".join(f'echo "@@X {k}"\ntb_tail {shlex.quote(p)}\n' for k, p in logs.items()) + """echo "@@P"
ps -A -o pid=,ppid=,pgid=,etime=,args=
echo "@@H"
echo "$HOME"
nproc 2>/dev/null || sysctl -n hw.ncpu
if [ -r /proc/loadavg ]; then cut -d' ' -f1 /proc/loadavg; else sysctl -n vm.loadavg | awk '{print $2}'; fi
if command -v free >/dev/null; then free -m | awk '/Mem:/{print $2, $7}'; else echo "$(( $(sysctl -n hw.memsize) / 1048576 )) -"; fi
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1; else echo -; fi
"""
        try:
            r = self.sh(name, script, timeout=15)
        except subprocess.TimeoutExpired:
            r = None
        live = self.live[name]
        if r is None or "@@H" not in r.stdout:
            live["online"] = False
            live["error"] = "timed out" if r is None else (r.stderr.strip().splitlines() or ["unreachable"])[-1]
            return
        live["online"], live["error"] = True, ""
        parts = sections(r.stdout)
        runs = {h[2:]: ls for h, ls in parts if h.startswith("T ")}
        tails = {h[2:]: ls for h, ls in parts if h.startswith("X ")}
        ps = next(ls for h, ls in parts if h == "P")
        h = next(ls for h, ls in parts if h == "H") + ["-"] * 5
        live["home"] = h[0]
        mem = h[3].split()
        gpu = [x.strip() for x in h[4].split(",")] if h[4] != "-" else None
        live["stats"] = {"cores": int(h[1]) if h[1].isdigit() else None,
                         "load": float(h[2]) if re.fullmatch(r"[\d.]+", h[2]) else None,
                         "mem_total": int(mem[0]) if mem and mem[0].isdigit() else None,
                         "mem_avail": int(mem[1]) if len(mem) > 1 and mem[1].isdigit() else None,
                         "gpu": gpu}
        now = time.time()
        with self.lock:
            mine = set()
            for t in self.tasks.values():
                if t.get("assigned") != name or t["status"] not in POLLED or t["rid"] not in runs:
                    continue
                pid, state, code, *log = runs[t["rid"]] + ["", "", ""]
                if pid.isdigit():
                    mine.add(int(pid))
                self.read_log(t, log)
                if state == "alive":
                    if t["status"] == "starting":
                        t["status"] = "running"
                    if t["status"] == "stopping" and now - t["stop_at"] > GRACE and not t.get("killed"):
                        t["killed"] = True
                        self.signal(t, "KILL")
                elif code != "none" or now - t["started"] > GRACE:
                    self.finish(t, int(code) if code.lstrip("-").isdigit() else None)
            self.save()
        self.find_external(name, ps, mine, tails, now)

    def read_log(self, x, lines):
        prog = [line[7:] for line in lines if line.startswith("@@PROG ")]
        tail = [line for line in lines if not line.startswith("@@PROG") and line]
        x["tail"] = tail[-15:]
        x["progress"] = progress(tail + prog) or x.get("progress")

    def finish(self, t, code):
        t["ended"], t["exit"] = time.time(), code
        if t.get("stopping"):
            t["status"] = "stopped"
            if t.get("move_to"):
                t.update(status="queued", host=t.pop("move_to"), note=f"moved from {t['assigned']}")
        elif code == 0:
            t["status"] = "done"
            if t.get("fetch") and self.hosts[t["assigned"]].get("ssh"):
                t["fetching"] = True
                threading.Thread(target=self.fetch, args=(t["id"],), daemon=True).start()
        else:
            t["status"] = "failed" if code is not None else "lost"

    # ---- jobs started by hand

    def find_external(self, name, ps, mine, tails, now):
        procs = {}
        for line in ps:
            f = line.split(None, 4)
            if len(f) == 5 and f[0].isdigit():
                procs[int(f[0])] = {"ppid": int(f[1]), "pgid": int(f[2]), "etime": f[3], "args": f[4]}

        def jobish(args):
            toks = args.split()
            exe = os.path.basename(toks[0])
            if "taskboard" in args:
                return False
            if exe == "uv":
                return "run" in toks[1:3]
            return bool(re.fullmatch(r"python[\d.]*", exe)) and "-m" in toks

        kids = {}
        for pid, p in procs.items():
            kids.setdefault(p["ppid"], []).append(pid)

        def tree(pid):
            out, todo = [], [pid]
            while todo:
                out.append(todo.pop())
                todo += kids.get(out[-1], [])
            return out

        roots = [pid for pid, p in procs.items() if jobish(p["args"])
                 and not (p["ppid"] in procs and jobish(procs[p["ppid"]]["args"]))
                 and p["pgid"] not in mine]
        new = [pid for pid in roots if (name, pid) not in self.ext_info]
        if new:
            self.lookup(name, new)
        homes = [c["path"] for c in self.checkouts()]
        remote = self.remote_home_dir(name) if self.hosts[name].get("ssh") else None
        seen = set()
        with self.lock:
            for pid in roots:
                info = self.ext_info.get((name, pid), {})
                cwd = info.get("cwd", "")
                if remote is not None:
                    if not remote or cwd.rstrip("/") != remote.rstrip("/"):
                        continue
                    repo = self.synced.get(name)
                else:
                    repo = next((c for c in sorted(homes, key=len, reverse=True)
                                 if cwd == c or cwd.startswith(c + "/")), None)
                    if not repo:
                        continue
                p = procs[pid]
                family = tree(pid)
                pythons = sum("python" in procs[q]["args"] and "resource_tracker" not in procs[q]["args"]
                              for q in family)
                key = f"{name}:{pid}"
                seen.add(key)
                m = re.search(r"((?:-m\s+\S+|run\s+\S+)(?:\s.*)?)$", p["args"])
                cmd = m[1] if m else p["args"]
                cmd = ("uv run python " + cmd) if cmd.startswith("-m") else "uv " + cmd
                x = self.ext.setdefault(key, {"key": key, "host": name, "pid": pid})
                x.update(command=cmd, cwd=cwd, repo=repo, log=info.get("log"), family=family,
                         started=now - seconds(p["etime"]), workers=max(1, pythons - 1), ended=None)
                if key in tails:
                    self.read_log(x, tails[key])
                if key in self.ext_stop:
                    x["stopping"] = True
                    if now - self.ext_stop[key] > GRACE and not x.get("killed"):
                        x["killed"] = True
                        self.kill_family(name, family, "KILL")
            for key, x in list(self.ext.items()):
                if x["host"] == name and key not in seen:
                    if not x.get("ended"):
                        x["ended"] = now
                        self.ext_stop.pop(key, None)
                    elif now - x["ended"] > 1800:
                        del self.ext[key]

    def lookup(self, name, pids):
        script = "".join(f"""echo "@@I {p}"
if [ -d /proc/{p} ]; then readlink /proc/{p}/cwd; readlink /proc/{p}/fd/1
else lsof -a -p {p} -d cwd,1 -Fn 2>/dev/null | sed -n 's/^n//p'; fi
""" for p in pids)
        try:
            out = self.sh(name, script, timeout=15).stdout
        except subprocess.TimeoutExpired:
            return
        for head, lines in sections(out):
            lines = [x for x in lines if x] + ["", ""]
            log = lines[1] if lines[1].startswith("/") and not lines[1].startswith("/dev/") else None
            self.ext_info[(name, int(head[2:]))] = {"cwd": lines[0], "log": log}

    def kill_family(self, name, family, sig):
        pids = " ".join(map(str, family))
        threading.Thread(target=self.sh, args=(name, f"kill -{sig} {pids} 2>/dev/null; true"), daemon=True).start()

    def ext_stop_job(self, key, move_to=None):
        with self.lock:
            x = self.ext[key]
            if x.get("ended") or key in self.ext_stop:
                return
            self.ext_stop[key] = time.time()
            x["stopping"] = True
            self.kill_family(x["host"], x["family"], "TERM")
            if move_to:
                src_tag = self.hosts[x["host"]].get("tag", x["host"])
                cmd = re.sub(rf"(--tag[ =]){re.escape(src_tag)}\b", r"\1{host}", x["command"])
                repo = x.get("repo") or str(ROOT)
                self.new_task({"command": cmd, "host": move_to, "name": self.short(cmd)}, repo=repo,
                              sync=True, fetch=True, slots=0, note=f"moved from {x['host']} (started by hand)")
                self.save()

    # ---- scheduling and launching

    def schedule(self):
        now = time.time()
        with self.lock:
            for gid, g in self.groups.items():  # a split run's follow-up, once every piece is in
                ts = [t for t in self.tasks.values() if t.get("group") == gid]
                if g.get("after") and not g.get("fired") and ts and all(
                        t["status"] == "done" and not t.get("fetching") for t in ts):
                    g["fired"] = True
                    self.new_task({"command": g["after"], "host": g["after_host"], "name": f"{g['name']} · then"},
                                  repo=g["repo"], sync=g.get("sync"), fetch=True, slots=g.get("slots", 0), note="")
            queued = sorted((t for t in self.tasks.values() if t["status"] == "queued" and t.get("hold", 0) < now),
                            key=lambda t: t["order"])
            for t in queued:
                names = list(self.hosts) if t["host"] == "any" else [t["host"]]
                names.sort(key=lambda n: self.used(n) - self.capacity(n))  # most free first
                for name in names:
                    if not self.live[name]["online"] or self.settings[name].get("paused"):
                        continue
                    cap = self.capacity(name)
                    eff = min(t["slots"] or cap, cap)
                    if eff <= 0 or self.used(name) + eff > cap:
                        continue
                    sync = t.get("sync") and self.hosts[name].get("ssh")
                    t.update(status="syncing" if sync else "starting", assigned=name, eff=eff,
                             attempt=t.get("attempt", 0) + 1, started=now, ended=None, exit=None,
                             progress=None, tail=[], stopping=False, killed=False, note="")
                    t["rid"] = f"{t['id']}-{t['attempt']}"
                    self.resolve(t, name)
                    threading.Thread(target=self.launch, args=(t["id"],), daemon=True).start()
                    break
            self.save()

    def resolve(self, t, name):
        tag = base = self.hosts[name].get("tag", name)
        if t.get("group"):
            taken = {x.get("tag") for x in self.tasks.values() if x.get("group") == t["group"] and x is not t}
            n = 2
            while tag in taken:
                tag, n = f"{base}{n}", n + 1
        t["tag"], t["ran"] = tag, t["command"].replace("{host}", tag)

    def busy(self, name, me):
        for t in self.tasks.values():
            if t is not me and t.get("assigned") == name and t["status"] in POLLED:
                return f'"{t["name"]}"'
        for x in self.ext.values():
            if x["host"] == name and not x.get("ended"):
                return f'"{self.short(x["command"])}" (started by hand)'
        return None

    def launch(self, tid):
        with self.lock:
            t = dict(self.tasks[tid])
        name, repo = t["assigned"], t["repo"]
        host = self.hosts[name]
        if t["status"] == "syncing":
            with self.sync_locks[name]:
                self.sync_then_start(tid, name, repo)
            with self.lock:
                if self.tasks.get(tid, {}).get("status") != "starting":
                    return
        elif host.get("ssh") and self.synced.get(name) not in (None, repo):
            with self.lock:
                self.tasks[tid]["note"] = f"ran with the code last synced from {self.synced[name]}"
        self.start(tid)

    def sync_then_start(self, tid, name, repo):  # the caller holds sync_locks[name]
        host = self.hosts[name]
        changed = self.sync_changes(name, repo)
        with self.lock:
            cur = self.tasks.get(tid)
            if not cur or cur["status"] != "syncing":
                return
            if cur.get("stopping"):
                cur.update(status="stopped", ended=time.time())
                if cur.get("move_to"):
                    cur.update(status="queued", host=cur.pop("move_to"))
                return self.save()
            blocker = self.busy(name, cur) if changed else None
            if changed is None or blocker:
                cur.update(status="queued", hold=time.time() + 30, attempt=cur["attempt"] - 1,
                           note=(f"can't reach {name} to sync" if changed is None else
                                 f"waiting to sync code to {name}: {blocker} is running in {host['dir']}"))
                return self.save()
        if changed:
            ok, msg = self.sync(name, repo)
            if not ok:
                with self.lock:
                    self.tasks[tid].update(status="failed", ended=time.time(), note=f"sync failed: {msg}")
                    return self.save()
        with self.lock:
            self.synced[name] = repo
            self.tasks[tid].update(status="starting", started=time.time())

    def start(self, tid):
        with self.lock:
            t = self.tasks[tid]
            name, rid, repo = t["assigned"], t["rid"], t["repo"]
        head, changed = commit_of(repo)
        with self.lock:
            t["commit"], t["changed"] = head, changed[:20]
            t = dict(t)
        host = self.hosts[name]
        script = f"""echo $$ > {RUNS}/{rid}.pid
trap 'echo $? > {RUNS}/{rid}.exit' EXIT
cd {self.workdir(name, repo)} || exit 97
{host.get("env", "")}
export TEMPORAL_WORKERS={t["eff"]} PYTHONUNBUFFERED=1
echo "[taskboard] $(hostname) $(date '+%F %T') commit {head}{" + uncommitted changes" if changed else ""} workers={t["eff"]}: "{shlex.quote(t["ran"])}
command -v uv >/dev/null || echo "[taskboard] uv is not on PATH here: install it, or set env in hosts.json"
{t["ran"]}
"""
        try:
            if host.get("ssh"):
                r = self.sh(name, f"""mkdir -p {RUNS}
cat > {RUNS}/{rid}.sh <<'TASKBOARD_EOF'
{script}TASKBOARD_EOF
setsid nohup bash {RUNS}/{rid}.sh > {RUNS}/{rid}.log 2>&1 < /dev/null &
echo launched
""")
                if "launched" not in r.stdout:
                    raise RuntimeError(r.stderr.strip() or "launch failed")
            else:  # no setsid on macOS: start the new session from here
                runs = Path(os.path.expandvars(RUNS))
                runs.mkdir(parents=True, exist_ok=True)
                (runs / f"{rid}.sh").write_text(script)
                with open(runs / f"{rid}.log", "w") as log:
                    self.procs.append(subprocess.Popen(
                        ["bash", str(runs / f"{rid}.sh")], stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True, cwd=repo))
        except Exception as e:
            with self.lock:
                self.tasks[tid].update(status="failed", ended=time.time(), note=f"launch: {e}")
                self.save()

    def signal(self, t, sig):
        script = (f'p=$(cat {RUNS}/{t["rid"]}.pid 2>/dev/null); '
                  f'[ -n "$p" ] && {{ kill -{sig} -- -$p 2>/dev/null || kill -{sig} $p; }}; true')
        threading.Thread(target=self.sh, args=(t["assigned"], script), daemon=True).start()

    # ---- rsync

    def rsync(self, args):
        r = subprocess.run(["rsync", "-az", "-e", " ".join(SSH), *args], capture_output=True, text=True, timeout=600)
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1], r.stdout

    def _sync_args(self, name, repo):
        h = self.hosts[name]
        return ["--checksum", *(f"--exclude={x}" for x in SYNC_EXCLUDE), f"{repo}/", f"{h['ssh']}:{h['dir']}/"]

    def sync_changes(self, name, repo):  # files a sync would change; None if it can't tell
        ok, _, out = self.rsync(["--dry-run", "--itemize-changes", *self._sync_args(name, repo)])
        return [line for line in out.splitlines() if line[:1] in "<>c"] if ok else None

    def sync(self, name, repo):  # the caller holds sync_locks[name]
        self.live[name]["sync"] = "syncing code..."
        ok, msg, _ = self.rsync(self._sync_args(name, repo))
        branch = next((c["branch"] for c in self.checkouts() if c["path"] == repo), repo)
        self.live[name]["sync"] = f"code from {branch}, {time.strftime('%H:%M')}" if ok else f"sync failed: {msg}"
        if ok:
            with self.lock:
                self.synced[name] = repo
                self.save()
        return ok, msg

    def sync_now(self, name, repo):
        with self.sync_locks[name]:
            self._sync_now(name, repo)

    def _sync_now(self, name, repo):
        changed = self.sync_changes(name, repo)
        with self.lock:
            blocker = self.busy(name, None) if changed else None
        if changed is None:
            self.live[name]["sync"] = f"can't reach {name}"
        elif blocker:
            self.live[name]["sync"] = f"not synced: {blocker} is running there"
        elif changed:
            self.sync(name, repo)
        else:
            with self.lock:
                self.synced[name] = repo
            self.live[name]["sync"] = f"already up to date, {time.strftime('%H:%M')}"

    def fetch(self, tid):  # copy back the result files this run wrote, into its checkout
        with self.lock:
            t = dict(self.tasks[tid])
        host = self.hosts[t["assigned"]]
        try:
            r = self.sh(t["assigned"], f"cd {host['dir']} && find results -type f -newer {RUNS}/{t['rid']}.sh 2>/dev/null")
            files = r.stdout.split()
            if files:
                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                    f.write("\n".join(files))
                ok, msg, _ = self.rsync(["--update", f"--files-from={f.name}", f"{host['ssh']}:{host['dir']}/", f"{t['repo']}/"])
                os.unlink(f.name)
                note = f"copied back {len(files)} result file(s)" if ok else f"copy back failed: {msg}"
            else:
                note = "no new result files to copy back"
        except Exception as e:
            note = f"copy back failed: {e}"
        with self.lock:
            self.tasks[tid].update(note=note, fetching=False)
            self.save()

    # ---- actions from the page

    @staticmethod
    def short(cmd):
        m = re.search(r"-m\s+(\S+)(.*)", cmd)
        if not m:
            return " ".join(cmd.split()[:3])[:40]
        rest = re.sub(r"--tag[ =]\S+", "", m[2]).strip()
        return m[1].split(".")[-1] + (f" {rest}" if rest else "")

    def new_task(self, d, repo, sync, fetch, slots, note, group=None, first=False):
        orders = [t["order"] for t in self.tasks.values()] or [0]
        tid = uuid.uuid4().hex[:8]
        cmd = d["command"].strip()
        self.tasks[tid] = {
            "id": tid, "name": d.get("name", "").strip() or self.short(cmd), "command": cmd,
            "host": d.get("host") or "any", "slots": int(slots or 0), "repo": repo, "sync": bool(sync),
            "fetch": bool(fetch), "status": "queued", "created": time.time(),
            "order": min(orders) - 1 if first else max(orders) + 1, "attempt": 0, "note": note, "group": group}
        return tid

    def add(self, d):
        repo = d.get("repo") or str(ROOT)
        if repo not in {c["path"] for c in self.checkouts()}:
            raise KeyError(repo)
        with self.lock:
            gid = None
            if d.get("group"):
                gid = uuid.uuid4().hex[:8]
                g = d["group"]
                self.groups[gid] = {"name": g["name"], "after": (g.get("after") or "").strip(), "repo": repo,
                                    "after_host": g.get("after_host") or "any", "sync": d.get("sync"),
                                    "slots": int(d.get("slots") or 0)}
            made = [self.new_task(t, repo, d.get("sync"), d.get("fetch"), d.get("slots"), "", gid, d.get("first"))
                    for t in d["tasks"] if t.get("command", "").strip()]
            self.save()
        self.schedule()
        return made

    def stop(self, tid, move_to=None):
        with self.lock:
            t = self.tasks[tid]
            if t["status"] == "queued":
                t["status"], t["ended"] = "stopped", time.time()
            elif t["status"] in LIVE and not t.get("stopping"):
                t.update(stopping=True, stop_at=time.time(), move_to=move_to)
                if t["status"] != "syncing":
                    t["status"] = "stopping"
                    self.signal(t, "TERM")
            elif t["status"] in LIVE:
                t["move_to"] = move_to
            self.save()

    def move(self, tid, host):
        with self.lock:
            t = self.tasks[tid]
            if t["status"] in LIVE:
                if t.get("assigned") != host:
                    self.stop(tid, move_to=host)
                return
            t.update(host=host, status="queued", note="", hold=0)
            self.save()
        self.schedule()

    def top(self, tid):
        with self.lock:
            self.tasks[tid]["order"] = min(t["order"] for t in self.tasks.values()) - 1
            self.save()
        self.schedule()

    def remove(self, tid):
        with self.lock:
            if self.tasks[tid]["status"] not in LIVE:
                del self.tasks[tid]
                self.save()

    def clear(self):
        with self.lock:
            for tid in [k for k, t in self.tasks.items() if t["status"] in ("done", "stopped")]:
                del self.tasks[tid]
            live_groups = {t.get("group") for t in self.tasks.values()}
            self.groups = {g: v for g, v in self.groups.items() if g in live_groups}
            for key in [k for k, x in self.ext.items() if x.get("ended")]:
                del self.ext[key]
            self.save()

    def set_host(self, name, d):
        with self.lock:
            for key in ("paused", "capacity"):
                if key in d:
                    self.settings[name][key] = int(d[key]) if key == "capacity" else bool(d[key])
            self.save()
        self.schedule()

    def log(self, key):
        with self.lock:
            if key in self.ext:
                name, path = self.ext[key]["host"], self.ext[key].get("log")
                if not path:
                    return "(this job's output isn't going to a file, so there's nothing to show)"
            else:
                t = self.tasks[key]
                if not t.get("rid"):
                    return "(not started yet)"
                name, path = t["assigned"], f"{RUNS}/{t['rid']}.log"
        try:
            r = self.sh(name, f"tail -c 400000 {path if path.startswith('$') else shlex.quote(path)} 2>&1", timeout=15)
            return r.stdout.replace("\r", "\n") or r.stderr
        except subprocess.TimeoutExpired:
            return "(timed out reading the log)"

    def waiting(self, t):  # why a queued task hasn't started
        why = []
        for n in (list(self.hosts) if t["host"] == "any" else [t["host"]]):
            cap = self.capacity(n)
            if not self.live[n]["online"]:
                why.append(f"{n} is offline")
            elif self.settings[n].get("paused"):
                why.append(f"{n} is paused")
            else:
                need = min(t["slots"] or cap, cap)
                why.append(f"{n} has {max(0, cap - self.used(n))} of {cap} slots free, needs {need}")
        return "; ".join(why)

    def snapshot(self):
        now = time.time()
        with self.lock:
            hosts = [{**{k: h.get(k) for k in ("name", "label", "ssh", "dir", "tag")}, **self.live[n],
                      "capacity": self.capacity(n), "used": self.used(n),
                      "paused": self.settings[n].get("paused", False), "synced": self.synced.get(n)}
                     for n, h in self.hosts.items()]
            tasks = []
            for t in sorted(self.tasks.values(), key=lambda t: t["order"]):
                t = dict(t)
                p, s = t.get("progress"), t.get("started")
                if t["status"] == "running" and p and p[0] > 0 and s:
                    t["eta"] = (now - s) * (p[1] - p[0]) / p[0]
                if t["status"] == "queued" and not t.get("note"):
                    t["waiting"] = self.waiting(t)
                tasks.append(t)
            ext = []
            for x in self.ext.values():
                x = {k: v for k, v in x.items() if k != "family"}
                p = x.get("progress")
                if not x.get("ended") and p and p[0] > 0:
                    x["eta"] = (now - x["started"]) * (p[1] - p[0]) / p[0]
                ext.append(x)
            groups = {g: {"name": v["name"], "after": v.get("after"), "fired": v.get("fired", False),
                          "pieces": sum(t.get("group") == g for t in self.tasks.values())}
                      for g, v in self.groups.items()}
        return {"now": now, "hosts": hosts, "tasks": tasks, "external": ext, "groups": groups,
                "checkouts": self.checkouts(), "root": str(ROOT)}


class Handler(BaseHTTPRequestHandler):
    board: Board

    def log_message(self, *a):
        pass

    def reply(self, body, kind="application/json", code=200):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            return self.reply((HERE / "index.html").read_text(), "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self.reply(self.board.snapshot())
        if url.path == "/api/catalog":
            repo = q.get("repo", [str(ROOT)])[0]
            if repo not in {c["path"] for c in self.board.checkouts()}:
                return self.reply({"error": "unknown checkout"}, code=400)
            return self.reply(catalog(repo))
        if url.path == "/api/log":
            return self.reply(self.board.log(q["id"][0]), "text/plain; charset=utf-8")
        self.reply({"error": "not found"}, code=404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        d = json.loads(self.rfile.read(n) or b"{}")
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/")]  # api/<kind>/<id>/<action>
        b = self.board
        try:
            match parts[1:]:
                case ["tasks"]:
                    return self.reply({"ids": b.add(d)})
                case ["tasks", "clear"]:
                    b.clear()
                case ["tasks", tid, "stop"]:
                    b.stop(tid)
                case ["tasks", tid, "move"]:
                    b.move(tid, d["host"])
                case ["tasks", tid, "top"]:
                    b.top(tid)
                case ["tasks", tid, "remove"]:
                    b.remove(tid)
                case ["external", key, "stop"]:
                    b.ext_stop_job(key)
                case ["external", key, "move"]:
                    b.ext_stop_job(key, move_to=d["host"])
                case ["hosts", name]:
                    b.set_host(name, d)
                case ["hosts", name, "sync"]:
                    if name not in b.hosts or d.get("repo") not in {c["path"] for c in b.checkouts()}:
                        raise KeyError(name)
                    threading.Thread(target=b.sync_now, args=(name, d["repo"]), daemon=True).start()
                case _:
                    return self.reply({"error": "not found"}, code=404)
        except KeyError as e:
            return self.reply({"error": f"unknown {e}"}, code=400)
        self.reply({"ok": True})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bind", default="127.0.0.1", help="0.0.0.0 to open it from other machines")
    p.add_argument("--hosts", default=str(HERE / "hosts.json"),
                   help="the computers it can use (default: hosts.json if it exists, else this computer alone)")
    p.add_argument("--state", help="where it keeps its tasks (default taskboard/state.json)")
    a = p.parse_args()
    global STATE
    STATE = Path(a.state).resolve() if a.state else STATE
    lock = open(STATE.with_suffix(".lock"), "w")  # two boards on one state file overwrite each other
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another taskboard is already using {STATE}; stop it, or pass --state")
    Handler.board = board = Board(a.hosts)
    threading.Thread(target=board.loop, daemon=True).start()
    print(f"taskboard on http://{a.bind}:{a.port}  (state in {STATE})", flush=True)
    ThreadingHTTPServer((a.bind, a.port), Handler).serve_forever()
