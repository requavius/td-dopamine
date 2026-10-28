"""The taskboard (taskboard/): reading progress from a log, and the catalog of what a
checkout can run, read from the source. Nothing here starts a process or needs ssh."""
from pathlib import Path

from taskboard.board import catalog, load_hosts, progress, seconds

ROOT = Path(__file__).resolve().parents[1]


def test_progress_formats():
    assert progress(["[3/40] person 7", "Warning: something"]) == [3, 40]  # run_pool and recovery
    assert progress(["tests/test_regression.py ....   [ 45%]"]) == [45, 100]  # pytest
    assert progress([" 12%|█▏        | 12/100 [00:03<00:22]"]) == [12, 100]  # tqdm
    assert progress(["nothing to see"]) is None


def test_ps_elapsed_time():
    assert seconds("05:07") == 307
    assert seconds("1:02:03") == 3723
    assert seconds("2-00:00:01") == 172801


def test_catalog_reads_the_experiments():
    entries = {e["module"]: e for e in catalog(ROOT)}
    assert "experiments.controllers" in entries and "experiments.common" not in entries
    assert "pytest" in entries
    # the package's __main__ is listed, with the subcommand options it delegates to cli.py
    cli = entries["temporal"]
    assert "temporal.cli" not in entries
    modes = next(o for o in cli["options"] if o["flags"] == ["--mode"])
    assert modes["sub"] == "recovery" and "single" in modes["choices"]
    # options added through common.args(), and usage lines from the docstring
    assert any(o["flags"] == ["--replot"] for o in entries["experiments.controllers"]["options"])
    assert any("--replot" in u for u in entries["experiments.controllers"]["usage"])


def test_without_hosts_file_it_uses_this_computer():
    (host,) = load_hosts(ROOT / "taskboard" / "no-such-file.json")
    assert host["ssh"] is None and host["tag"] == "local" and host["slots"] >= 1
