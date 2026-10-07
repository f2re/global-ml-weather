"""Supervisor limits must terminate work, rather than merely update a status."""
from datetime import datetime, timedelta, timezone
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def supervisor():
    spec = importlib.util.spec_from_file_location(
        "remote_research_runner", Path(__file__).parents[1] / "scripts/remote-research-runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supervisor_records_actual_command_exit(tmp_path):
    module = supervisor()
    with (tmp_path / "stdout").open("wb") as out, (tmp_path / "stderr").open("wb") as err:
        code = module.bounded_run(
            [sys.executable, "-c", "raise SystemExit(7)"], cwd=tmp_path, stdout=out, stderr=err,
            deadline=datetime.now(timezone.utc) + timedelta(seconds=10), timeout=5,
            env=os.environ.copy(), storage_root=tmp_path,
            minimum_free_bytes=0, maximum_directory_bytes=1024**2)
    assert code == 7


@pytest.mark.skipif(os.name != "posix", reason="Process groups require POSIX")
def test_supervisor_reaps_process_group_on_deadline(tmp_path):
    module = supervisor()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    module.stop(process)
    assert process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)
    with (tmp_path / "stdout").open("wb") as out, (tmp_path / "stderr").open("wb") as err:
        with pytest.raises(TimeoutError):
            module.bounded_run(
                [sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path, stdout=out, stderr=err,
                deadline=datetime.now(timezone.utc) - timedelta(seconds=1), timeout=5,
                env=os.environ.copy(), storage_root=tmp_path,
                minimum_free_bytes=0, maximum_directory_bytes=1024**2)


@pytest.mark.skipif(os.name != "posix", reason="Process groups require POSIX")
def test_supervisor_kills_descendant_after_leader_exit(tmp_path):
    module = supervisor()
    ready = tmp_path / "ready"
    child_code = "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(" + repr(str(ready)) + ").write_text(str(os.getpid())); time.sleep(60)"
    leader = subprocess.Popen([sys.executable, "-c",
                               "import subprocess,sys; subprocess.Popen([sys.executable,'-c'," + repr(child_code) + "])"],
                              start_new_session=True)
    leader.wait(timeout=5)
    try:
        end = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < end:
            time.sleep(.05)
        assert ready.exists()
        pid = int(ready.read_text())
        module.stop(leader)
        # Linux may retain a reparented zombie briefly; it is no longer executing.
        end = time.monotonic() + 5
        while time.monotonic() < end:
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(.05)
        else:
            pytest.fail("Descendant still executing after group termination")
    finally:
        module.stop(leader)


def test_supervisor_refuses_exhausted_disk(tmp_path):
    module = supervisor()
    with (tmp_path / "stdout").open("wb") as out, (tmp_path / "stderr").open("wb") as err:
        with pytest.raises(RuntimeError, match="Free disk"):
            module.bounded_run(
                [sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path, stdout=out, stderr=err,
                deadline=datetime.now(timezone.utc) + timedelta(seconds=10), timeout=5,
                env=os.environ.copy(), storage_root=tmp_path,
                minimum_free_bytes=2**63, maximum_directory_bytes=1024**2)


def test_supervisor_evidence_is_atomically_replaced(tmp_path):
    module = supervisor()
    path = tmp_path / "evidence.json"
    module.save(path, {"status": "running"})
    original = module.digest(path)
    module.save(path, {"status": "blocked"})
    assert module.digest(path) != original
    assert not path.with_suffix(".tmp").exists()
