"""Fixed, bounded station-native research workflow supervised by a user systemd service.

The service owns the process group; SSH is only used to deploy and inspect it.
No commands or imports are accepted from experiment manifests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def environment_identity():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "scipy", "rasterio", "netCDF4", "cdsapi", "pytest")},
        "accelerators": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip() if shutil.which("nvidia-smi") else "CPU",
    }


def directory_bytes(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def stop(child):
    # A leader may exit while descendants still hold its process group.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def bounded_run(command, *, cwd, stdout, stderr, deadline, timeout, env,
                storage_root, minimum_free_bytes, maximum_directory_bytes):
    """Reap all descendants on timeout, resource stop or parent interruption."""
    started = time.monotonic()
    child = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                             env=env, start_new_session=True)
    last_scan = -60.0
    try:
        while child.poll() is None:
            if now() >= deadline or time.monotonic() - started >= timeout:
                raise TimeoutError("Execution deadline exceeded")
            if shutil.disk_usage(cwd).free < minimum_free_bytes:
                raise RuntimeError("Free disk reserve exhausted")
            elapsed = time.monotonic() - started
            if elapsed - last_scan >= 60:
                if directory_bytes(storage_root) > maximum_directory_bytes:
                    raise RuntimeError("Experiment directory budget exceeded")
                last_scan = elapsed
            time.sleep(1)
        return child.returncode
    finally:
        stop(child)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/remote-r3-stations"
    logs = root / "outputs/remote-r3-execution"
    plan = root / "configs/observation_r3_training.json"
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise SystemExit("Deadline must have a UTC offset")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if sha != args.expected_sha or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root):
        raise SystemExit("Source revision differs or tracked sources are dirty")
    logs.mkdir(parents=True, exist_ok=True)
    evidence = logs / "evidence.json"
    with (logs / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = environment_identity()
        report = json.loads(evidence.read_text()) if evidence.exists() else {
            "source_commit": sha, "cwd": str(root), "started_utc": now().isoformat(),
            "deadline_utc": deadline.isoformat(), "plan_sha256": digest(plan),
            "driver_sha256": digest(Path(__file__)), "attempts": [],
            "data_kind": "real", "meteorologically_validated": False,
            "environment": environment,
        }
        for key, actual in (("source_commit", sha), ("deadline_utc", deadline.isoformat()),
                            ("plan_sha256", digest(plan)), ("driver_sha256", digest(Path(__file__)))):
            if report[key] != actual:
                raise SystemExit("Resume identity differs: " + key)
        if report["environment"] != environment:
            report.update(status="blocked", reason="Numerical environment differs", finished_utc=now().isoformat())
            save(evidence, report)
            return 1
        if report.get("status") == "completed_station_native_research":
            return 0
        if deadline <= now():
            report.update(status="timed_out", finished_utc=now().isoformat())
            save(evidence, report)
            return 1
        if len(report["attempts"]) >= 3:
            report.update(status="blocked", reason="Attempt limit reached", finished_utc=now().isoformat())
            save(evidence, report)
            raise SystemExit("Attempt limit reached; inspect preserved evidence")
        if report["attempts"] and "finished_utc" not in report["attempts"][-1]:
            previous = report["attempts"][-1]
            previous.update(status="interrupted", detected_utc=now().isoformat())
            for step in previous["steps"]:
                if "returncode" not in step:
                    step.update(status="interrupted", returncode=None)
        attempt = {"number": len(report["attempts"]) + 1, "started_utc": now().isoformat(), "steps": []}
        report["attempts"].append(attempt)
        report["status"] = "running"
        save(evidence, report)
        python = sys.executable
        env = dict(os.environ, OMP_NUM_THREADS="8", CUDA_VISIBLE_DEVICES="0")
        report["environment_limits"] = {"OMP_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": "0"}
        report["resource_limits"] = {"cache_gib": 16, "directory_gib": 96, "minimum_free_disk_gib": 64}
        smoke = "outputs/remote-r3-checks"
        checks = [
            ("pytest", ["-m", "pytest", "-q"]),
            ("contracts", ["-m", "pytest", "tests/test_ecosystem.py", "tests/test_agent_contracts.py",
                           "tests/test_pipeline.py", "tests/test_reference_assets.py", "-q"]),
            ("baseline", ["-m", "global_weather.cli", "train-smoke", "--optimizer-steps", "2",
                          "--horizon-hours", "72", "--output", smoke + "/baseline"]),
            ("adaptive", ["-m", "global_weather.adaptive_cli", "--horizon-hours", "72", "--output", smoke + "/adaptive"]),
            ("multimodal", ["-m", "global_weather.multimodal", "selftest", "--output", smoke + "/multimodal"]),
            ("pipeline", ["scripts/pipeline-test.py"]),
            ("docs", ["scripts/check_docs.py", "--strict"]),
        ]
        cache = output / "cache"
        admission = output / "admission.json"
        dataset = output / "dataset"
        legacy_cache = "/home/user/global-ml-weather-r2/outputs/remote-r2-scale/cache/ghcnh"
        workflow = checks + [
            ("admit", ["-m", "global_weather.observation_data", "admit", "--source-cache", legacy_cache,
                        "--output", str(admission), "--station-limit", "1024", "--minimum-coverage", "0.05"]),
            ("download", ["-m", "global_weather.observation_acquisition", "--admission", str(admission),
                          "--cache", str(cache), "--output", str(output / "acquisition.json"), "--allow-network"]),
            ("prepare", ["-m", "global_weather.observation_data", "prepare", "--source-cache", legacy_cache,
                         "--source-cache", str(cache), "--admission", str(admission), "--output", str(dataset)]),
            ("train", ["-m", "global_weather.observation_training", "train", "--dataset", str(dataset),
                       "--config", str(plan), "--output", str(output / "training")]),
            ("test", ["-m", "global_weather.observation_training", "test", "--dataset", str(dataset),
                      "--training", str(output / "training"), "--output", str(output / "test.json")]),
        ]
        try:
            for name, arguments in workflow:
                command = [python, *arguments]
                # Successful checks on identical sources need not overwrite their artifacts after reboot.
                if name in {n for n, _ in checks} and any(
                        s["name"] == name and s.get("returncode") == 0
                        and s.get("command") == command
                        and set(s.get("logs", {})) == {"stdout", "stderr"}
                        and all(digest(logs / ref["path"]) == ref["sha256"] for ref in s["logs"].values())
                        for previous in report["attempts"][:-1] for s in previous["steps"]):
                    continue
                step = {"name": name, "command": command, "cwd": str(root),
                        "started_utc": now().isoformat(), "timeout_seconds": 1800 if name in {n for n, _ in checks} else max(1, (deadline - now()).total_seconds() - 30)}
                attempt["steps"].append(step)
                save(evidence, report)
                filenames = {k: f"attempt-{attempt['number']:02d}-{name}.{k}.log" for k in ("stdout", "stderr")}
                try:
                    with (logs / filenames["stdout"]).open("wb") as out, (logs / filenames["stderr"]).open("wb") as err:
                        code = bounded_run(command, cwd=root, stdout=out, stderr=err, deadline=deadline,
                                           timeout=step["timeout_seconds"], env=env, storage_root=output,
                                           minimum_free_bytes=64 * 1024**3, maximum_directory_bytes=96 * 1024**3)
                    step.update(returncode=code, status="passed" if code == 0 else "failed")
                except (OSError, RuntimeError, TimeoutError, KeyboardInterrupt) as exc:
                    step.update(returncode=None, status="failed", reason=type(exc).__name__ + ": " + str(exc))
                step["finished_utc"] = now().isoformat()
                step["logs"] = {k: {"path": filename, "sha256": digest(logs / filename)} for k, filename in filenames.items()}
                save(evidence, report)
                if step["status"] != "passed":
                    report.update(status="blocked", finished_utc=now().isoformat())
                    save(evidence, report)
                    return 1
            report.update(status="completed_station_native_research", finished_utc=now().isoformat())
            save(evidence, report)
            return 0
        finally:
            attempt["finished_utc"] = now().isoformat()
            save(evidence, report)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Supervisor received termination")
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
