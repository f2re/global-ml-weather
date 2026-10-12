"""Autonomous unified observation research runner supervised by systemd or background daemon.

Runs detached from terminal; supervises training on GPU 0 with official 30-year climatology.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def now() -> datetime:
    return datetime.now(timezone.utc)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def stop(child: subprocess.Popen) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha")
    parser.add_argument("--dataset", default="/home/user/global-ml-weather-r3/outputs/remote-r3-v2-stations/dataset")
    parser.add_argument("--climatology", default="/home/user/global-ml-weather-r8-v2/outputs/remote-r8-stations/climatology")
    parser.add_argument("--output", default="outputs/remote-unified-observations")
    parser.add_argument("--config", default="configs/unified_observation_training.json")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    logs_dir = output / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.expected_sha and sha != args.expected_sha:
        raise SystemExit(f"Revision mismatch: expected {args.expected_sha}, got {sha}")

    cfg_path = root / args.config
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    mesh_level = cfg.get("mesh_level", 1)

    climatology_dir = Path(args.climatology)
    if mesh_level != 1 or not (climatology_dir / "seasonal-climatology.npz").exists():
        target_clim = root / f"outputs/climatology-level-{mesh_level}"
        if not (target_clim / "seasonal-climatology.npz").exists():
            raw_cache = Path("/home/user/global-weather-climate-cache/ncep1-1991-2020")
            if raw_cache.exists():
                print(f"Preparing 30-year climatology for mesh level {mesh_level}...", flush=True)
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "global_weather.seasonal_climatology",
                    "--source-dir", str(raw_cache),
                    "--output", str(target_clim),
                    "--mesh-level", str(mesh_level),
                ], cwd=root)
                climatology_dir = target_clim
        else:
            climatology_dir = target_clim

    evidence = output / "evidence.json"
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        report = {
            "source_commit": sha,
            "started_utc": now().isoformat(),
            "status": "running",
            "gpu_device": "0",
            "dataset": args.dataset,
            "climatology": str(climatology_dir),
            "config": args.config,
        }
        save(evidence, report)

        env = dict(
            os.environ,
            PYTHONPATH=str(root),
            CUDA_VISIBLE_DEVICES="0",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="8",
            PYTHONUNBUFFERED="1",
        )

        cmd = [
            sys.executable,
            "-m",
            "global_weather.unified_observation_training",
            "--dataset", args.dataset,
            "--climatology", str(climatology_dir),
            "--output", str(output),
            "--config", str(root / args.config),
        ]

        log_file = output / "training.log"
        with log_file.open("a") as out_stream:
            child = subprocess.Popen(
                cmd,
                cwd=root,
                stdout=out_stream,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            print(f"Unified observation training started (PID {child.pid}). Logging to {log_file}", flush=True)

            try:
                code = child.wait()
            except KeyboardInterrupt:
                stop(child)
                code = 130

        report["finished_utc"] = now().isoformat()
        report["returncode"] = code
        report["status"] = "completed" if code == 0 else "failed"
        save(evidence, report)
        print(f"Runner completed with exit code {code}", flush=True)
        return code


if __name__ == "__main__":
    sys.exit(main())
