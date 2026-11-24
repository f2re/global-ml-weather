"""Exercise the launcher contract with stand-ins for systemd and model stages."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_station_satellite_detached.sh"
BEST = "9" * 64
BASE = "8" * 64

pytestmark = pytest.mark.skipif(
    not hasattr(os, "getuid") or os.getuid() == 0,
    reason="The satellite launcher requires a nonroot POSIX user.",
)


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def campaign(tmp_path):
    repo = tmp_path / "code"
    run = tmp_path / "run"
    dataset = tmp_path / "dataset"
    base = tmp_path / "base"
    tools = tmp_path / "tools"
    for path in (repo / "scripts", run, dataset, base, tools):
        path.mkdir(parents=True)
    launcher = repo / "scripts" / SCRIPT.name
    shutil.copyfile(SCRIPT, launcher)

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)

    git("init", "-q")
    git("add", str(launcher))
    git("-c", "user.name=Launcher fixture", "-c", "user.email=launcher@example.test",
        "commit", "-qm", "launcher fixture")
    revision = git("rev-parse", "HEAD").stdout.strip()
    manifest = run / "input.json"
    manifest.write_text("manifest fixture", encoding="utf-8")
    station = dataset / "dataset.json"
    station.write_text("station fixture", encoding="utf-8")
    climate = base / "station_climatology.npz"
    climate.write_text("climatology fixture", encoding="utf-8")
    python = tools / "python"
    python.write_text(f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if args[0] == "-":
    sys.stdin.read()
    print(json.loads(Path(args[1]).read_text())["sha256"])
    raise SystemExit(0)
if args[1] == "-":
    sys.stdin.read()
    stage = "admission"
else:
    stage = {{
        "global_weather.station_satellite_data": "fit",
        "global_weather.station_anomaly_training": "train",
        "global_weather.station_anomaly_evaluation": "test",
    }}[args[2]]
with open(os.environ["PILOT_STAGE_TRACE"], "a") as log:
    log.write(json.dumps([stage, args]) + "\\n")
if stage == os.environ.get("PILOT_FAIL_STAGE"):
    raise SystemExit(23)
if stage != "admission":
    output = Path(args[args.index("--output") + 1])
    if stage == "train":
        output.mkdir()
        (output / "best.json").write_text(json.dumps({{"sha256": "{BEST}"}}))
        (output / "complete.json").write_text("{{}}")
    else:
        output.write_text("{{}}")
""", encoding="utf-8")
    python.chmod(0o700)
    sudo = tools / "sudo"
    sudo.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >"$PILOT_MANAGER_TRACE"\n', encoding="utf-8")
    sudo.chmod(0o700)
    if shutil.which("sha256sum") is None:
        checksum = tools / "sha256sum"
        checksum.write_text('#!/bin/bash\nexec shasum -a 256 "$@"\n', encoding="utf-8")
        checksum.chmod(0o700)
    stage_trace = tmp_path / "stages"
    manager_trace = tmp_path / "manager"
    env = {
        **os.environ,
        "PATH": f"{tools}:{os.environ['PATH']}",
        "INVOCATION_ID": "f" * 32,
        "PILOT_STAGE_TRACE": str(stage_trace),
        "PILOT_MANAGER_TRACE": str(manager_trace),
        "CUDA_VISIBLE_DEVICES": "1",
    }
    env.pop("PILOT_FAIL_STAGE", None)
    args = list(map(str, [
        revision, python, dataset, base, manifest,
        _digest(manifest), _digest(station), BASE, _digest(climate),
    ]))

    def invoke(*, worker=False):
        mode = ["--worker"] if worker else []
        return subprocess.run(["bash", str(launcher), *mode, *args], env=env,
                              text=True, capture_output=True, check=False)

    return invoke, args, env, run, stage_trace, manager_trace


def test_launcher_pins_an_offline_nonroot_system_service(campaign):
    invoke, args, _, run, stage_trace, manager_trace = campaign
    result = invoke()
    assert result.returncode == 0, result.stderr
    command = manager_trace.read_text().splitlines()
    assert command[:2] == ["-n", "systemd-run"]
    assert f"--uid={os.getuid()}" in command
    assert "--property=PrivateNetwork=yes" in command
    assert "--property=ProtectSystem=strict" in command
    assert f"--property=ReadWritePaths={run}" in command
    assert "--property=PrivateTmp=yes" not in command
    assert "--setenv=CUDA_VISIBLE_DEVICES=1" in command
    assert command[command.index("--worker") + 1:] == args
    assert not stage_trace.exists()


def test_worker_stops_on_fit_error_leaves_evidence_and_refuses_reuse(campaign):
    invoke, _, env, run, stage_trace, _ = campaign
    env["PILOT_FAIL_STAGE"] = "fit"
    result = invoke(worker=True)
    assert result.returncode == 23, result.stderr
    trace = stage_trace.read_text()
    assert '"admission"' in trace and '"fit"' in trace
    assert '"train"' not in trace and '"test"' not in trace
    assert "stage=fit exit=23" in (run / "station-pilot.log").read_text()
    assert not (run / "training").exists()
    assert invoke(worker=True).returncode != 0
    assert stage_trace.read_text() == trace


def test_worker_evaluates_the_validation_selected_checkpoint(campaign):
    invoke, args, _, run, stage_trace, _ = campaign
    result = invoke(worker=True)
    assert result.returncode == 0, result.stderr
    stages = [json.loads(line) for line in stage_trace.read_text().splitlines()]
    assert [name for name, _ in stages] == ["admission", "fit", "train", "test"]
    train = stages[2][1]
    test = stages[3][1]
    assert train[train.index("--expected-base-sha") + 1] == args[7]
    assert test[test.index("--expected-checkpoint-sha") + 1] == BEST
    assert test[test.index("--satellite-manifest") + 1] == str(run / "frozen.json")
    assert (run / "test.json").is_file()
    assert "stage=evidence exit=0" in (run / "station-pilot.log").read_text()


def test_changed_input_is_rejected_before_launch(campaign):
    invoke, _, _, run, stage_trace, manager_trace = campaign
    (run / "input.json").write_text("changed", encoding="utf-8")
    result = invoke()
    assert result.returncode != 0
    assert "SHA256" in result.stderr
    assert not manager_trace.exists()
    assert not stage_trace.exists()
