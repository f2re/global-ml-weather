"""Saved-artifact summaries preserve evidence and reject contradictory reports."""

import hashlib
import json

import pytest

from global_weather.experiments.report import build_report, main


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def completed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    weights = run / "epochs" / "000002" / "weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture weights; never deserialized")
    ref = {
        "path": "epochs/000002/weights.pt",
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    history = [
        {
            "epoch": epoch,
            "training_horizon_hours": 3,
            "validation_horizon_hours": 72,
            "train_loss": loss,
            "validation_loss": loss + 0.1,
            "elapsed_seconds": epoch * 1.5,
        }
        for epoch, loss in ((1, 0.5), (2, 0.25))
    ]
    setup = {
        "data_kind": "synthetic",
        "dataset_fingerprint": "a" * 64,
        "config": {"hidden": 16},
        "runtime": {"type": "cpu"},
        "software": {"source_sha256": {}},
        "estimated_activation_bytes": 12345,
    }
    best = {
        "data_kind": "synthetic",
        "dataset_fingerprint": "a" * 64,
        "selection": "validation_only",
        "best_score": 0.35,
        "weights": ref,
        "config": setup["config"],
        "runtime": setup["runtime"],
        "software": setup["software"],
    }
    trained = {
        "schema": "weather-training-report-1",
        "status": "trained_research",
        "data_kind": "synthetic",
        "dataset_fingerprint": "a" * 64,
        "normalization_fingerprint": "b" * 64,
        "history": history,
        "epochs_completed": 2,
        "best_validation_loss": 0.35,
        "test_set_used_for_selection": False,
    }
    for name, value in [
        ("setup.json", setup),
        ("history.json", history),
        ("report.json", trained),
        ("best.json", best),
    ]:
        write(run / name, value)
    evaluation = tmp_path / "test.json"
    scores = [
        {
            "lead_hours": lead,
            "variable": "temperature",
            "units": "K",
            "pressure_hpa": 500,
            "count": 3,
            "rmse": 2.0,
            "mae": 1.5,
            "bias": -0.5,
            "control_rmse": 4.0,
            "rmse_skill": 0.5,
        }
        for lead in (24, 72)
    ]
    write(
        evaluation,
        {
            "split": "test",
            "data_kind": "synthetic",
            "weights": ref,
            "selection_dataset_fingerprint": "a" * 64,
            "dataset_fingerprint": "c" * 64,
            "normalization_fingerprint": "b" * 64,
            "normalized_forecast_loss": 0.4,
            "sample_ids": ["test"],
            "scores": scores,
        },
    )
    return run, evaluation, tmp_path / "summary"


def test_summary_preserves_scores_and_synthetic_status(completed):
    run, evaluation, output = completed
    result = build_report(run, output, evaluation=evaluation)
    assert result["data_kind"] == "synthetic"
    assert result["meteorologically_validated"] is False
    assert result["best_epoch"] == 2
    assert [r["lead_hours"] for r in result["scores"]] == [24, 72]
    assert result["measured_resources"] == {"epoch_elapsed_seconds": [1.5, 3.0]}
    assert result["estimated_resources"] == {"activation_bytes": 12345}
    assert "memory" not in result["measured_resources"]
    assert "synthetic" in (output / "learning.svg").read_text()
    assert "Ошибка, K" in (output / "lead_000.svg").read_text()
    assert "48" not in [str(r["lead_hours"]) for r in result["scores"]]
    manifest = json.loads((output / "artifacts.json").read_text())
    for name, checksum in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum


def test_completed_training_without_evaluation_has_no_weather_metrics(completed):
    run, _, output = completed
    result = build_report(run, output)
    assert result["scores"] == []
    assert result["evaluation"] is None
    assert not (output / "scores.csv").exists()
    assert not list(output.glob("lead_*.svg"))


@pytest.mark.parametrize(
    "mutation",
    ["kind", "norm", "weight", "duplicate", "unit", "skill", "count", "nan", "mae"],
)
def test_rejects_contradictory_evaluation_before_output(completed, mutation):
    run, evaluation, output = completed
    data = json.loads(evaluation.read_text())
    row = data["scores"][0]
    if mutation == "kind":
        data["data_kind"] = "real"
    elif mutation == "norm":
        data["normalization_fingerprint"] = "d" * 64
    elif mutation == "weight":
        data["weights"]["sha256"] = "d" * 64
    elif mutation == "duplicate":
        data["scores"].append(dict(row))
    elif mutation == "unit":
        data["scores"][1]["units"] = "degC"
    elif mutation == "skill":
        row["rmse_skill"] = 0.99
    elif mutation == "count":
        row["count"] = 0
    elif mutation == "nan":
        row["rmse"] = float("nan")
    elif mutation == "mae":
        row["mae"] = 3.0
    write(evaluation, data)
    with pytest.raises(ValueError):
        build_report(run, output, evaluation=evaluation)
    assert not output.exists()


def test_zero_control_error_has_undefined_skill(completed):
    run, evaluation, output = completed
    data = json.loads(evaluation.read_text())
    data["scores"][0].update(control_rmse=0.0, rmse_skill=None)
    write(evaluation, data)
    assert (
        build_report(run, output, evaluation=evaluation)["scores"][0]["rmse_skill"]
        is None
    )


def test_changed_weight_bytes_rejected_without_loading_pickle(completed):
    run, _, output = completed
    (run / "epochs/000002/weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        build_report(run, output)


def test_path_traversal_rejected(completed):
    run, _, output = completed
    best = json.loads((run / "best.json").read_text())
    best["weights"]["path"] = "../outside.pt"
    write(run / "best.json", best)
    with pytest.raises(ValueError, match="путь"):
        build_report(run, output)


def test_output_not_overwritten(completed):
    run, _, output = completed
    output.mkdir()
    (output / "keep").write_text("retained")
    with pytest.raises(FileExistsError):
        build_report(run, output)
    assert (output / "keep").read_text() == "retained"


@pytest.mark.parametrize("surface", ["run", "evaluation", "output"])
def test_symlink_parent_is_rejected(completed, surface):
    run, evaluation, output = completed
    link = run.parent / "linked"
    link.symlink_to(run.parent, target_is_directory=True)
    if surface == "run":
        run = link / run.name
    elif surface == "evaluation":
        evaluation = link / evaluation.name
    else:
        output = link / output.name
    with pytest.raises(ValueError, match="Символические"):
        build_report(run, output, evaluation=evaluation)
    assert not (run.parent / "summary").exists()


def test_incomplete_epoch_rejected(completed):
    run, _, output = completed
    history = json.loads((run / "history.json").read_text())
    del history[0]["train_loss"]
    write(run / "history.json", history)
    report = json.loads((run / "report.json").read_text())
    report["history"] = history
    write(run / "report.json", report)
    with pytest.raises(ValueError, match="Неполная"):
        build_report(run, output)


def test_architecture_mismatch_rejected(completed):
    run, _, output = completed
    setup = json.loads((run / "setup.json").read_text())
    setup["config"]["hidden"] = 64
    write(run / "setup.json", setup)
    with pytest.raises(ValueError, match="Конфигурация"):
        build_report(run, output)


def test_resume_epoch_extension_preserves_authoritative_config(completed):
    run, _, output = completed
    setup = json.loads((run / "setup.json").read_text())
    setup["config"]["epochs"] = 1
    write(run / "setup.json", setup)
    best = json.loads((run / "best.json").read_text())
    best["config"]["epochs"] = 2
    write(run / "best.json", best)
    assert build_report(run, output)["config"]["epochs"] == 2


def test_cli_failure_has_nonzero_exit(completed):
    run, _, output = completed
    (run / "report.json").unlink()
    assert main(["--run", str(run), "--output", str(output)]) == 1
