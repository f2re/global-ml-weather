"""Observation supervision invariants; execute only in the remote environment."""
import builtins
from datetime import timedelta
import json

import numpy as np
import pytest
import torch

from global_weather import observation_data as data
from global_weather import observation_training as training
from global_weather.grid import build_pyramid
from global_weather.observation_model import (
    LEAD_HOURS, StationObservationModel, exact_lead_index, station_loss,
)


def records_at(timestamp, offset=0., *, station="USW00000001", delay=60):
    observed = data.utc(timestamp)
    bases = (280., 270., 3., 4., 99000., 101000.)
    return [dict(
        observation_id=f"GHCNh/{station}/{observed.isoformat()}/{variable}",
        provider="NOAA_GHCNh", source="station", revision=0, valid=True,
        variable=variable, units=unit, value=base + offset,
        latitude=55., longitude=37., observed_at=observed.isoformat(),
        available_at=(observed + timedelta(minutes=delay)).isoformat(),
        provider_qc={"native": {"Quality_Code": "1", "Source_Code": "223",
                                "Measurement_Code": ""}},
    ) for variable, unit, base in zip(data.VARIABLES, data.UNITS, bases)]


def prepared(tmp_path, *, future_offset=100.):
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    records = records_at("2021-01-02T15:00:00Z", 1.)
    records += records_at("2021-01-03T00:00:00Z", 2.)
    records += records_at("2021-01-03T03:00:00Z", 4.)
    records += records_at("2022-01-03T03:00:00Z", future_offset)
    records += records_at("2022-07-03T03:00:00Z", future_offset * 2)
    (cache / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    admission = tmp_path / "admission.json"
    data.admit(cache, admission, minimum_coverage=.00001)
    root = tmp_path / "dataset"
    data.prepare(cache, admission, root)
    return data.ObservationDataset(root)


def test_future_validation_test_values_do_not_fit_norms_or_select_stations(tmp_path):
    left = prepared(tmp_path / "left", future_offset=100.)
    right = prepared(tmp_path / "right", future_offset=100000.)
    np.testing.assert_array_equal(left.mean, right.mean)
    np.testing.assert_array_equal(left.std, right.std)
    assert left.manifest["stations"] == right.manifest["stations"]
    assert left.norm["period"] == [data.START.isoformat(), data.TRAIN_END.isoformat()]


def test_input_is_physical_and_delayed_observation_is_masked(tmp_path):
    ds = prepared(tmp_path)
    index = next(i for i, row in enumerate(ds.samples)
                 if data.utc(row["issue"]) == data.utc("2021-01-03T00:00:00Z"))
    sample = ds.sample(index)
    assert sample["input"].shape == (12, 1, 6)
    assert sample["input"][2, 0, 0] == 281.  # 15 UTC is available before the issue.
    assert sample["input_mask"][2, 0, 0]
    assert not sample["input_mask"][-1].any()  # 00 UTC arrives at 01 UTC.
    assert sample["target"][0, 0, 0] == 284.
    np.testing.assert_allclose(sample["normalized_target"][0, 0],
                               (sample["target"][0, 0] - ds.mean) / ds.std)
    original = sample["input"].copy()
    ds.values[ds.samples[index]["hour"] + 3] += 10000.
    np.testing.assert_array_equal(ds.sample(index)["input"], original)


def test_split_windows_have_strict_84_hour_separation(tmp_path):
    ds = prepared(tmp_path)
    for left, right in (("train", "validation"), ("validation", "test")):
        last = max(data.utc(ds.samples[i]["issue"]) for i in ds.subset(left))
        first = min(data.utc(ds.samples[i]["issue"]) for i in ds.subset(right))
        assert last + timedelta(hours=72) < first - timedelta(hours=12)


def test_preparation_cannot_import_era5_or_reference_normalization(tmp_path, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if "era5" in name.lower() or "import_climatology" in name:
            raise AssertionError("Observation preparation accessed reanalysis: " + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    ds = prepared(tmp_path)
    assert ds.manifest["source_roles"]["target"] == "GHCNh"
    assert ds.manifest["source_roles"]["norm"] == "train_GHCNh"
    assert ds.manifest["source_roles"]["static"] is None


@pytest.mark.parametrize("filename", ["norm.json", "hourly.npz", "observations.sqlite"])
def test_dataset_rejects_changed_artifacts_before_resume(tmp_path, filename):
    ds = prepared(tmp_path)
    path = ds.root / filename
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="changed|hash|differs"):
        data.ObservationDataset(ds.root)


def test_revoked_latest_revision_does_not_resurrect_old_measurement(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    row = records_at("2021-01-02T15:00:00Z")[0]
    withdrawn = dict(row, revision=1, valid=False, value=None)
    (cache / "observations.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(withdrawn) + "\n", encoding="utf-8")
    database = tmp_path / "unique.sqlite"
    with pytest.raises(ValueError, match="revision|Revoked|issue-aware"):
        data._unique(cache, database, train_only=True)


def test_reader_rejects_era5_target_role_before_first_training(tmp_path):
    ds = prepared(tmp_path)
    manifest = dict(ds.manifest)
    manifest["source_roles"] = dict(manifest["source_roles"], target="ERA5")
    (ds.root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="role|source|GHCNh|ERA5"):
        data.ObservationDataset(ds.root)


def test_checkpoint_rejects_traversal_and_mutated_bytes(tmp_path):
    with pytest.raises(ValueError, match="reference"):
        training.checkpoint_path(tmp_path, {"directory": "../../elsewhere", "epoch": 1,
                                            "sha256": "0" * 64})
    folder = tmp_path / "epoch-0001"
    folder.mkdir()
    path = folder / "state.pt"
    path.write_bytes(b"original")
    ref = {"directory": folder.name, "epoch": 1, "sha256": training.digest(path)}
    assert training.checkpoint_path(tmp_path, ref) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        training.checkpoint_path(tmp_path, ref)


def test_training_forbids_era5_and_recovers_best_after_latest_publication(tmp_path, monkeypatch):
    ds = prepared(tmp_path / "data")
    real_dataset = data.ObservationDataset
    class ShortDataset(real_dataset):
        def subset(self, split):
            assert split != "test", "Training or epoch selection requested final test."
            candidates = super().subset(split)
            usable = [i for i in candidates if self.sample(i)["target_mask"].any()
                      and (split != "train" or self.sample(i)["input_mask"].any())]
            return usable[:1]
    monkeypatch.setattr(training, "ObservationDataset", ShortDataset)
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if "era5" in name.lower() or "import_climatology" in name:
            raise AssertionError("Training accessed reanalysis: " + name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    original_save = training.save
    interrupted = []
    def fail_between_pointers(path, value):
        if str(path).endswith("best.json") and not interrupted:
            interrupted.append(True)
            raise OSError("Simulated interruption after latest publication")
        return original_save(path, value)
    monkeypatch.setattr(training, "save", fail_between_pointers)
    config = dict(mesh_level=0, hidden=8, threads=1, seed=17, learning_rate=.001,
                  weight_decay=0., gradient_clip=1., epochs=1, patience=2)
    output = tmp_path / "training"
    with pytest.raises(OSError, match="Simulated interruption"):
        training.train(ds.root, output, config)
    assert (output / "latest.json").exists()
    assert not (output / "best.json").exists()
    training.train(ds.root, output, config)
    best = json.loads((output / "best.json").read_text())
    latest = json.loads((output / "latest.json").read_text())
    assert best == latest
    assert best["epoch"] == 1
    assert json.loads((output / "complete.json").read_text())["scientific_acceptance"] is False


def test_station_loss_excludes_missing_nan_and_weights_variables_equally():
    prediction = torch.zeros((1, 2, 6), requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    target[0, :, 0] = 2.
    mask[0, :, 0] = True
    target[0, 0, 1] = 1.
    mask[0, 0, 1] = True
    result = station_loss(prediction, target, mask, torch.zeros(6), torch.ones(6))
    assert float(result) == pytest.approx(1.)  # mean(Huber(2)=1.5, Huber(1)=.5)
    result.backward()
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[~mask] == 0).all()
    with pytest.raises(ValueError, match="finite|observed|targets"):
        station_loss(prediction, target, torch.zeros_like(mask), torch.zeros(6), torch.ones(6))


def test_exact_target_endpoint_never_rounds_future_observation():
    issue = data.utc("2021-01-01T00:00:00Z")
    assert exact_lead_index(issue, issue + timedelta(hours=72)) == 23
    with pytest.raises(ValueError, match="exactly"):
        exact_lead_index(issue, issue + timedelta(hours=3, minutes=1))


def test_native_gradients_do_not_claim_profile_scientific_acceptance():
    torch.manual_seed(17)
    model = StationObservationModel(build_pyramid(0), [[55., 37.]],
                                    np.array([280., 270., 0., 0., 100000., 101000.]),
                                    np.array([10., 10., 5., 5., 1000., 1000.]), hidden=8)
    history = model.train_mean.expand(12, 1, 6).clone()
    mask = torch.ones_like(history, dtype=torch.bool)
    frame = model(history, mask)
    assert frame.lead_hours == LEAD_HOURS
    assert frame.native_normalized.shape == (24, 1, 6)
    assert frame.profile_diagnostics.shape[-2:] == (37, 6)
    assert not frame.profile_target_mask.any()
    assert frame.scientific_acceptance is False
    training.loss(frame.native_normalized, torch.ones_like(frame.native_normalized),
                  torch.ones_like(frame.native_normalized, dtype=torch.bool)).backward()
    assert model.input_encoder[0].weight.grad is not None
    assert torch.isfinite(model.input_encoder[0].weight.grad).all()
    assert model.input_encoder[0].weight.grad.abs().sum() > 0
    assert model.native_head[-1].weight.grad.abs().sum() > 0
    assert model.profile_head.weight.grad is None


def test_masked_history_payload_cannot_change_forecast():
    torch.manual_seed(17)
    model = StationObservationModel(build_pyramid(0), [[55., 37.]],
                                    np.zeros(6), np.ones(6), hidden=8).eval()
    history = torch.zeros((12, 1, 6))
    mask = torch.zeros_like(history, dtype=torch.bool)
    alternative = torch.full_like(history, float("nan"))
    with torch.no_grad():
        left = model(history, mask).native_normalized
        right = model(alternative, mask).native_normalized
    torch.testing.assert_close(left, right, rtol=0, atol=0)
