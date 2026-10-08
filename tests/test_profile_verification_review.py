"""Frozen external diagnostics must preserve observation-trained state."""
from datetime import timedelta
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from global_weather import profile_verification as verification
from global_weather.grid import build_grid
from global_weather.observations import utc
from global_weather.vertical import PRESSURE_HPA

ISSUE = utc("2022-08-01T00:00:00Z")


def mocked_frozen_run(tmp_path, monkeypatch, *, mutate=None):
    events = []
    training = tmp_path / "training"
    checkpoint = training / "epoch-0001" / "state.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen-checkpoint-for-mocked-verification")
    ref = dict(directory="epoch-0001", epoch=1, sha256=verification.digest(checkpoint))
    (training / "best.json").write_text(json.dumps(ref), encoding="utf-8")
    completion = dict(status="measured_upper_air_research_trained", best_epoch=1,
                      identity={"config": {"max_records_per_window": 20}})
    (training / "complete.json").write_text(json.dumps(completion), encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "dataset.json").write_text('{"source_roles":{"norm":"train IGRA"}}', encoding="utf-8")

    class FrozenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.grid = build_grid(0)
            self.register_buffer("pressure_pa", torch.tensor(PRESSURE_HPA) * 100.)
            self.register_buffer("mean", torch.tensor([260., .002, 5., -2., 1000.]))
            self.register_buffer("std", torch.ones(5))
            self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
            self.eval()

        def forward(self, inputs, issue):
            events.append("prediction")
            assert not torch.is_grad_enabled()
            assert all(utc(row["observed_at"]) <= issue and utc(row["available_at"]) <= issue for row in inputs)
            frames = []
            for lead in range(0, 73, 3):
                profiles = torch.zeros((self.grid.n_cells, 37, 6))
                profiles[..., :5] = self.mean
                profiles[..., 0] += lead / 3
                profiles[..., 5] = float("nan")
                mask = torch.ones_like(profiles, dtype=torch.bool)
                mask[..., 5] = False
                frames.append(SimpleNamespace(profiles=profiles, profile_variable_mask=mask,
                              profile_mask=torch.ones((self.grid.n_cells, 37), dtype=torch.bool),
                              lead_hours=lead))
            return frames

    class CausalDataset:
        root = dataset_root

        def records(self, start, end, *, issue, split):
            events.append("inputs")
            assert start == ISSUE - timedelta(hours=12)
            assert end == issue == ISSUE
            assert split == "test"
            return [dict(observed_at=(issue - timedelta(hours=1)).isoformat(),
                         available_at=issue.isoformat(), variable="temperature")]

        def sample(self, *args, **kwargs):
            raise AssertionError("External inference read future station targets.")

        def verify(self):
            events.append("source_hash_check")

    model = FrozenModel()
    dataset = CausalDataset()
    monkeypatch.setattr(verification, "load_frozen", lambda *args: (model, dataset))

    def external_targets(pressure, surface, output, **kwargs):
        events.append("external")
        assert events.index("prediction") < events.index("external")
        assert kwargs["issue_time"] == ISSUE.isoformat()
        shape = (25, model.grid.n_cells, 37, 6)
        profiles = np.zeros(shape, dtype=np.float32)
        profiles[..., :5] = model.mean.numpy()
        profiles[..., 0] += np.arange(25)[:, None, None] + 3.
        mask = np.ones(shape, dtype=bool)
        mask[0, 0, 0, 0] = False  # One variable mask must not remove unrelated channels.
        np.savez(output, profiles=profiles, profile_mask=mask,
                 pressure_hpa=np.asarray(PRESSURE_HPA), lead_hours=np.arange(0, 73, 3))
        if mutate == "norm":
            model.mean.add_(1.)
        elif mutate == "weights":
            with torch.no_grad():
                model.weight.add_(1.)
        elif mutate == "manifest":
            (dataset.root / "dataset.json").write_text('{"source_roles":{"norm":"ERA5"}}', encoding="utf-8")
        return {"source": "mock external ERA5 fields"}

    monkeypatch.setattr(verification, "prepare_targets", external_targets)
    return training, dataset, model, events


def test_external_diagnostic_preserves_weights_norms_and_uses_only_causal_inputs(tmp_path, monkeypatch):
    training, dataset, model, events = mocked_frozen_run(tmp_path, monkeypatch)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    checkpoint_hash = verification.digest(training / "epoch-0001" / "state.pt")
    manifest_hash = verification.digest(dataset.root / "dataset.json")
    output = tmp_path / "external"
    rows = verification.verify(dataset.root, training, "pressure.nc", "surface.nc", ISSUE, output)
    assert events[:3] == ["inputs", "prediction", "external"]
    assert "source_hash_check" in events
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert verification.digest(training / "epoch-0001" / "state.pt") == checkpoint_hash
    assert verification.digest(dataset.root / "dataset.json") == manifest_hash
    temperature = next(row for row in rows if row["lead_hours"] == 0
                       and row["pressure_pa"] == 100000 and row["variable"] == "temperature")
    humidity = next(row for row in rows if row["lead_hours"] == 0
                    and row["pressure_pa"] == 100000 and row["variable"] == "specific_humidity")
    assert temperature["cells"] == model.grid.n_cells - 1
    assert temperature["rmse"] == pytest.approx(3.)
    assert humidity["cells"] == model.grid.n_cells
    report = json.loads((output / "verification.json").read_text())
    assert report["weight_updates"] == report["norm_updates"] == 0
    assert report["epoch_selection"] is False
    assert report["independent_truth"] is False
    assert report["future_targets_used_for_inference"] is False


@pytest.mark.parametrize("mutation", ["norm", "weights", "manifest"])
def test_external_reader_cannot_mutate_frozen_training_state(tmp_path, monkeypatch, mutation):
    training, dataset, _, _ = mocked_frozen_run(tmp_path, monkeypatch, mutate=mutation)
    output = tmp_path / "external"
    with pytest.raises(ValueError, match="changed|normalization"):
        verification.verify(dataset.root, training, "pressure.nc", "surface.nc", ISSUE, output)
    assert not (output / "verification.json").exists()


def test_external_verification_rejects_checkpoint_path_traversal_before_reading_targets(tmp_path, monkeypatch):
    training, dataset, _, events = mocked_frozen_run(tmp_path, monkeypatch)
    (training / "best.json").write_text(json.dumps(dict(directory="../../outside", epoch=1, sha256="0" * 64)))
    with pytest.raises(ValueError, match="reference"):
        verification.verify(dataset.root, training, "pressure.nc", "surface.nc", ISSUE, tmp_path / "external")
    assert not events


def test_external_verification_rejects_non_test_issue_before_prediction(tmp_path, monkeypatch):
    training, dataset, _, events = mocked_frozen_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="held-out"):
        verification.verify(dataset.root, training, "pressure.nc", "surface.nc", "2021-08-01T00:00:00Z", tmp_path / "external")
    assert not events
