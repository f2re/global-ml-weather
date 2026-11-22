"""Synthetic engineering probes for issue-time satellite fusion and station compatibility."""
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from global_weather.grid import build_pyramid, latlon
from global_weather.multimodal.contracts import Channel, Sensor, Sequence
from global_weather.station_anomaly_model import SatelliteStationAnomalyModel, StationAnomalyModel


ISSUE = datetime(2024, 8, 1, 12, tzinfo=timezone.utc)


@pytest.fixture
def station_models():
    torch.manual_seed(31)
    grids = build_pyramid(0)
    coords = latlon(grids[0].xyz[:4])
    scales = np.ones(6, dtype=np.float32)
    sensor = Sensor(
        "arctic_ir", "arktika_m", "ARCM1", "imager",
        (Channel("4", "brightness_temperature", "K", 250.0, 5.0, "fixture-calibration"),),
        "1" * 64, "2023-12-31T00:00:00Z", "synthetic",
    )
    legacy = StationAnomalyModel(grids, coords, scales, hidden=8)
    model = SatelliteStationAnomalyModel(grids, coords, scales, sensors=(sensor,), hidden=8, base_channels=8)
    model.initialize_station_core(legacy.state_dict())

    observed = ISSUE.timestamp() - 3600
    image = torch.ones((1, 8, 8))
    sequence = Sequence(
        sensor_id=sensor.id,
        values=(255 * image[:, None]).clone(),
        valid=image[:, None].bool(),
        observed_unix=torch.tensor([observed], dtype=torch.float64),
        available_unix=torch.tensor([observed + 120], dtype=torch.float64),
        latitude=image[0] * float(coords[0, 0]),
        longitude=image[0] * float(coords[0, 1]),
        view_zenith_deg=20 * image,
        solar_zenith_deg=torch.full_like(image, float("nan")),
        footprint_km=5 * image,
        area_m2=1e6 * image[0],
        frame_ids=("2" * 64,),
        channel_ids=("4",),
        grid_id="fixture-stable-grid",
        geometry_reference="fixture geometry",
        source_sha256="3" * 64,
        sensor_signature=sensor.measurement_signature,
        data_kind="synthetic",
        scan_start_unix=torch.tensor([observed - 120], dtype=torch.float64),
    )
    history = torch.randn((12, len(coords), 6))
    valid = torch.ones_like(history, dtype=torch.bool)
    return legacy, model, sequence, history, valid


def test_no_causal_satellite_support_matches_legacy_model(station_models):
    legacy, model, sequence, history, valid = station_models
    future = replace(sequence, available_unix=torch.tensor([ISSUE.timestamp() + 1], dtype=torch.float64))
    boundary = replace(sequence, scan_start_unix=torch.tensor([ISSUE.timestamp() - 43200], dtype=torch.float64))
    masked = replace(sequence, valid=torch.zeros_like(sequence.valid), values=torch.full_like(sequence.values, float("nan")))
    with torch.no_grad():
        original = legacy(history, valid)
        for frames in ((), (future,), (boundary,), (masked,)):
            result = model(history, valid, issue_time=ISSUE, satellite_sequences=frames)
            assert torch.equal(result.predicted_anomalies, original.predicted_anomalies)
            assert torch.equal(result.profile_diagnostics, original.profile_diagnostics)
            assert result.satellite_support.dtype == torch.bool
            assert not result.satellite_support.any()


def test_satellite_updates_only_supported_initial_columns_and_receives_gradients(station_models):
    legacy, model, sequence, history, valid = station_models
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(model.satellite_prefixes))
    with torch.no_grad():
        base_analysis, _ = legacy.analysis_state(history, valid)
        sat_analysis, _ = model.analysis_state(history, valid, issue_time=ISSUE, satellite_sequences=(sequence,))
        cell = int(model.operator.cells[0])
        inactive = torch.arange(len(base_analysis)) != cell
        assert torch.equal(base_analysis[inactive], sat_analysis[inactive])
        assert not torch.equal(base_analysis[cell], sat_analysis[cell])

    output = model(history, valid, issue_time=ISSUE, satellite_sequences=(sequence,))
    assert output.satellite_support.sum() == 1
    assert output.satellite_support[cell]
    assert not output.profile_target_mask.any()
    assert torch.isfinite(output.predicted_anomalies).all()
    output.predicted_anomalies[0].square().mean().backward()
    for module in (model.satellite_bank.encoders, model.source_fusion):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.count_nonzero() > 0 for gradient in gradients)


def test_duplicate_causal_frames_and_changed_station_geometry_are_rejected(station_models):
    legacy, model, sequence, history, valid = station_models
    with pytest.raises(ValueError, match="Duplicate satellite frame"):
        model(history, valid, issue_time=ISSUE, satellite_sequences=(sequence, sequence))
    with pytest.raises(ValueError, match="require an issue time"):
        model(history, valid, satellite_sequences=(sequence,))
    changed = legacy.state_dict()
    changed["operator.station_xyz"] = changed["operator.station_xyz"].flip(0)
    with pytest.raises(ValueError, match="geometry or normalization differs: operator.station_xyz"):
        model.initialize_station_core(changed)
    with pytest.raises(ValueError, match="different sensor contracts"):
        model.set_extra_state({**model.get_extra_state(), "sensor_contracts": []})


def test_station_sensor_allowlist_includes_only_the_reviewed_goes_channel():
    grids = build_pyramid(0)
    coords = latlon(grids[0].xyz[:4])
    scales = np.ones(6, dtype=np.float32)
    sensor = Sensor(
        "goes18_c13", "goes_abi", "GOES-18", "imager",
        (Channel("13", "brightness_temperature", "K", 250.0, 5.0, "fixture-calibration"),),
        "1" * 64, "2023-12-31T00:00:00Z", "real",
    )
    model = SatelliteStationAnomalyModel(grids, coords, scales, sensors=(sensor,), hidden=8, base_channels=8)
    assert model.sensors == (sensor,)
    with pytest.raises(ValueError, match="GOES-18"):
        replace(sensor, platform="GOES-16")
    with pytest.raises(ValueError, match="GOES-18"):
        replace(sensor, channels=(replace(sensor.channels[0], id="14"),))
    with pytest.raises(ValueError, match="real GOES-18 ABI C13"):
        SatelliteStationAnomalyModel(grids, coords, scales, sensors=(replace(sensor, data_kind="synthetic"),),
                                     hidden=8, base_channels=8)
