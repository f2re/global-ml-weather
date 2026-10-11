"""Tests for station anomaly modeling, ingestion to spherical grid, and climate normalization."""
from datetime import datetime, timezone
import numpy as np
import pytest
import torch

from global_weather.grid import build_pyramid, latlon
from global_weather.observation_model import LEAD_HOURS
from global_weather.station_anomaly_model import (
    AnomalyForecast,
    StationAnomalyModel,
    anomaly_loss,
)
from global_weather.station_climatology import StationClimatology


def fixture_anomaly_setup():
    grids = build_pyramid(0)
    coords = latlon(grids[0].xyz[:4])
    n_stations = len(coords)
    monthly_means = np.zeros((12, n_stations, 6), dtype=np.float32)
    # Seasonal cycle for temperature (variable 0): 270 K in Jan, 295 K in July
    for m in range(12):
        monthly_means[m, :, 0] = 270.0 + 25.0 * np.sin(np.pi * m / 11.0)
        monthly_means[m, :, 1] = 265.0 + 20.0 * np.sin(np.pi * m / 11.0)  # dew point
        monthly_means[m, :, 2] = 2.0  # u10
        monthly_means[m, :, 3] = -1.0  # v10
        monthly_means[m, :, 4] = 98000.0  # surface pressure
        monthly_means[m, :, 5] = 101325.0  # mslp
    stds = np.array([8.0, 7.0, 4.0, 4.0, 1500.0, 1200.0], dtype=np.float32)
    climatology = StationClimatology(monthly_means, stds)
    model = StationAnomalyModel(grids, coords, stds, hidden=16)
    return model, climatology, coords


def test_station_anomaly_model_predicts_dimensionless_departure():
    model, climatology, _ = fixture_anomaly_setup()
    n_stations = climatology.n_stations

    # 12-hour history of observed departures from normal (e.g. +1.5 sigma heat anomaly)
    history_anomalies = torch.full((12, n_stations, 6), 1.5, dtype=torch.float32)
    history_mask = torch.ones((12, n_stations, 6), dtype=torch.bool)

    forecast = model(history_anomalies, history_mask)

    assert isinstance(forecast, AnomalyForecast)
    assert forecast.predicted_anomalies.shape == (24, n_stations, 6)
    assert forecast.lead_hours == LEAD_HOURS
    assert "departure from climatology" in forecast.prediction_target
    assert not forecast.scientific_acceptance

    # Reconstruct physical prediction using climatology at lead time
    jan_normal = torch.as_tensor(climatology.sample(slice(None), datetime(2021, 1, 15, tzinfo=timezone.utc)))
    target_normals = jan_normal.unsqueeze(0).expand(24, -1, -1)
    physical_preds = model.reconstruct_physical(forecast.predicted_anomalies, target_normals)

    assert physical_preds.shape == (24, n_stations, 6)
    # Check that temperature (channel 0) is around 270 K (Jan normal) + predicted anomaly
    assert 200.0 < float(physical_preds[0, 0, 0]) < 350.0


def test_data_ingestion_updates_grid_state():
    model, _, _ = fixture_anomaly_setup()
    n_stations = model.operator.cells.numel()

    # Case A: Strong observed anomaly present
    anomalies = torch.randn(12, n_stations, 6)
    mask_active = torch.ones(12, n_stations, 6, dtype=torch.bool)
    forecast_active = model(anomalies, mask_active)

    # Case B: Completely masked history (no measurements arrive)
    mask_empty = torch.zeros(12, n_stations, 6, dtype=torch.bool)
    forecast_empty = model(anomalies, mask_empty)

    # The predicted departure must be different because data was actively ingested into the grid
    diff = (forecast_active.predicted_anomalies - forecast_empty.predicted_anomalies).abs().sum()
    assert float(diff) > 1e-4, "Data ingestion did not change global predictions"


def test_gradient_flow_to_all_ingest_dynamic_and_head_parameters():
    torch.manual_seed(42)
    model, _, _ = fixture_anomaly_setup()
    n_stations = model.operator.cells.numel()

    history_anomalies = torch.randn(12, n_stations, 6)
    history_mask = torch.ones(12, n_stations, 6, dtype=torch.bool)
    forecast = model(history_anomalies, history_mask)

    target_anomalies = torch.randn(24, n_stations, 6)
    target_mask = torch.ones(24, n_stations, 6, dtype=torch.bool)

    loss = anomaly_loss(forecast.predicted_anomalies, target_anomalies, target_mask)
    assert torch.isfinite(loss)
    loss.backward()

    # Check gradients on all components
    trained_modules = [
        ("input_encoder", model.input_encoder),
        ("ingest", model.ingest),
        ("dynamic", model.dynamic),
        ("transition", model.transition),
        ("anomaly_head", model.anomaly_head),
        ("geometry", model.geometry),
    ]
    for name, module in trained_modules:
        for param_name, param in module.named_parameters():
            assert param.grad is not None, f"No grad for {name}.{param_name}"
            assert torch.isfinite(param.grad).all(), f"Nonfinite grad for {name}.{param_name}"
            assert param.grad.abs().sum() > 0, f"Zero grad for {name}.{param_name}"

    # Profile head is uncalibrated and untrained in surface anomaly supervision
    assert model.profile_head.weight.grad is None


def test_missing_and_nan_observations_are_masked():
    model, _, _ = fixture_anomaly_setup()
    n_stations = model.operator.cells.numel()

    history_anomalies = torch.zeros(12, n_stations, 6)
    history_mask = torch.ones(12, n_stations, 6, dtype=torch.bool)
    # Inject NaN in a masked position
    history_mask[3, 1, 0] = False
    history_anomalies[3, 1, 0] = float("nan")

    forecast = model(history_anomalies, history_mask)
    assert torch.isfinite(forecast.predicted_anomalies).all()

    target_anomalies = torch.zeros(24, n_stations, 6)
    target_mask = torch.ones(24, n_stations, 6, dtype=torch.bool)
    target_mask[5, 2, 4] = False
    target_anomalies[5, 2, 4] = float("nan")

    loss = anomaly_loss(forecast.predicted_anomalies, target_anomalies, target_mask)
    assert torch.isfinite(loss)


def test_station_anomaly_training_end_to_end(tmp_path):
    import json
    from test_observation_training import prepared
    from global_weather.station_anomaly_training import train

    dataset = prepared(tmp_path / "data")
    output = tmp_path / "training"
    config = dict(mesh_level=0, hidden=8, threads=1, seed=17, learning_rate=0.01,
                  epochs=2, patience=2, max_train_samples=2, max_val_samples=1)
    train(dataset.root, output, config)

    assert (output / "station_climatology.npz").exists()
    assert (output / "latest.json").exists()
    assert (output / "best.json").exists()
    assert (output / "complete.json").exists()

    complete = json.loads((output / "complete.json").read_text())
    assert complete["status"] == "surface_climatology_anomaly_research_trained"
    assert complete["prediction_target"] == "dimensionless anomaly departure from climatology"
    assert complete["best_epoch"] in (1, 2)

