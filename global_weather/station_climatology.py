"""Station-native seasonal and monthly climatological normals.

Calculated strictly over the chronological train period; no reanalysis is used.
Provides baseline normals for anomaly modeling and physical reconstruction.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from .observations import utc
from .pipeline.io import atomic_json, digest, read_json, sha256


class StationClimatology:
    """Monthly station normals and train standard deviations for surface variables."""

    def __init__(self, monthly_means: np.ndarray, stds: np.ndarray, metadata: dict | None = None):
        monthly = np.asarray(monthly_means, dtype=np.float32)
        scales = np.asarray(stds, dtype=np.float32)
        if monthly.ndim != 3 or monthly.shape[0] != 12 or monthly.shape[2] != 6:
            raise ValueError("Monthly means must have shape [12, n_stations, 6].")
        if scales.ndim != 1 or scales.shape[0] != 6 or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Scales must have shape [6] with finite positive values.")
        if not np.isfinite(monthly).all():
            raise ValueError("Monthly climatological means must be finite.")
        self.monthly = monthly
        self.std = scales
        self.metadata = metadata or {}
        self.n_stations = monthly.shape[1]

    def sample(self, station_indices: np.ndarray | list[int] | slice, when: datetime | str) -> np.ndarray:
        t = utc(when)
        month_idx = t.month - 1
        return self.monthly[month_idx, station_indices]

    def for_window(self, timestamps: list[datetime | str]) -> np.ndarray:
        """Return climatology array of shape [len(timestamps), n_stations, 6]."""
        result = np.zeros((len(timestamps), self.n_stations, 6), dtype=np.float32)
        for i, when in enumerate(timestamps):
            t = utc(when)
            result[i] = self.monthly[t.month - 1]
        return result

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray, timestamps: list[datetime],
            train_mask: np.ndarray) -> StationClimatology:
        """Fit monthly means per station and pooled train standard deviations."""
        if values.shape != mask.shape or values.ndim != 3 or values.shape[2] != 6:
            raise ValueError("Values and mask must have shape [hours, stations, 6].")
        n_hours, n_stations, n_vars = values.shape
        months = np.array([utc(ts).month - 1 for ts in timestamps], dtype=int)
        monthly = np.zeros((12, n_stations, n_vars), dtype=np.float32)

        # Pooled train mean and standard deviation across all valid station observations
        train_valid = mask & train_mask[:, None, None]
        global_means = np.zeros(n_vars, dtype=np.float32)
        global_stds = np.zeros(n_vars, dtype=np.float32)
        for v in range(n_vars):
            valid_train_v = train_valid[..., v]
            if not valid_train_v.any():
                raise ValueError(f"No train observations for variable {v}.")
            vals_v = values[..., v][valid_train_v]
            global_means[v] = float(np.mean(vals_v))
            global_stds[v] = float(np.std(vals_v))
            if global_stds[v] <= 0:
                raise ValueError(f"Non-positive standard deviation for variable {v}.")

        # Per-station station-wide train mean
        station_means = np.zeros((n_stations, n_vars), dtype=np.float32)
        for s in range(n_stations):
            for v in range(n_vars):
                vm = train_valid[:, s, v]
                if vm.any():
                    station_means[s, v] = float(np.mean(values[:, s, v][vm]))
                else:
                    station_means[s, v] = global_means[v]

        # Monthly station means
        for m in range(12):
            in_month = (months == m) & train_mask
            for s in range(n_stations):
                for v in range(n_vars):
                    vm = in_month & mask[:, s, v]
                    if vm.any():
                        monthly[m, s, v] = float(np.mean(values[:, s, v][vm]))
                    else:
                        monthly[m, s, v] = station_means[s, v]

        meta = {
            "schema": "station-monthly-climatology-1",
            "n_stations": n_stations,
            "variables": n_vars,
            "data_kind": "real",
            "global_stds": global_stds.tolist(),
        }
        return cls(monthly, global_stds, meta)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            monthly=self.monthly,
            std=self.std,
            metadata_json=json.dumps(self.metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> StationClimatology:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            monthly = data["monthly"]
            std = data["std"]
            meta = json.loads(str(data["metadata_json"]))
        return cls(monthly, std, meta)
