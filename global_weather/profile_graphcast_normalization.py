"""Pinned GraphCast global level statistics for measured-profile R7.

No observations are fitted. Dataset hashes bind this immutable imported artifact
 to a measured-data experiment; norm coverage does not imply measured coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .import_climatology import PINNED_HASHES, bundled_directory, import_graphcast
from .observation_training import digest, save
from .profile_normalization import PRESSURE_PA, VARIABLES, PressureNormalization, _identity, _index, _lock
from .vertical import PROFILE_UNITS

SCHEMA = 'graphcast-pressure-normalization-1'
ARCHITECTURE = 'pressure-profile-graphcast-v1'
STATUS = 'measured_graphcast_profile_research_trained'


def _reference() -> dict:
    directory = bundled_directory()
    bundle = import_graphcast(directory / 'mean_by_level.nc', directory / 'stddev_by_level.nc',
                              expected_hashes=PINNED_HASHES)
    provenance = bundle._payload['provenance']
    if (provenance['source_attributes'].get('date_start') != '1979-01-02'
            or provenance['source_attributes'].get('date_end') != '2015'
            or provenance['fit_period'] is None):
        raise ValueError('Pinned GraphCast fitting period differs.')
    stats = [bundle.get(variable, units) for variable, units in zip(VARIABLES, PROFILE_UNITS[:5])]
    if any(stat.pressure_pa != tuple(PRESSURE_PA) for stat in stats):
        raise ValueError('GraphCast pressure coordinates differ.')
    return {'schema': SCHEMA, 'role': 'fixed_external_normalization_not_observation',
            'variables': list(VARIABLES), 'units': list(PROFILE_UNITS[:5]),
            'pressure_pa': PRESSURE_PA.tolist(), 'transforms': ['identity'] * 5,
            'mean': np.asarray([stat.mean for stat in stats]).T.tolist(),
            'std': np.asarray([stat.std for stat in stats]).T.tolist(),
            'support': np.ones((37, 5), dtype=bool).tolist(),
            'provenance': provenance, 'scientific_acceptance': False}


class GraphCastNormalization(PressureNormalization):
    """Affine physical z-scores; bounded log-pressure interpolation inherited."""

    humidity_transform = 'identity'
    architecture = ARCHITECTURE
    status = STATUS

    def __init__(self, payload: dict) -> None:
        reference = _reference()
        if any(payload.get(key) != value for key, value in reference.items()):
            raise ValueError('GraphCast artifact differs from pinned bytes, units or provenance.')
        identity = payload.get('source_identity')
        required = {'database_sha256', 'dataset_manifest_sha256', 'source_sha256', 'admission_sha256'}
        if (not isinstance(identity, dict) or set(identity) != required
                or any(not isinstance(value, str) or len(value) != 64
                       or any(c not in '0123456789abcdef' for c in value) for value in identity.values())):
            raise ValueError('GraphCast artifact requires four measured dataset hashes.')
        self.payload = payload
        self.mean = np.asarray(payload['mean'], dtype=np.float64)
        self.std = np.asarray(payload['std'], dtype=np.float64)
        self.support = np.asarray(payload['support'], dtype=bool)
        self.pressure_pa = PRESSURE_PA.copy()

    def verify_sources(self) -> None:
        """Recheck pinned NetCDF bytes and period during a running experiment."""
        if any(self.payload.get(key) != value for key, value in _reference().items()):
            raise ValueError('GraphCast artifact source changed during experiment.')

    def normalize(self, variable: str | int, value: float | np.ndarray,
                  pressure_pa: float | np.ndarray) -> np.ndarray:
        variable = _index(variable)
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all() or (variable == 1 and (value < 0).any()):
            raise ValueError('Measurements must be finite; specific humidity cannot be negative.')
        mean, std, support = self.at(variable, pressure_pa)
        return np.where(support, (value - mean) / std, np.nan)

    def inverse(self, variable: str | int, value: float | np.ndarray,
                pressure_pa: float | np.ndarray) -> np.ndarray:
        mean, std, support = self.at(variable, pressure_pa)
        return np.where(support, np.asarray(value, dtype=np.float64) * std + mean, np.nan)


def create(dataset_path: str | Path, output_file: str | Path) -> dict:
    """Import verified fixed statistics once; never fit a measured scalar."""
    from .profile_training import ProfileDataset

    destination = Path(output_file)
    with _lock(destination):
        dataset = ProfileDataset(dataset_path)
        identity = _identity(dataset)
        if destination.exists():
            payload = json.loads(destination.read_text())
            GraphCastNormalization(payload)
            if payload['source_identity'] != identity:
                raise ValueError('Immutable GraphCast normalization dataset identity changed.')
        else:
            payload = {**_reference(), 'source_identity': identity}
            GraphCastNormalization(payload)
            if destination.with_suffix('.tmp').exists() or destination.with_suffix('.tmp').is_symlink():
                raise ValueError('Ambiguous unfinished normalization; preserve for inspection.')
            if _identity(dataset) != identity:
                raise ValueError('Measured dataset changed during GraphCast import.')
            save(destination, payload)
        if _identity(dataset) != identity:
            raise ValueError('Measured dataset changed during GraphCast verification.')
        return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    payload = create(args.dataset, args.output)
    print(json.dumps({'schema': payload['schema'], 'supported_bins': 185,
                      'output': args.output, 'sha256': digest(args.output)}, allow_nan=False))


if __name__ == '__main__':
    main()
