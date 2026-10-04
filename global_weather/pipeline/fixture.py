"""Explicit synthetic analytic fixture. Never label these arrays as observations."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from .io import atomic_json, reference, write_arrays
from ..grid import build_grid, latlon
from ..vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS


def fields(grid, when):
    n = grid.n_cells; p = np.array(PRESSURE_HPA)*100.
    phase = (when-datetime(2020, 1, 1, tzinfo=timezone.utc)).total_seconds()/86400*.15
    xyz = grid.xyz
    wave = xyz[:, 0]*np.cos(phase)+xyz[:, 1]*np.sin(phase)+.3*xyz[:, 2]
    profile = np.zeros((n, 37, 6), np.float32)
    profile[:, :, 0] = 210+70*(p[None, :]/100000)**.2+5*wave[:, None]
    profile[:, :, 1] = .0004 + .008*(p[None, :]/100000)**2*(1+.2*wave[:, None])
    profile[:, :, 2] = 10*wave[:, None]*(.5+np.log(100000/p)[None, :]/7)
    profile[:, :, 3] = 4*xyz[:, 2, None]+2*np.cos(phase)*xyz[:, 0, None]
    tv = profile[:, :, 0]*(1+.608*profile[:, :, 1])
    profile[:, 1:, 4] = np.cumsum(287.05*.5*(tv[:, :-1]+tv[:, 1:])*np.log(p[:-1]/p[1:]), axis=1)
    profile[:, :, 4] += 100*wave[:, None]
    profile[:, :, 5] = .05*wave[:, None]*(p[None, :]/100000)
    surface = np.column_stack((280+5*wave, 276+3*wave, 5*wave, 3*xyz[:, 2]+np.cos(phase)*xyz[:, 0],
                               101000+400*wave, 101100+300*wave, 2+.3*wave, .5+.1*wave)).astype(np.float32)
    return profile, surface


def create_fixture(output, *, mesh_level=0, horizon_hours=3):
    """Five disjoint issue dates; fields are analytic, not a random-model teacher."""
    from ..observations import DEFAULT_VARIABLES
    from .fit import fit_normalization
    if mesh_level not in (0, 1, 2) or horizon_hours not in (3, 6, 12, 24, 48, 72):
        raise ValueError('Синтетический пример ограничен сеткой 0–2 и трёхчасовым шагом.')
    root = Path(output).absolute(); root.mkdir(parents=True, exist_ok=False)
    grid = build_grid(mesh_level); ll = latlon(grid.xyz)
    registry = {k: asdict(v) for k, v in DEFAULT_VARIABLES.items()}
    atomic_json(root/'registry.json', registry)
    write_arrays(root/'static.npz', elevation_m=np.zeros(grid.n_cells, np.float32),
                 land_fraction=np.zeros(grid.n_cells, np.float32), surface_units=np.array(['m', '1']),
                 grid_fingerprint=grid.fingerprint)
    rows = []; leads = list(range(0, horizon_hours+1, 3))
    origin = datetime(2020, 1, 1, 12, tzinfo=timezone.utc)
    # Six-day spacing ensures 72-hour targets and 12-hour inputs do not touch across splits.
    for index, split in enumerate(('train', 'train', 'train', 'validation', 'test')):
        issue = origin+timedelta(days=6*index); name = f'sample-{index}'
        profiles, surface = zip(*(fields(grid, issue+timedelta(hours=h)) for h in leads))
        profiles, surface = np.stack(profiles), np.stack(surface)
        pm, sm = np.ones_like(profiles, bool), np.ones_like(surface, bool)
        sm[0, :, 6] = False; surface[0, :, 6] = np.nan
        write_arrays(root/f'{name}.npz', profiles=profiles, profile_mask=pm, surface=surface, surface_mask=sm,
                     pressure_hpa=np.array(PRESSURE_HPA), lead_hours=np.array(leads), issue_time=issue.isoformat(),
                     grid_fingerprint=grid.fingerprint, profile_variables=np.array(PROFILE_VARIABLES),
                     profile_units=np.array(PROFILE_UNITS), surface_variables=np.array(SURFACE_VARIABLES),
                     surface_units=np.array(SURFACE_UNITS))
        records = []
        for age in (0, 3, 6, 9):
            measured = issue-timedelta(hours=age, minutes=10)
            pr, su = fields(grid, measured)
            for cell in range(0, grid.n_cells, max(1, grid.n_cells//8)):
                for k, variable in enumerate(('t2m', 'td2m', 'u10', 'v10', 'surface_pressure')):
                    if (cell+age+k) % 5 == 0:
                        continue  # Missing quantities are absent, never fake zeros.
                    records.append({'observation_id': f'{name}-{age}-{cell}-{variable}', 'source': 'station',
                                    'variable': variable, 'value': float(su[cell, k]), 'units': registry[variable]['units'],
                                    'latitude': float(ll[cell, 0]), 'longitude': float(ll[cell, 1]), 'elevation_m': 0.,
                                    'observed_at': measured.isoformat(), 'available_at': (measured+timedelta(minutes=5)).isoformat(),
                                    'valid': True, 'quality': 1.})
                records.append({'observation_id': f'{name}-{age}-{cell}-sonde', 'source': 'radiosonde',
                                'variable': 'temperature', 'value': float(pr[cell, PRESSURE_HPA.index(500), 0]),
                                'units': 'K', 'pressure_pa': 50000., 'latitude': float(ll[cell, 0]),
                                'longitude': float(ll[cell, 1]), 'observed_at': measured.isoformat(),
                                'available_at': (measured+timedelta(minutes=5)).isoformat(), 'valid': True})
        import json
        (root/f'{name}.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
        rows.append({'id': name, 'split': split, 'issue_time': issue.isoformat(),
                     'observations': reference(root, root/f'{name}.jsonl'), 'targets': reference(root, root/f'{name}.npz'),
                     'provenance': {'observations': 'SYNTHETIC analytic fixture, not station records',
                                    'targets': 'SYNTHETIC analytic fields, not ERA5',
                                    'availability': 'SYNTHETIC fixture schedule, not measured latency',
                                    'target_operator': 'analytic values at cell centres'}})
    manifest = {'schema': 'global-weather-dataset-1', 'data_kind': 'synthetic',
                'mesh_level': mesh_level, 'grid_fingerprint': grid.fingerprint, 'step_hours': 3,
                'horizon_hours': horizon_hours, 'pressure_hpa': list(PRESSURE_HPA),
                'registry': reference(root, root/'registry.json'), 'static': reference(root, root/'static.npz'),
                'normalization': None, 'samples': rows, 'license': 'synthetic test fixture; no weather observations'}
    atomic_json(root/'unscaled.json', manifest)
    fit_normalization(root/'unscaled.json', root/'dataset.json')
    return root/'dataset.json'
