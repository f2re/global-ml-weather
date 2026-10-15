"""Physical admission of satellite channels for R11 (schema satellite-channel-admission-1).

A radiometric passport (calibration, units, noise) lets a channel enter a
token store. That alone does not make the channel a physical constraint on
the state. This module admits a channel as a TARGET only when its admission
record supplies:

* the measured quantity in K and the channel frequency or wavenumber;
* the passport SHA256 that the token store recorded for the channel;
* NEdT and an observation error not smaller than NEdT;
* a clear-sky nadir temperature weighting function on the 37 model levels,
  plus the surface weight, from a named radiative-transfer computation;
* clear-sky screening, and a surface weight of at most 0.1, because the
  operator has no cloud and no emissivity model;
* train statistics with at least `minimum_train_count` values.

For an admitted target the observation operator is physical, not a free
network:

    Tb = sum_k w_k T_k + w_s T_s + b_c + sum_j beta_cj p_j,

where the weighting function w is fixed by the admission record. The bias b_c
and the coefficients beta_cj on predictors p_j are learned jointly with the
model, like variational bias correction (Dee 2004, ECMWF Workshop on
Assimilation of High Spectral Resolution Sounders; Auligne et al. 2007,
QJRMS 133, 631-642). The predictors are 1, the air-mass factor
1/cos(zenith) - 1, and the 1000-300 hPa and 200-50 hPa thicknesses.
Channels admitted only as inputs reach the encoder at the layers of their
weighting function; they are never targets.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..vertical import PRESSURE_HPA
from .tokens import PRESSURE_BANDS

SCHEMA = 'satellite-channel-admission-1'
MAX_TARGET_SURFACE_WEIGHT = 0.1
PREDICTORS = ('constant', 'air_mass', 'thickness_1000_300', 'thickness_200_50')
ROLES = ('input', 'target')


def _hex(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load(path) -> dict:
    """Read and check one admission file; raise on the first physical inconsistency."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Satellite admission must be a regular file.')
    payload = json.loads(path.read_text())
    if payload.get('schema') != SCHEMA:
        raise ValueError('Unknown satellite admission schema.')
    minimum = payload.get('minimum_train_count', 1000)
    if type(minimum) is not int or minimum < 2:
        raise ValueError('minimum_train_count must be an integer >= 2.')
    channels = {}
    levels = [float(p) for p in PRESSURE_HPA]
    for entry in payload.get('channels', []):
        name = entry.get('channel')
        if not isinstance(name, str) or ':' not in name or name in channels:
            raise ValueError('Each channel needs a unique "instrument:channel" name.')
        if entry.get('quantity') != 'brightness_temperature' or entry.get('units') != 'K':
            raise ValueError(f'{name}: only brightness temperature in K is admitted physically.')
        frequency, wavenumber = entry.get('central_frequency_ghz'), entry.get('central_wavenumber_cm-1')
        if not any(isinstance(x, (int, float)) and x > 0 for x in (frequency, wavenumber)):
            raise ValueError(f'{name}: a positive central frequency or wavenumber is required.')
        if not _hex(entry.get('passport_sha256')):
            raise ValueError(f'{name}: passport_sha256 must be a lowercase SHA256.')
        nedt, error = entry.get('noise_equivalent_dt_k'), entry.get('observation_error_k')
        if not (isinstance(nedt, (int, float)) and nedt > 0 and isinstance(error, (int, float))
                and math.isfinite(error) and error >= nedt):
            raise ValueError(f'{name}: observation error must be finite and not below NEdT > 0.')
        roles = entry.get('roles')
        if not isinstance(roles, list) or not roles or set(roles) - set(ROLES):
            raise ValueError(f'{name}: roles must be a non-empty subset of {ROLES}.')
        function = entry.get('weighting_function') or {}
        if function.get('pressure_hpa') != levels:
            raise ValueError(f'{name}: the weighting function must use exactly the 37 model levels.')
        weights = np.asarray(function.get('weights', []), float)
        surface = function.get('surface_weight')
        if (weights.shape != (37,) or not np.isfinite(weights).all() or (weights < 0).any()
                or not isinstance(surface, (int, float)) or not 0 <= surface <= 1):
            raise ValueError(f'{name}: weights must be 37 finite non-negative values plus a surface weight in [0, 1].')
        if abs(weights.sum() + surface - 1) > 1e-3:
            raise ValueError(f'{name}: atmospheric and surface weights must sum to 1.')
        if not isinstance(function.get('source'), str) or not function['source'].strip():
            raise ValueError(f'{name}: name the radiative-transfer computation of the weighting function.')
        if 'target' in roles:
            if surface > MAX_TARGET_SURFACE_WEIGHT:
                raise ValueError(f'{name}: surface-sensitive channel cannot be a target without an emissivity model.')
            if entry.get('clear_sky_screened') is not True:
                raise ValueError(f'{name}: a target channel requires clear-sky screening.')
        channels[name] = {'roles': sorted(roles), 'weights': weights, 'surface_weight': float(surface),
                          'observation_error_k': float(error), 'passport_sha256': entry['passport_sha256']}
    return {'schema': SCHEMA, 'minimum_train_count': minimum, 'channels': channels}


def check_against_stores(admission: dict, stores: list, statistics: dict) -> dict:
    """Every admitted channel must exist in a store with the same passport and enough train values."""
    passports = {}
    for store in stores:
        for channel, hashes in store.manifest.get('channel_passports', {}).items():
            passports.setdefault(channel, set()).update(hashes)
    report = {}
    for name, entry in admission['channels'].items():
        if name not in passports:
            raise ValueError(f'{name}: admitted channel is not present in any token store.')
        if passports[name] != {entry['passport_sha256']}:
            raise ValueError(f'{name}: store passports differ from the admitted passport.')
        stat = statistics.get(f'satellite_radiance|brightness_temperature|{name}')
        if stat is None or stat['count'] < admission['minimum_train_count']:
            raise ValueError(f'{name}: too few train values for admission.')
        report[name] = {'roles': entry['roles'], 'train_count': stat['count'],
                        'surface_weight': entry['surface_weight'],
                        'peak_pressure_hpa': float(PRESSURE_HPA[int(np.argmax(entry['weights']))])}
    return report


def channel_tables(admission: dict | None, channels: list[str]) -> dict:
    """Arrays indexed like the model channel embedding (0 = no channel).

    weights [C+1, 37], surface [C+1], physical [C+1] bool, target [C+1] bool,
    slots [C+1, bands + 3] encoder slot weights for admitted inputs.
    """
    size = max(1, len(channels)) + 1  # matches the model embedding, which keeps one channel row minimum
    weights = np.zeros((size, 37))
    surface = np.zeros(size)
    physical = np.zeros(size, bool)
    target = np.zeros(size, bool)
    slots = np.zeros((size, PRESSURE_BANDS + 3))
    if admission:
        centres = np.linspace(np.log(100000.), np.log(100.), PRESSURE_BANDS)
        width = centres[0] - centres[1]
        log_p = np.log(np.asarray(PRESSURE_HPA, float) * 100.)
        projection = np.clip(1 - np.abs(log_p[:, None] - centres[None]) / width, 0, 1)
        projection /= projection.sum(1, keepdims=True)
        for index, name in enumerate(channels, start=1):
            entry = admission['channels'].get(name)
            if entry is None:
                continue
            weights[index], surface[index], physical[index] = entry['weights'], entry['surface_weight'], True
            target[index] = 'target' in entry['roles']
            slots[index, :PRESSURE_BANDS] = entry['weights'] @ projection
            slots[index, PRESSURE_BANDS] = entry['surface_weight']
    return {'weights': weights, 'surface': surface, 'physical': physical, 'target': target, 'slots': slots}


def thickness_predictors(geopotential, pressure_pa):
    """1000-300 and 200-50 hPa thickness predictors, normalised to O(1) [.., 2]."""
    import torch
    p = np.asarray(pressure_pa, float) / 100.
    index = {level: int(np.argmin(np.abs(p - level))) for level in (1000, 300, 200, 50)}
    a = (geopotential[..., index[300]] - geopotential[..., index[1000]]) / 9.80665
    b = (geopotential[..., index[50]] - geopotential[..., index[200]]) / 9.80665
    return torch.stack(((a - 9000.) / 500., (b - 8500.) / 500.), -1)
