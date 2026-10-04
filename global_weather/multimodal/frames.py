"""Causal calibrated scenes. Native DN, RGB and assumed geometry are not inputs."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch

SOURCES = ('electro_l', 'arktika_m', 'meteor_msu_mr', 'meteor_mtvza')
QUANTITIES = {'brightness_temperature': {'K'}, 'reflectance': {'1'},
              'spectral_radiance': {'W m-2 sr-1 um-1', 'mW m-2 sr-1 (cm-1)-1'}}


def utc(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Нужна временная метка с часовым поясом.')
    return value.astimezone(timezone.utc)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sensor_registry(value):
    if not isinstance(value, dict) or value.get('schema') != 'global-weather-multimodal-1':
        raise ValueError('Неизвестная схема многоканальных входов.')
    if set(value) != {'schema', 'sensors'} or not isinstance(value['sensors'], dict) or not 1 <= len(value['sensors']) <= 16:
        raise ValueError('Требуется реестр из 1–16 приборов.')
    variables = set()
    for name, spec in value['sensors'].items():
        if not re.fullmatch('[a-z][a-z0-9_]{0,47}', name):
            raise ValueError('Недопустимый идентификатор прибора.')
        required = {'source', 'platform', 'instrument', 'encoder', 'temporal', 'projection', 'channels'}
        if set(spec) != required or spec['source'] not in SOURCES or not spec['platform'] or not spec['instrument']:
            raise ValueError('Не определён физический прибор.')
        if spec['encoder'] not in ('unet', 'microwave') or spec['temporal'] not in ('registered', 'events'):
            raise ValueError('Неизвестная структура кодировщика.')
        if spec['projection'] not in ('bounded-centre', 'gaussian'):
            raise ValueError('Неизвестный оператор переноса.')
        if spec['source'] == 'meteor_mtvza' and (spec['encoder'] != 'microwave' or spec['temporal'] != 'events'):
            raise ValueError('МТВЗА требует отдельного событийного кодировщика.')
        if not isinstance(spec['channels'], list) or not 1 <= len(spec['channels']) <= 64:
            raise ValueError('Недопустимое число физических каналов.')
        ids = set()
        for ch in spec['channels']:
            if set(ch) != {'id', 'variable', 'quantity', 'units', 'calibration'}:
                raise ValueError('У канала нужны ID, величина, единицы и версия калибровки.')
            if not all(isinstance(x, str) and x for x in ch.values()):
                raise ValueError('Пустые метаданные канала.')
            if ch['quantity'] not in QUANTITIES or ch['units'] not in QUANTITIES[ch['quantity']]:
                raise ValueError('Не допускаются цифровые отсчёты или неизвестные единицы.')
            if ch['id'] in ids or ch['variable'] in variables:
                raise ValueError('Повторный канал либо неоднозначная норма.')
            ids.add(ch['id']); variables.add(ch['variable'])
    return value


@dataclass
class Scene:
    sensor: str
    values: torch.Tensor          # [C,H,W], z-score; raw only inside fitting
    valid: torch.Tensor
    geometry: torch.Tensor        # [8,H,W]: xyz, view, age, size, solar cosine and solar validity
    cells: torch.Tensor           # [links], target cell
    pixels: torch.Tensor          # [links], source flat pixel
    weights: torch.Tensor         # [links], positive geometry weights
    identity: str
    observed: datetime
    available: datetime
    grid_id: str
    coordinates: np.ndarray       # [H,W,2], comparison for registered sequences

    def to(self, device):
        return Scene(**{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in vars(self).items()})


def _arrays(path):
    # Reuse bounded NPZ reader: refuses objects, duplicate members and inflated headers.
    from ..pipeline.io import read_arrays
    return read_arrays(path,limit=128*1024**2)


def read_scene(path, spec, *, kind, issue, max_pixels=65536):
    a = _arrays(path)
    required = {'values', 'valid', 'latitude', 'longitude', 'view_zenith_deg',
                'footprint_major_km', 'footprint_minor_km', 'footprint_azimuth_deg',
                'observed_utc_s', 'solar_zenith_deg', 'metadata'}
    if set(a) != required:
        raise ValueError('Неполные физические массивы спутникового кадра.')
    from ..pipeline.io import parse_json
    meta = parse_json(str(a['metadata']))
    if meta.get('schema') != 'satellite-scene-1' or meta.get('data_kind') != kind:
        raise ValueError('Схема или происхождение кадра не совпадают.')
    for key in ('source', 'platform', 'instrument', 'channels'):
        if meta.get(key) != spec[key]:
            raise ValueError('Платформа, порядок каналов, единицы или калибровка не совпадают.')
    if not meta.get('grid_id') or not meta.get('geometry_reference') or not meta.get('availability_reference'):
        raise ValueError('Не подтверждены сетка, геометрия или готовность.')
    hashes = meta.get('source_sha256')
    if not isinstance(hashes, list) or not hashes or any(not re.fullmatch('[0-9a-f]{64}', h) for h in hashes):
        raise ValueError('Нужны хэши физических исходников.')
    if spec['projection'] == 'gaussian' and meta.get('footprint_definition') != 'gaussian_fwhm':
        raise ValueError('Для антенного оператора нужны измеренные оси FWHM.')
    x, m = a['values'], a['valid']
    if x.ndim != 3 or x.shape[0] != len(spec['channels']) or m.shape != x.shape or m.dtype != bool or x.dtype.kind != 'f':
        raise ValueError('Ожидаются значения и маски [канал,строка,столбец].')
    shape = x.shape[1:]
    if not 1 <= np.prod(shape) <= max_pixels or spec['encoder'] == 'unet' and min(shape) < 8:
        raise ValueError('Недопустимый размер кадра; разбейте его на геопривязанные фрагменты.')
    if any(a[k].shape != shape or a[k].dtype.kind not in 'fi' for k in required - {'values','valid','metadata'}):
        raise ValueError('Геометрия и времена должны иметь размер растра.')
    supported = m.any(0)
    if not np.isfinite(x[m]).all() or any(not np.isfinite(a[k][supported]).all() for k in required - {'values','valid','metadata','solar_zenith_deg'}):
        raise ValueError('Неизвестное пригодное значение, геометрия или время.')
    ready = utc(meta['available_at'])
    measured = a['observed_utc_s']
    if np.any(measured[supported] > ready.timestamp()):
        raise ValueError('Кадр объявлен готовым раньше измерения.')
    lat, lon, view = a['latitude'], a['longitude'], a['view_zenith_deg']
    major, minor = a['footprint_major_km'], a['footprint_minor_km']
    az, sun = a['footprint_azimuth_deg'], a['solar_zenith_deg']
    bad = (np.abs(lat) > 90) | (np.abs(lon) > 180) | (view < 0) | (view >= 90)
    bad |= (minor <= 0) | (major < minor) | (major > 5000) | (np.abs(az) > 360) | (sun < 0) | (sun > 180)
    if np.any(bad & supported):
        raise ValueError('Геометрия за пределами физического контракта.')
    use = (measured > issue.timestamp()-43200) & (measured <= issue.timestamp()) & (ready <= issue)
    m = m & use[None]
    for i, ch in enumerate(spec['channels']):
        if ch['quantity'] == 'reflectance':
            m[i] &= np.isfinite(sun) & (sun < 90)
        elif ch['quantity'] == 'brightness_temperature' and np.any((x[i] <= 0) & m[i]):
            raise ValueError('Яркостная температура в K неположительна.')
    a = dict(a, valid=m)
    return a, meta


def project_links(a, grid, mode, *, max_links=1000000):
    """Positive encoding weights, not conservative remapping of physical fields.

    Gaussian mode uses cell-centre area quadrature. Reject under-resolved footprints
    rather than masquerade an antenna-average as a resolved pixel.
    """
    from ..grid import unit_xyz, EARTH_RADIUS_M
    keep = np.flatnonzero(a['valid'].any(0))
    lat, lon = a['latitude'].ravel()[keep], a['longitude'].ravel()[keep]
    xyz = unit_xyz(lat, lon)
    if mode == 'bounded-centre':
        cells = grid.tree.query(xyz)[1].astype(np.int64)
        if np.any(a['footprint_major_km'].ravel()[keep] > np.sqrt(grid.areas_m2[cells])/1000):
            raise ValueError('Размер пятна превышает ячейку. Нужен антенный оператор.')
        return cells, keep, np.ones(len(keep))
    all_cells, all_pixels, all_weights = [], [], []
    for index, point, la, lo in zip(keep, xyz, np.deg2rad(lat), np.deg2rad(lon)):
        major, minor = a['footprint_major_km'].flat[index]*1000, a['footprint_minor_km'].flat[index]*1000
        sigma_a, sigma_b = major/2.354820045, minor/2.354820045
        radius = min(3*sigma_a/EARTH_RADIUS_M, np.pi)
        neighbours = grid.tree.query_ball_point(point, 2*np.sin(radius/2))
        if len(neighbours) < 3:
            raise ValueError('Антенное пятно не разрешено квадратурой сетки. Нужна более подробная сетка.')
        ids = np.array(neighbours, dtype=np.int64)
        vectors = grid.xyz[ids]
        dot = np.clip(vectors@point, -1, 1); angle = np.arccos(dot)
        tangent = vectors - dot[:,None]*point
        tangent *= (angle/np.maximum(np.linalg.norm(tangent,axis=1),1e-15))[:,None]*EARTH_RADIUS_M
        east = np.array([-np.sin(lo), np.cos(lo), 0.])
        north = np.array([-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)])
        e, n = tangent@east, tangent@north
        az = np.deg2rad(a['footprint_azimuth_deg'].flat[index])
        along, across = n*np.cos(az)+e*np.sin(az), e*np.cos(az)-n*np.sin(az)
        exponent = (along/sigma_a)**2 + (across/sigma_b)**2
        retain = exponent <= 9
        ids, exponent = ids[retain], exponent[retain]
        if len(ids) < 3:
            raise ValueError('Недостаточно точек в эллиптическом пятне.')
        w = np.exp(-.5*exponent)*grid.areas_m2[ids]; w /= w.sum()
        all_cells.extend(ids); all_pixels.extend([index]*len(ids)); all_weights.extend(w)
        if len(all_cells) > max_links:
            raise ValueError('Превышен предел связей антенны.')
    return np.array(all_cells, np.int64), np.array(all_pixels,np.int64), np.array(all_weights)


def load_scene(path, spec, name, grid, norm, issue, kind, checksum):
    a, meta = read_scene(path, spec, kind=kind, issue=issue)
    if not a['valid'].any():
        return None
    x = a['values'].copy(); m = a['valid']
    for i, ch in enumerate(spec['channels']):
        mu, sd = norm.get(ch['variable'], ch['units']).at()
        x[i] = np.where(m[i], (x[i]-mu)/sd, 0.)
    from ..grid import unit_xyz
    valid = m.any(0)
    lat, lon = (np.where(valid, a[k], 0.) for k in ('latitude','longitude'))
    pos = unit_xyz(lat, lon).transpose(2,0,1)
    age = np.where(valid, (issue.timestamp()-a['observed_utc_s'])/43200, 0.)
    geom = np.concatenate((pos, np.stack((np.cos(np.deg2rad(a['view_zenith_deg'])), age,
            a['footprint_major_km']/100., np.where(np.isfinite(a['solar_zenith_deg']),np.cos(np.deg2rad(a['solar_zenith_deg'])),0.),
            np.isfinite(a['solar_zenith_deg']).astype(float)))))
    geom = np.where(valid[None],geom,0.)
    if not np.isfinite(x).all() or not np.isfinite(geom).all():
        raise ValueError('Неконечные нормализованные входы.')
    cells, pixels, weights = project_links(a, grid, spec['projection'])
    observed = datetime.fromtimestamp(float(a['observed_utc_s'][valid].max()), timezone.utc)
    identity = json.dumps(['raster:'+name,checksum,0,meta['available_at']],separators=(',',':'))
    return Scene(name,torch.tensor(x,dtype=torch.float32),torch.tensor(m),torch.tensor(geom,dtype=torch.float32),
                 torch.tensor(cells),torch.tensor(pixels),torch.tensor(weights,dtype=torch.float32),
                 identity,observed,utc(meta['available_at']),meta['grid_id'],np.stack((lat,lon),-1))
