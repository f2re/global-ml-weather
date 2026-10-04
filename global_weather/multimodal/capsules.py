"""Local adapters for calibrated producer files. Input files are never modified."""
import json
from pathlib import Path
import numpy as np
from .frames import fingerprint, sensor_registry, utc


def from_capsules(job_path, output):
    """Combine co-registered IR channels; preserve nominal acquisition precision."""
    from ..pipeline.io import read_json, write_arrays, local_path, sha256
    from ..connectors.raster_bridge import read_capsule
    job = read_json(job_path)
    required = {'schema', 'sensor', 'sensor_spec', 'capsules', 'data_kind'}
    if not isinstance(job, dict) or set(job) != required or job['schema'] != 'capsule-scene-job-1':
        raise ValueError('Неверный контракт группы капсул.')
    if job['data_kind'] not in ('real', 'synthetic'):
        raise ValueError('Укажите происхождение данных.')
    spec = job['sensor_spec']
    sensor_registry({'schema': 'global-weather-multimodal-1', 'sensors': {job['sensor']: spec}})
    if spec['projection'] != 'bounded-centre':
        raise ValueError('Один диаметр капсулы не определяет гауссову диаграмму антенны.')
    if any(ch['quantity'] == 'reflectance' for ch in spec['channels']):
        raise ValueError('Для отражательных каналов нужен отдельный источник солнечной геометрии.')
    if not isinstance(job['capsules'], list) or len(job['capsules']) != len(spec['channels']):
        raise ValueError('Нужна капсула для каждого канала.')
    root = Path(job_path).absolute().parent
    metas, fields, pinned = [], [], []
    geometry = ('latitude', 'longitude', 'view_zenith_deg', 'footprint_km')
    for entry, channel in zip(job['capsules'], spec['channels']):
        if not isinstance(entry, dict) or set(entry) != {'path', 'manifest_sha256'}:
            raise ValueError('Укажите путь и хэш манифеста капсулы.')
        directory = local_path(root, entry['path'])
        manifest = directory / 'manifest.json'
        if sha256(manifest) != entry['manifest_sha256']:
            raise ValueError('Изменился манифест капсулы.')
        meta, data = read_capsule(directory)
        if meta.get('data_kind') not in (None, job['data_kind']):
            raise ValueError('Происхождение капсулы не совпадает.')
        if any(meta.get(k) != spec[k] for k in ('source', 'platform', 'instrument')):
            raise ValueError('Не совпадает физический прибор.')
        pairs = [('channel_id', 'id'), ('quantity', 'quantity'), ('units', 'units'), ('calibration_reference', 'calibration')]
        if any(meta.get(a) != channel[b] for a, b in pairs):
            raise ValueError('Не совпадают канал, единицы или калибровка.')
        if not set(geometry).issubset(data) or not meta.get('geometry_reference'):
            raise ValueError('Не предоставлена проверенная геометрия.')
        if fields:
            if any(meta.get(k) != metas[0].get(k) for k in ('crs', 'transform', 'observed_at')):
                raise ValueError('Каналы имеют разные сетки или сроки.')
            if any(data[k].shape != fields[0][k].shape or not np.allclose(data[k], fields[0][k], equal_nan=True) for k in geometry):
                raise ValueError('Геометрия или пятна каналов не совпадают.')
        metas.append(meta)
        fields.append(data)
        pinned.extend([(manifest, entry['manifest_sha256']), (directory / 'pixels.npz', meta['arrays_sha256'])])
    first = fields[0]
    shape = first['values'].shape
    metadata = {k: spec[k] for k in ('source', 'platform', 'instrument', 'channels')}
    metadata.update(schema='satellite-scene-1', data_kind=job['data_kind'],
                    grid_id=fingerprint({'crs': metas[0]['crs'], 'transform': metas[0]['transform'], 'shape': shape}),
                    geometry_reference=metas[0]['geometry_reference'], source_sha256=[h for _, h in pinned],
                    available_at=max(utc(m['available_at']) for m in metas).isoformat(),
                    availability_reference='Pinned producer manifests; original readiness preserved',
                    time_representation='nominal_scene_time; pixel scan times not reconstructed',
                    footprint_definition='bounding_diameter; not Gaussian FWHM')
    result = {k: first[k] for k in ('latitude', 'longitude', 'view_zenith_deg')}
    result.update(values=np.stack([a['values'] for a in fields]), valid=np.stack([a['valid'] for a in fields]),
                  footprint_major_km=first['footprint_km'], footprint_minor_km=first['footprint_km'],
                  footprint_azimuth_deg=np.zeros(shape), solar_zenith_deg=np.full(shape, np.nan),
                  observed_utc_s=np.full(shape, utc(metas[0]['observed_at']).timestamp()),
                  metadata=np.array(json.dumps(metadata, allow_nan=False)))
    if any(sha256(path) != checksum for path, checksum in pinned):
        raise ValueError('Исходник изменился при чтении.')
    write_arrays(Path(output), **result)
    return {'status': 'physical_scene_prepared', 'sha256': sha256(output), 'time_precision': metadata['time_representation']}


def tile_scene(input_path, output_directory, *, tile_size=128):
    """Non-overlapping index selection. Small final strips are merged, not duplicated."""
    from ..pipeline.io import read_arrays, write_arrays, parse_json, atomic_json, reference, sha256
    if type(tile_size) is not int or not 8 <= tile_size <= 256:
        raise ValueError('Размер фрагмента должен быть от 8 до 256.')
    data = read_arrays(input_path, limit=512*1024**2)
    meta = parse_json(str(data['metadata']))
    if meta.get('schema') != 'satellite-scene-1' or data['values'].ndim != 3:
        raise ValueError('Нужен подготовленный спутниковый кадр.')
    height, width = data['values'].shape[1:]
    if min(height, width) < 8:
        raise ValueError('Изображение должно иметь не менее восьми строк и столбцов.')
    def bounds(size):
        edges = list(range(0, size, tile_size)) + [size]
        if len(edges) > 2 and edges[-1] - edges[-2] < 8:
            edges.pop(-2)
        return list(zip(edges[:-1], edges[1:]))
    root = Path(output_directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    outputs = []
    source = sha256(input_path)
    for y0, y1 in bounds(height):
        for x0, x1 in bounds(width):
            if (y1-y0)*(x1-x0) > 65536:
                raise ValueError('Краевой фрагмент превышает предел; уменьшите tile-size.')
            if not data['valid'][:, y0:y1, x0:x1].any():
                continue
            tile = {k: (a[:, y0:y1, x0:x1] if k in ('values', 'valid') else a[y0:y1, x0:x1])
                    for k, a in data.items() if k != 'metadata'}
            metadata = dict(meta, grid_id=fingerprint([meta['grid_id'], y0, y1, x0, x1]),
                            source_sha256=list(meta['source_sha256'])+[source],
                            tile_bounds=[y0, y1, x0, x1], parent_scene_sha256=source,
                            tile_operator='non_overlapping_index_selection; no physical interpolation')
            tile['metadata'] = np.array(json.dumps(metadata, allow_nan=False))
            path = root / f'tile-{y0:05d}-{x0:05d}.npz'
            write_arrays(path, **tile)
            outputs.append(reference(root, path))
    if sha256(input_path) != source:
        raise ValueError('Исходный кадр изменился при разбиении.')
    report = {'schema': 'scene-tiles-1', 'tiles': outputs, 'source_sha256': source, 'spatial_interpolation': False}
    atomic_json(root / 'tiles.json', report)
    return report
