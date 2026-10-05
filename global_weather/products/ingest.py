"""Derived products enter the existing observation encoder as labelled features.

Atmospheric records retain their 12h history. Only explicitly registered slow
surface products can have a longer, frozen age policy. No future data injection.
"""
from datetime import timedelta
import json
from pathlib import Path
import numpy as np
from .catalog import CATALOG, variable_spec
from .core import utc, require_hash, validate_metadata, canonical, digest
from .io import load_product, exclusive_bytes, arrays, sha256, regular

SATELLITE_SOURCES = ('electro_l', 'arktika_m', 'meteor_msu_mr', 'meteor_mtvza')


def check_variable(var):
    name = var.product
    if name not in CATALOG:
        raise ValueError('Неизвестная спутниковая продукция.')
    p = CATALOG[name]
    if (var.units != p.units or var.vertical != p.vertical or var.method not in p.methods
            or var.source not in SATELLITE_SOURCES or not var.platform or var.channel_id is not None):
        raise ValueError('Продукт, метод, единицы, платформа и положение в колонке должны совпадать.')
    if name == 'soil_moisture_surface':
        if type(var.product_depth_m) not in (float,int) or not 0 < var.product_depth_m <= .1:
            raise ValueError('Реестр должен фиксировать глубину поверхностного слоя почвы.')
    elif var.product_depth_m is not None:
        raise ValueError('Глубина почвы не относится к этому продукту.')
    if type(var.history_hours) is not int or not 0 < var.history_hours <= p.max_age_hours:
        raise ValueError('Возраст продукции превышает предел зарегистрированного метода.')


def record_history(registry, variable):
    v = registry[variable]
    if isinstance(v, dict):
        return v.get('history_hours', 12) if v.get('product') else 12
    return v.history_hours if v.product else 12


def maximum_history(registry):
    # Validate limits before using a registry to alter the training split boundary.
    from ..observations import Variable
    variables = {k: Variable(**v) if isinstance(v, dict) else v for k, v in registry.items()}
    return max([12]+[record_history(variables, k) for k in variables])


def evidence_history(identity):
    parts = json.loads(identity)
    if len(parts) == 5 and isinstance(parts[4], dict):
        extra = parts[4]
        if extra.get('product') not in CATALOG: raise ValueError('Неизвестный продукт в памяти анализа.')
        age = extra.get('history_hours')
        if type(age) is not int or not 0 < age <= CATALOG[extra['product']].max_age_hours:
            raise ValueError('Неверный срок хранения продукта.')
        return age
    return 12


def check_record(record, var, issue):
    check_variable(var)
    meta = record.get('derivation')
    if not isinstance(meta, dict): raise ValueError('У продукции нет происхождения.')
    validate_metadata(meta, name=var.product, method=var.method, issue_time=issue)
    for k in ('source', 'platform', 'observed_at', 'available_at'):
        if record.get(k) != meta.get(k): raise ValueError('Метаданные продукта не совпадают с записью наблюдения.')
    if record.get('source') != var.source or record.get('platform') != var.platform:
        raise ValueError('Несовместимый поставщик продукции.')
    require_hash(record.get('product_sha256'))
    require_hash(record.get('geometry_sha256'))
    if record.get('qc') != 0 or record.get('units') != CATALOG[var.product].units:
        raise ValueError('Продукт не прошёл проверку качества.')
    x = record.get('value')
    lo, hi = CATALOG[var.product].limits
    if type(x) not in (int, float) or not np.isfinite(x) or not lo <= x <= hi:
        raise ValueError('Физически недопустимое значение продукции.')
    if not utc(issue)-timedelta(hours=var.history_hours) < utc(record['observed_at']) <= utc(issue):
        raise ValueError('Продукт устарел.')
    lower = utc(issue)-timedelta(hours=max(12,var.history_hours))
    for dependency in meta['dependencies']:
        start = dependency.get('temporal_support',{}).get('start',dependency['observed_at'])
        if utc(start) <= lower:
            raise ValueError('Вход продукции выходит за окно, учтённое при разделении выборки.')
    if var.product == 'soil_moisture_surface':
        a = meta.get('attributes', {})
        if a.get('depth_top_m') != 0 or a.get('depth_bottom_m') != var.product_depth_m:
            raise ValueError('Не указана глубина поверхностной влажности почвы.')
        require_hash(a.get('lut_sha256'))
    return meta


def export_product(product_path, geometry_path, output, *, variable=None, history_hours=None, max_records=100000, max_output_bytes=64*1024**2):
    """All valid pixels or explicit failure, no silent sampling or fake geometry.

    Geometry NPZ: latitude, longitude, view_zenith_deg, footprint_km, grid_id.
    It must be independently prepared on exactly the product raster grid.
    """
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 256*1024**2:
        raise ValueError('Неверный предел размера экспорта.')
    regular(product_path); regular(geometry_path)
    product_hash = sha256(product_path); geometry_hash = sha256(geometry_path)
    p = load_product(product_path); g = arrays(geometry_path)
    if set(g) != {'latitude','longitude','view_zenith_deg','footprint_km','grid_id'}:
        raise ValueError('Нужны координаты, реальная геометрия и идентификатор сетки.')
    if str(g['grid_id']) != p.metadata['grid_id'] or any(g[k].shape != p.values.shape for k in g if k != 'grid_id'):
        raise ValueError('Геометрия и продукция не совмещены.')
    if p.metadata['source'] not in SATELLITE_SOURCES:
        raise ValueError('Источник не зарегистрирован.')
    var = variable or ':'.join([p.metadata['source'], p.metadata['platform'], p.name, p.method])
    registry = variable_spec(p.name, p.metadata['source'], p.metadata['platform'], history_hours=history_hours,
                             depth_bottom_m=p.metadata.get('attributes',{}).get('depth_bottom_m') if p.name=='soil_moisture_surface' else None,
                             method=p.method)
    from ..observations import Variable
    v = Variable(**registry); check_variable(v)
    count = int(p.valid.sum())
    if type(max_records) is not int or not 1 <= count <= max_records <= 1000000:
        raise ValueError('Нет пригодных пикселей или превышен предел; подвыборка не делается автоматически.')
    for k in g:
        if k != 'grid_id' and not np.isfinite(g[k][p.valid]).all(): raise ValueError('Неизвестная геометрия пригодного пикселя.')
    if ((np.abs(g['latitude'][p.valid]) > 90).any() or (np.abs(g['longitude'][p.valid]) > 180).any()
            or ((g['view_zenith_deg'][p.valid] < 0) | (g['view_zenith_deg'][p.valid] >= 90)).any()
            or (g['footprint_km'][p.valid] <= 0).any()):
        raise ValueError('Недопустимые координаты или геометрия.')
    if sha256(product_path) != product_hash or sha256(geometry_path) != geometry_hash:
        raise ValueError('Исходник изменился при чтении.')
    def write(stream):
        written = 0
        for index in np.ndindex(p.values.shape):
            if not p.valid[index]: continue
            stable = digest([p.metadata['source'],p.metadata['platform'],p.name,p.method,p.metadata['primary_sha256'],p.metadata['observed_at'],index])
            rec = dict(observation_id=stable, source=p.metadata['source'], platform=p.metadata['platform'],
                       variable=var, value=float(p.values[index]), units=registry['units'],
                       observed_at=p.metadata['observed_at'], available_at=p.metadata['available_at'],
                       valid=True, revision=0, quality=1., qc=0, derivation=p.metadata,
                       product_sha256=product_hash, geometry_sha256=geometry_hash,
                       uncertainty=None if not np.isfinite(p.uncertainty[index]) else float(p.uncertainty[index]))
            for k in ('latitude','longitude','view_zenith_deg','footprint_km'): rec[k] = float(g[k][index])
            check_record(rec, v, p.metadata['available_at'])
            encoded = (canonical(rec)+'\n').encode()
            written += len(encoded)
            if written > max_output_bytes:
                raise ValueError('Превышен размер экспорта; уменьшите область явным оператором.')
            stream.write(encoded)
        if sha256(product_path) != product_hash or sha256(geometry_path) != geometry_hash:
            raise ValueError('Исходник изменился при экспорте.')
    exclusive_bytes(output, write)
    return dict(status='derived_features_exported', records=count, registry={var: registry},
                sha256=sha256(output), data_kind=p.metadata['data_kind'],
                note='Нужны собственные фиксированные нормы и обучение с этими входами; качество не подтверждено.')
