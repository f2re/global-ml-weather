"""Explicit reviewed GeoTIFF -> model JSONL bridge, without false DN=K fallback.

A review describes an upstream-produced physical or affine-calibrated raster.
SatDump CBOR is NOT calibrated by this module. Geometry must be independently
exported on the same raster pixels; constant viewing angles are not invented.
Large microwave footprints remain rejected until an antenna operator exists.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile
import numpy as np
from .project_bridge import local_file, fingerprint, load_json, publish_json, utc, iso, local_directory, read_product, output_path


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Нужна фактическая SHA256.')
    return value


def validate_native_binding(review, native, root, expected_source_hashes):
    """Bind a review to native metadata; never infer a platform from a filename.

    The reviewer documents the actual producer checkout/build separately. A
    source revision in a JSON is a claim, not cryptographic attestation.
    """
    source, platform = review['source'], review['platform']
    channel, instrument = review['channel_id'], review['instrument']
    if review['producer']['repository'] == 'f2re/arktika-worker':
        a = load_json(native)
        if (not isinstance(a, dict) or source != 'arktika_m' or platform not in ('ARCM1','ARCM2')
                or a.get('platform') != platform or str(a.get('channel')) != channel
                or a.get('id') != review.get('asset_id') or a.get('category') != 'channel'
                or a.get('time_assumed') is not False or instrument != 'MSU-GS/A'):
            raise ValueError('Рецензия не соответствует нативному каналу Арктики.')
        utc(a['time'])
        receipt = local_file(root/review['download_receipt'], root)
        sha = fingerprint(receipt)
        if sha != _sha(review['download_receipt_sha256']):
            raise ValueError('Журнал загрузки изменился.')
        r = load_json(receipt)
        # The raster can be a derived tile; receipt refers to the original data.
        original = local_file(root/review['original_raster'], root)
        original_sha = fingerprint(original)
        if (r.get('asset_id') != a['id'] or r.get('sha256') != original_sha
                or r.get('size') != original.stat().st_size
                or utc(r['time']) != utc(review['download_completed_at'])):
            raise ValueError('Нет согласованного журнала исходной загрузки.')
        expected_source_hashes.extend(((receipt,sha),(original,original_sha)))
    else:
        product = read_product(native)
        if product.get('type') != 'image' or product.get('instrument') != instrument:
            raise ValueError('Прибор в product.cbor не совпал с рецензией.')
        expected = {'meteor_msu_mr':'msu_mr','electro_l':'msu_gs','arktika_m':'msu_gs'}
        images = product.get('images')
        if (expected.get(source) != instrument or not isinstance(images,list)
                or channel not in [x.get('name') for x in images if isinstance(x,dict)]):
            raise ValueError('Канал/семейство источника не совпали с product.cbor.')
        ds = local_file(root/review['dataset_metadata'], root)
        sha = fingerprint(ds)
        if sha != _sha(review['dataset_metadata_sha256']): raise ValueError('dataset.json изменился.')
        dataset = load_json(ds)
        if (not isinstance(dataset,dict) or dataset.get('satellite') != platform
                or not isinstance(dataset.get('products'),list)
                or not any((ds.parent/name/'product.cbor') == native for name in dataset['products']
                           if isinstance(name,str) and '..' not in Path(name).parts)):
            raise ValueError('Платформа/продукт не совпали с dataset.json.')
        # Family is explicitly reviewed, never deduced from arbitrary filenames.
        if review.get('platform_family') != source:
            raise ValueError('Семейство платформы должно быть явно проверено.')
        if product.get('decode_quality',{}).get('status') == 'no_data':
            raise ValueError('Нет полных сканов прибора.')
        expected_source_hashes.append((ds,sha))


def export_geotiff(review_path, data_root, output, *, issue_time, max_records=65536):
    """One bounded raster, one physical channel, causal per-pixel timestamps.

    No overwrite. A partial channel uses its own finite/QC mask. Rejected pixels
    are counted, never turned into zero-valued observations. No interpolation.
    """
    if not isinstance(max_records, int) or not 1 <= max_records <= 1_000_000:
        raise ValueError('Допустимый предел: 1–1000000 пикселей.')
    try:
        import rasterio
        from rasterio.warp import transform
    except ImportError as exc:
        raise RuntimeError('Нужен rasterio: установите дополнение .[compat].') from exc
    review_digest = fingerprint(review_path)
    review = load_json(review_path)
    if not isinstance(review,dict): raise ValueError('Рецензия должна быть JSON-объектом.')
    if review.get('schema') != 'global-weather.physical-raster/1':
        raise ValueError('Неизвестная схема физического экспорта.')
    if review.get('validation_status') != 'reviewed' or review.get('data_kind') not in ('real','synthetic'):
        raise ValueError('Нужна явная проверка; synthetic не становится real.')
    for field in ('reviewer', 'calibration_reference', 'geometry_reference', 'time_reference', 'quality_reference', 'license'):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError('Не хватает основания проверки: '+field)
    producer = review.get('producer', {})
    allowed_producers = {'f2re/arktika-worker', 'f2re/SatDump'}
    if producer.get('repository') not in allowed_producers or not re.fullmatch('[0-9a-f]{40}', producer.get('revision','')):
        raise ValueError('Укажите фактический репозиторий и ревизию производителя.')
    native_hash = _sha(review.get('native_metadata_sha256'))
    prefix=review.get('observation_id_prefix')
    origin=review.get('pixel_origin')
    if not isinstance(prefix,str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}',prefix):
        raise ValueError('Нужен стабильный идентификатор исходной съёмки/сетки.')
    if not isinstance(origin,list) or len(origin)!=2 or any(type(v) is not int or v<0 for v in origin):
        raise ValueError('Нужно положение тайла [row,column] в исходной сетке.')
    if type(review.get('revision')) is not int or review['revision']<0:
        raise ValueError('Нужна неотрицательная ревизия физического продукта.')
    source = review.get('source'); platform = review.get('platform'); channel = review.get('channel_id')
    if source not in ('arktika_m','electro_l','meteor_msu_mr','meteor_mtvza'):
        raise ValueError('Неизвестный спутниковый источник.')
    if not all(isinstance(x,str) and re.fullmatch(r'[A-Za-z0-9_. -]{1,80}', x) for x in (platform, channel, review.get('variable'))):
        raise ValueError('Нужны явные безопасные обозначения платформы, канала и переменной.')
    if not isinstance(review.get('instrument'),str) or not review['instrument'].strip():
        raise ValueError('Нужен идентификатор прибора.')
    if source == 'meteor_mtvza':
        raise ValueError('microwave_antenna_operator_required: МТВЗА не заменяется точкой.')
    root = local_directory(data_root)
    def item(name, expected):
        p = local_file(root/review[name], root)
        if fingerprint(p) != _sha(expected): raise ValueError('Файл изменился: '+name)
        return p
    raster = item('raster', review.get('raster_sha256'))
    geom = item('geometry', review.get('geometry_sha256'))
    native = item('native_metadata', native_hash)
    source_hashes = [(raster,review['raster_sha256']),(geom,review['geometry_sha256']),(native,native_hash)]
    validate_native_binding(review, native, root, source_hashes)
    if raster.suffix.lower() not in ('.tif','.tiff') or geom.suffix != '.npz':
        raise ValueError('Поддерживаются только числовой GeoTIFF и геометрия NPZ.')
    status = review.get('calibration_status')
    if status not in ('metadata','declared','verified'):
        raise ValueError('Калибровка assumed/unknown запрещена.')
    quantity, units = review.get('quantity'), review.get('units')
    accepted = {'brightness_temperature': {'K'}, 'reflectance': {'1'},
                'spectral_radiance': {'W m-2 sr-1 um-1','mW m-2 sr-1 (cm-1)-1'}}
    if quantity not in accepted or units not in accepted[quantity]: raise ValueError('Несовместимые величина и единицы.')
    if source in ('arktika_m','electro_l') and channel in ('1','2','3') and quantity == 'brightness_temperature':
        raise ValueError('Видимые каналы МСУ-ГС не являются температурными.')
    if review.get('channel_mapping_verified') is not True: raise ValueError('Соответствие канала не подтверждено.')
    for name in ('scale','offset'):
        if type(review.get(name)) not in (int,float) or not math.isfinite(review[name]): raise ValueError('Неверная шкала.')
    if review['scale'] <= 0: raise ValueError('Шкала должна быть положительной.')
    ready = utc(review['available_at']); issue = utc(issue_time)
    if ready > issue: raise ValueError('Продукт не был готов к моменту выпуска.')
    if ready < utc(review['download_completed_at']): raise ValueError('Обработка не могла завершиться до получения данных.')
    with zipfile.ZipFile(geom) as z:
        if len(z.infolist()) > 12 or len(set(z.namelist())) != len(z.namelist()) or sum(x.file_size for x in z.infolist()) > 128*1024*1024:
            raise ValueError('Превышен предел геометрии NPZ.')
    for suffix in ('.msk','.aux.xml','.ovr'):
        if Path(str(raster)+suffix).exists():
            raise ValueError('Нужен автономный GeoTIFF: внешние маски перенесите в проверенную геометрию.')
    with rasterio.Env(GDAL_PAM_ENABLED=False, GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', PROJ_NETWORK='OFF'), \
            rasterio.open(raster, driver='GTiff') as ds, np.load(geom, allow_pickle=False) as geometry:
        if ds.driver != 'GTiff' or ds.count != 1 or ds.crs is None or ds.width*ds.height > max_records:
            raise ValueError('Нужен ограниченный одноканальный GeoTIFF с CRS; подготовьте тайл явно.')
        shape = (ds.height, ds.width)
        required = ('observed_at_unix','view_zenith_deg','footprint_km','valid')
        for key in required:
            if key not in geometry or geometry[key].shape != shape: raise ValueError('Геометрия должна совпадать с каждым пикселем: '+key)
        if geometry['valid'].dtype != np.bool_: raise ValueError('valid должен быть Boolean.')
        values = ds.read(1, masked=True)
        field = np.asarray(values,dtype=np.float64)*review['scale']+review['offset']
        rr, cc = np.indices(shape)
        xx, yy = rasterio.transform.xy(ds.transform, rr.ravel(), cc.ravel())
        longitude, latitude = transform(ds.crs, 'EPSG:4326', xx, yy)
        lat, lon = np.reshape(latitude,shape), np.reshape(longitude,shape)
        time, angle, footprint = (np.asarray(geometry[k],dtype=float) for k in required[:3])
        if 'grid_transform' not in geometry or not np.allclose(geometry['grid_transform'],np.array(tuple(ds.transform)[:6]),rtol=0,atol=1e-10):
            raise ValueError('Привязка геометрии и растра различается.')
        if 'grid_crs' not in geometry or str(geometry['grid_crs']) != ds.crs.to_string():
            raise ValueError('CRS геометрии и растра различается.')
        valid = ~np.ma.getmaskarray(values) & geometry['valid'] & np.isfinite(field)
        for a in (lat,lon,time,angle,footprint): valid &= np.isfinite(a)
        valid &= (abs(lat)<=90) & (abs(lon)<=180) & (angle>=0) & (angle<90) & (footprint>0)
        if (valid & (time>ready.timestamp())).any(): raise ValueError('Время наблюдения позже готовности продукта.')
        valid &= (time>=(issue-timedelta(hours=12)).timestamp()) & (time<=issue.timestamp())
        # Model contract uses the half-open window (t-12h,t].
        valid &= time>(issue-timedelta(hours=12)).timestamp()
        if quantity == 'reflectance':
            if 'solar_zenith_deg' not in geometry or geometry['solar_zenith_deg'].shape != shape:
                raise ValueError('Для отражения нужна солнечная геометрия каждого пикселя.')
            solar = np.asarray(geometry['solar_zenith_deg'],dtype=float)
            valid &= np.isfinite(solar) & (solar>=0) & (solar<90)
        bounds=review.get('physical_valid_range')
        if not isinstance(bounds,list) or len(bounds)!=2 or not all(type(v) in (int,float) and math.isfinite(v) for v in bounds) or bounds[0]>=bounds[1]:
            raise ValueError('Нужен физически обоснованный диапазон пригодности.')
        valid &= (field>=bounds[0]) & (field<=bounds[1])
        rows=[]
        calibration_id = review['calibration_reference']
        for y,x in zip(*np.where(valid)):
            observed = iso(datetime.fromtimestamp(float(time[y,x]),timezone.utc))
            row = dict(observation_id=f"{prefix}:{channel}:{int(y)+origin[0]}:{int(x)+origin[1]}",source=source,
                       platform=platform,channel_id=channel,variable=review['variable'],value=float(field[y,x]),units=units,
                       latitude=float(lat[y,x]),longitude=float(lon[y,x]),observed_at=observed,available_at=iso(ready),
                       valid=True,quality=1.0,revision=review['revision'],footprint_km=float(footprint[y,x]),view_zenith_deg=float(angle[y,x]),
                       provenance=dict(producer=producer,native_metadata_sha256=native_hash,data_kind=review['data_kind']),
                       radiometry=dict(instrument=review['instrument'],quantity=quantity,units=units,
                                       physical_channel_ids=[channel],calibration_id=calibration_id,channel_mapping_verified=True))
            if quantity=='reflectance': row['solar_zenith_deg']=float(solar[y,x])
            rows.append(row)
    for p, expected in source_hashes:
        if fingerprint(p)!=expected: raise ValueError('Вход изменился во время обработки.')
    if fingerprint(review_path) != review_digest: raise ValueError('Рецензия изменилась.')
    if not rows: raise ValueError('Нет пригодных доступных наблюдений; пустой набор не создаётся.')
    out=output_path(output)
    if out.is_relative_to(root): raise ValueError('Выход должен быть вне каталога поставщика.')
    if out.exists() or Path(str(out)+'.manifest.json').exists(): raise FileExistsError('Выход уже существует.')
    out.parent.mkdir(parents=True,exist_ok=True)
    if any(p.is_symlink() for p in (out,*out.parents)): raise ValueError('Символьная ссылка выхода.')
    fd,temp=tempfile.mkstemp(prefix='.physical-',dir=out.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.link(temp,out)
    finally: os.unlink(temp)
    summary=dict(schema='global-weather.physical-export/1',records=len(rows),rejected_pixels=int(np.size(valid)-len(rows)),
                 data_kind=review['data_kind'],review_sha256=review_digest,jsonl_sha256=fingerprint(out),
                 status='physical_contract_exported',training_ready=False,
                 remaining=['normalization_registry','observation_operator','independent_validation'])
    publish_json(str(out)+'.manifest.json',summary)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('review','data-root','output','issue-time'): p.add_argument('--'+name,required=True)
    p.add_argument('--max-records',type=int,default=65536)
    a=p.parse_args(argv)
    print(json.dumps(export_geotiff(a.review,a.data_root,a.output,issue_time=a.issue_time,max_records=a.max_records),ensure_ascii=False))


if __name__=='__main__':main()
