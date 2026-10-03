"""Physical rasters -> bounded, provenance-bearing arrays -> observation JSONL.

No resampling, channel invention or calibration inference. Geometry missing in a
native product remains missing; an explicit, reviewed geometry supplement is needed
before point-token export. Microwave footprint integration is NOT implemented here.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .ecosystem import regular, child, read_json, sha256, utc, iso


def _dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _new_directory(out):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink(): raise FileExistsError('Output exists; it is never overwritten.')
    if any(p.is_symlink() for p in out.parents): raise ValueError('Symlink output parent.')
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def import_raster(spec, out, *, available_at, availability_reference, geometry=None, max_pixels=2_000_000):
    try:
        import rasterio
        from rasterio.warp import transform
    except ImportError as exc:
        raise RuntimeError('Install the optional ecosystem dependencies for physical raster import.') from exc
    ready, observed = utc(available_at), utc(spec['observed_at'])
    if not isinstance(availability_reference, str) or not availability_reference.strip():
        raise ValueError('Availability evidence reference is required; file mtime is not evidence.')
    if ready < observed or (spec.get('earliest_available_at') and ready < utc(spec['earliest_available_at'])):
        raise ValueError('Ready time precedes measurement or upstream download completion.')
    if type(max_pixels) is not int or not 1 <= max_pixels <= 8_000_000: raise ValueError('Invalid pixel limit.')
    out = _new_directory(out)
    files = [regular(spec['raster']), *[regular(p) for p in spec['inputs']]]
    if spec.get('quality'): files.append(regular(spec['quality']))
    if geometry is not None: files.append(regular(geometry))
    checksums = {str(p): sha256(p) for p in files}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_PAM_ENABLED='NO'):
        with rasterio.open(files[0]) as ds:
            if ds.driver != 'GTiff' or not ds.crs or ds.count != 1 or ds.width*ds.height > max_pixels:
                raise ValueError('Expected bounded, georeferenced single-band GeoTIFF.')
            if ds.units[0] not in (None, spec['units']): raise ValueError('Raster/manifest units conflict.')
            if spec['values_already_physical'] and (ds.scales != (1.,) or ds.offsets != (0.,)):
                raise ValueError('Processed values.tif must not be scaled twice.')
            if not spec['values_already_physical']:
                if ds.scales != (1.,) and not np.isclose(ds.scales[0], spec['scale']):
                    raise ValueError('Raster/asset scale conflict.')
                if ds.offsets != (0.,) and not np.isclose(ds.offsets[0], spec['offset']):
                    raise ValueError('Raster/asset offset conflict.')
            if ds.tags().get('time') and utc(ds.tags()['time']) != observed: raise ValueError('Raster/manifest time conflict.')
            raster = ds.read(1, masked=True).astype('float32')
            values = np.asarray(raster.data, dtype='float32')*spec['scale']+spec['offset']
            valid = ~np.ma.getmaskarray(raster) & np.isfinite(values)
            if spec.get('declared_nodata') is not None:
                declared_nodata = spec['declared_nodata']
                if ds.nodata is not None and ds.nodata != declared_nodata:
                    raise ValueError('Raster/asset nodata conflict.')
                valid &= np.asarray(raster.data) != declared_nodata
            # A negative Kelvin value is physically invalid. Do not clip extremes to training percentiles.
            valid &= values > 0
            yy, xx = np.indices(values.shape, dtype='float64')
            xx, yy = ds.transform*(xx+.5, yy+.5)
            lon, lat = transform(ds.crs, 'EPSG:4326', xx.ravel().tolist(), yy.ravel().tolist())
            lon, lat = np.asarray(lon).reshape(values.shape), np.asarray(lat).reshape(values.shape)
            valid &= np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
            lon = (lon+180)%360-180
            crs, affine = ds.crs.to_string(), list(ds.transform)[:6]
            if spec.get('quality'):
                with rasterio.open(spec['quality']) as qc:
                    if qc.count != 1 or (qc.height,qc.width) != values.shape or qc.crs != ds.crs or qc.transform != ds.transform:
                        raise ValueError('Quality raster is on another grid.')
                    flags = qc.read(1, masked=True)
                    valid &= ~np.ma.getmaskarray(flags) & (flags.data == 0)
    arrays = dict(values=np.where(valid,values,0).astype('float32'), valid=valid,
                  latitude=np.where(valid,lat,0), longitude=np.where(valid,lon,0))
    reasons = ['geometry_supplement_required']
    geo_ref = None
    if geometry is not None:
        geo = read_json(geometry)
        if set(geo)-{'reference','view_zenith_deg','footprint_km'} or not isinstance(geo.get('reference'),str) or not geo['reference'].strip():
            raise ValueError('Reviewed view geometry reference required.')
        for key in ('view_zenith_deg','footprint_km'):
            a = np.asarray(geo[key], dtype='float64')
            if a.shape not in ((), values.shape): raise ValueError('Geometry shape differs from raster.')
            a = np.broadcast_to(a, values.shape).copy()
            if not np.isfinite(a[valid]).all(): raise ValueError('Missing geometry at valid pixels.')
            if key == 'view_zenith_deg' and ((a[valid]<0)|(a[valid]>=90)).any(): raise ValueError('Invalid viewing angle.')
            if key == 'footprint_km' and (a[valid]<=0).any(): raise ValueError('Invalid native footprint.')
            arrays[key] = np.where(valid,a,0)
        geo_ref, reasons = geo['reference'], []
    if not valid.any(): raise ValueError('No physical pixels survive QC.')
    manifest = {k:v for k,v in spec.items() if k not in ('raster','quality','inputs')}
    manifest.update(schema='physical-raster-v1', available_at=iso(available_at), availability_reference=availability_reference,
                    time_support='native_scene_time_not_per_pixel_time', geometry_reference=geo_ref,
                    model_ready=False, observation_export_ready=not reasons,
                    pending=reasons+['model_registry_normalization_and_footprint_admission'],
                    crs=crs, transform=affine, shape=list(values.shape), valid_pixels=int(valid.sum()),
                    source_files=[dict(name=p.name, sha256=checksums[str(p)]) for p in files],
                    source_integrity='hashes_at_import_not_supplier_signature',
                    physical_validation='metadata_and_qc_not_independent_radiometric_validation')
    tmp = Path(tempfile.mkdtemp(prefix='.'+out.name+'-',dir=out.parent))
    try:
        np.savez_compressed(tmp/'pixels.npz', **arrays)
        manifest['arrays_sha256'] = sha256(tmp/'pixels.npz')
        for p in files:
            if sha256(p) != checksums[str(p)]: raise ValueError('Native input changed during import.')
        _dump(tmp/'manifest.json',manifest)
        # mkdir is an exclusive reservation; do not replace another writer's directory.
        out.mkdir()
        try:
            for p in tmp.iterdir(): os.replace(p,out/p.name)
        except BaseException:
            shutil.rmtree(out)
            raise
    finally:
        shutil.rmtree(tmp,ignore_errors=True)
    return manifest


def read_capsule(directory, *, max_array_bytes=256*1024*1024):
    import zipfile
    root = Path(directory)
    m = read_json(child(root,'manifest.json')); p = child(root,'pixels.npz')
    if m.get('schema') != 'physical-raster-v1' or sha256(p) != m.get('arrays_sha256'):
        raise ValueError('Capsule checksum/schema mismatch.')
    with zipfile.ZipFile(p) as z:
        if len(z.infolist()) > 8 or sum(i.file_size for i in z.infolist()) > max_array_bytes:
            raise ValueError('Array archive exceeds memory budget.')
    with np.load(p,allow_pickle=False) as z: a = {k:z[k] for k in z.files}
    required = {'values','valid','latitude','longitude'}
    if not required.issubset(a) or set(a)-required-{'view_zenith_deg','footprint_km'}: raise ValueError('Unexpected arrays.')
    shape = tuple(m['shape'])
    if len(shape)!=2 or any(v.shape!=shape for v in a.values()) or a['valid'].dtype != bool:
        raise ValueError('Array dimensions/mask mismatch.')
    if any(v.dtype.kind not in 'biuf' for v in a.values()):
        raise ValueError('Only numerical arrays are supported.')
    if any(not np.isfinite(v[a['valid']]).all() for k,v in a.items() if k!='valid'):
        raise ValueError('Nonfinite admitted physical values.')
    valid = a['valid']
    if not valid.any() or (a['values'][valid] <= 0).any():
        raise ValueError('No valid positive Kelvin measurements.')
    if (np.abs(a['latitude'][valid]) > 90).any() or (np.abs(a['longitude'][valid]) > 180).any():
        raise ValueError('Coordinates outside geographical ranges.')
    if utc(m['available_at']) < utc(m['observed_at']):
        raise ValueError('Capsule availability precedes its observation.')
    if m.get('valid_pixels') != int(valid.sum()):
        raise ValueError('Declared valid pixel count differs from the mask.')
    return m,a


def export_observations(directory, output, *, max_records=100_000):
    """Export all valid pixels or fail; no silent point subsampling/oversampling.

    The existing model's pack_observations remains authoritative for the registry,
    normalisation, 12h window and native footprint size relative to the mesh.
    """
    m,a = read_capsule(directory)
    if not m.get('observation_export_ready') or not {'view_zenith_deg','footprint_km'}.issubset(a):
        raise ValueError('Complete view geometry is required before observation export.')
    valid = a['valid']; count = int(valid.sum())
    if type(max_records) is not int or not 0 < count <= max_records <= 1_000_000:
        raise ValueError('Pixel count exceeds explicit record budget; no silent subsampling.')
    if m['source'] not in ('arktika_m','electro_l') or m['quantity']!='brightness_temperature' or m['units']!='K':
        raise ValueError('No unsupported source/quantity may enter the infrared exporter.')
    variable = ':'.join((m['source'],m['platform'],m['instrument'],m['channel_id']))
    target = Path(output).absolute()
    if target.exists() or target.is_symlink() or any(p.is_symlink() for p in target.parents):
        raise ValueError('Unsafe/existing output.')
    target.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+target.name,dir=target.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            for row,col in zip(*np.where(valid)):
                angle, footprint = float(a['view_zenith_deg'][row,col]),float(a['footprint_km'][row,col])
                if not 0<=angle<90 or footprint<=0: raise ValueError('Invalid geometry in stored capsule.')
                record = dict(observation_id=f"{m['source_files'][0]['sha256']}:{m['channel_id']}:{row}:{col}",
                    source=m['source'],variable=variable,platform=m['platform'],channel_id=m['channel_id'],
                    value=float(a['values'][row,col]),units='K',latitude=float(a['latitude'][row,col]),
                    longitude=float(a['longitude'][row,col]),observed_at=m['observed_at'],available_at=m['available_at'],
                    revision=0,valid=True,quality=1.,view_zenith_deg=angle,footprint_km=footprint,
                    geometry_reference=m['geometry_reference'],time_support=m['time_support'],
                    radiometry=dict(instrument=m['instrument'],quantity='brightness_temperature',units='K',
                                    physical_channel_ids=[m['channel_id']],calibration_id=m['calibration_reference'],
                                    channel_mapping_verified=True))
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        os.link(temporary,target)  # exclusive publication, not replace
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dict(records=count, variable=variable, sha256=sha256(target), model_ready=False,
                next_step='register exact variable/platform/channel and its frozen sensor-specific normalization; pack_observations validates mesh footprint')
