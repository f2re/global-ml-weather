"""Подготовка DEM для существующих входов elevation_m и land_fraction.

Точечная выборка высоты не является средним по сферической ячейке.
Батиметрия не подставляется в атмосферную высоту. На воде и в смешанных
ячейках сохраняется независимая исходная орография.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .pipeline.io import read_json, read_arrays, write_arrays, sha256
from .grid import build_grid, latlon

MAX_DEM_BYTES=900_000_000


def local_file(path):
    path=Path(path).absolute()
    if not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Нужен обычный локальный файл без символических ссылок.')
    return path


def verify_dem(path,manifest):
    path=local_file(path)
    if manifest.get('schema')!='global-dem-source-1' or manifest.get('registration')!='pixel':
        raise ValueError('Неизвестная схема или регистрация DEM.')
    size=path.stat().st_size
    if size>MAX_DEM_BYTES or size!=manifest.get('bytes') or sha256(path)!=manifest.get('sha256'):
        raise ValueError('Размер или SHA256 DEM не совпали; предел 900000000 байт.')
    if manifest.get('spacing_arcsec')!=30 or manifest.get('shape')!=[21600,43200]:
        raise ValueError('Этот адаптер требует проверенный глобальный DEM 30 угловых секунд.')
    if manifest.get('scale_factor_m')!=.5 or manifest.get('add_offset_m')!=0:
        raise ValueError('Упаковка высот отличается от проверенного источника.')
    return path


def sample_grid(values,latitude,longitude,*,scale,offset,fill,block_shape):
    """Блочное чтение ближайших центров. Глобальный массив не читается целиком."""
    lat,lon=np.broadcast_arrays(np.asarray(latitude,float),np.asarray(longitude,float))
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or (np.abs(lat)>90).any():
        raise ValueError('Некорректные координаты.')
    ny,nx=values.shape
    if ny<2 or nx!=2*ny or len(block_shape)!=2 or min(block_shape)<1:
        raise ValueError('Нужен глобальный растр 2:1 и положительные блоки.')
    rows=np.clip(np.floor((lat+90)*ny/180).astype(np.int64),0,ny-1).ravel()
    cols=np.floor(((lon+180)%360)*nx/360).astype(np.int64).ravel()
    by,bx=block_shape;groups=(rows//by)*(1+(nx-1)//bx)+cols//bx
    result=np.empty(len(rows),dtype=np.float64)
    order=np.argsort(groups,kind='stable');cuts=np.r_[0,1+np.flatnonzero(np.diff(groups[order])),len(order)]
    for start,end in zip(cuts[:-1],cuts[1:]):
        ids=order[start:end]
        if not len(ids):continue
        r0=int(rows[ids[0]]//by*by);c0=int(cols[ids[0]]//bx*bx)
        block=np.asarray(values[r0:min(r0+by,ny),c0:min(c0+bx,nx)])
        selected=block[rows[ids]-r0,cols[ids]-c0]
        if not np.isfinite(selected).all() or (selected==fill).any():
            raise ValueError('DEM содержит отсутствующие высоты в запрошенных точках.')
        result[ids]=selected.astype(float)*scale+offset
    return result.reshape(lat.shape)


def dem_samples(path,manifest,latitude,longitude):
    import h5py
    path=verify_dem(path,manifest)
    with h5py.File(path,'r') as source:
        if not all(k in source for k in ('elevation','lat','lon')):raise ValueError('Нет высоты или координат DEM.')
        z=source['elevation'];y=source['lat'][:];x=source['lon'][:]
        units=z.attrs.get('units');units=units.decode() if isinstance(units,bytes) else units
        scale=float(np.asarray(z.attrs.get('scale_factor',1)).reshape(-1)[0])
        offset=float(np.asarray(z.attrs.get('add_offset',0)).reshape(-1)[0])
        if z.shape!=tuple(manifest['shape']) or z.dtype!=np.dtype('int16') or units!='m' or scale!=.5 or offset!=0:
            raise ValueError('Физическая схема файла не совпала с паспортом.')
        if not np.allclose(y,-90+(np.arange(21600)+.5)/120,atol=1e-10,rtol=0):raise ValueError('Неверные широты DEM.')
        if not np.allclose(x,-180+(np.arange(43200)+.5)/120,atol=1e-10,rtol=0):raise ValueError('Неверные долготы DEM.')
        fill=int(np.asarray(z.attrs.get('_FillValue',-32768)).reshape(-1)[0])
        return sample_grid(z,latitude,longitude,scale=scale,offset=offset,fill=fill,block_shape=z.chunks or (180,360))


def combine_surface(dem,reference_height,land_fraction):
    """DEM только при независимой доле суши ровно 1; вода/смешанные ячейки остаются исходными.

    Это явное приближение, не высокодетальная маска озёр и берегов.
    Отрицательная высота суши не отсекается и не трактуется как море.
    """
    dem,reference,land=map(lambda a:np.asarray(a,float),(dem,reference_height,land_fraction))
    if dem.ndim!=1 or dem.shape!=reference.shape or land.shape!=dem.shape:
        raise ValueError('Поля должны иметь одну ось ячеек.')
    if not all(np.isfinite(a).all() for a in (dem,reference,land)) or ((land<0)|(land>1)).any():
        raise ValueError('Отсутствуют физические высоты или доля суши.')
    mask=land==1.
    return np.where(mask,dem,reference),mask


def prepare(dem_path,manifest_path,reference_static,output,*,mesh_level,confirm_landmask=False):
    if not confirm_landmask:
        raise ValueError('Подтвердите независимую маску суши через --confirm-landmask. Знак DEM маской не является.')
    source=read_json(local_file(manifest_path));grid=build_grid(mesh_level)
    ref_path=local_file(reference_static);ref_hash=sha256(ref_path);base=read_arrays(ref_path)
    if str(base.get('grid_fingerprint'))!=grid.fingerprint or np.asarray(base.get('surface_units')).tolist()!=['m','1']:
        raise ValueError('Исходные статические поля имеют другую сетку или единицы.')
    ll=latlon(grid.xyz);samples=dem_samples(dem_path,source,ll[:,0],ll[:,1])
    elevation,applied=combine_surface(samples,base['elevation_m'],base['land_fraction'])
    if not applied.any():raise ValueError('Нет ячеек с подтверждённой полной долей суши; замена не выполнена.')
    if sha256(ref_path)!=ref_hash:raise ValueError('Исходные статические поля изменились.')
    meta={'schema':'terrain-static-1','dem_sha256':source['sha256'],'reference_static_sha256':ref_hash,
          'dem_spacing_arcsec':30,'dem_bytes':source['bytes'],'surface_height_units':'m',
          'operator':'nearest_pixel_centre_at_pure_land_cells; reference_orography_over_water_and_mixed_cells',
          'area_conservative':False,'earth_shape':'sphere','negative_land_preserved':True,
          'land_mask_source':'reference_static.land_fraction; independently checked by operator',
          'height_definition':'DEM topography, approximately compatible with geopotential height; not geopotential m2/s2',
          'changed_cells':int(applied.sum()),'weather_skill_verified':False}
    data=dict(base);data.update(elevation_m=elevation.astype(np.float32),dem_sample_m=samples.astype(np.float32),
            dem_applied_mask=applied,terrain_provenance=np.asarray(json.dumps(meta,sort_keys=True)))
    write_arrays(output,**data)
    return dict(meta,status='terrain_static_prepared',output_sha256=sha256(output),grid_fingerprint=grid.fingerprint)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dem',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--reference-static',required=True);p.add_argument('--mesh-level',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--confirm-landmask',action='store_true')
    a=p.parse_args(argv)
    report=prepare(a.dem,a.manifest,a.reference_static,a.output,mesh_level=a.mesh_level,confirm_landmask=a.confirm_landmask)
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
