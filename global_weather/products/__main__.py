"""Local product calculation from hash-pinned field files. No downloads or arbitrary code."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
from .catalog import CATALOG
from .core import canonical
from .io import read_json, resolve, load_field, save_product, sha256, exclusive_bytes
from .algorithms import (spectral_index, cloud_brightness_temperature, cloud_height,
                         liquid_water_path, surface_temperature)
from .soil import SoilEmissivityLUT, soil_moisture
from .ingest import export_product

REQUIRED = {
    'ndvi': ('a','b','eligible','solar_zenith'),
    'ndvi_toa': ('a','b','eligible','solar_zenith'),
    'ndmi': ('a','b','eligible','solar_zenith'),
    'ndsi': ('a','b','eligible','solar_zenith'),
    'cloud_top_brightness_temperature': ('bt','cloudy'),
    'cloud_top_height': ('cloud_temperature','temperature_profile','height_profile','opaque_single_layer'),
    'cloud_liquid_water_path': ('optical_depth','effective_radius','liquid_single_layer'),
    'land_surface_temperature': ('radiance','emissivity','transmittance','upwelling','downwelling','clear_land'),
    'soil_moisture_surface': ('tb_h','tb_v','soil_temperature','vegetation_temperature','tau_nadir','omega','incidence','eligible'),
}


def calculate(job_path, output):
    job = read_json(job_path)
    required = {'schema','product','inputs','available_at','availability_reference','data_kind'}
    if (not isinstance(job,dict) or not required.issubset(job)
            or set(job)-required-{'lut','parameters'} or job['schema'] != 'satellite-product-job-1'):
        raise ValueError('Неверный контракт задания продукции.')
    if not isinstance(job['availability_reference'],str) or not job['availability_reference'].strip():
        raise ValueError('Нужна ссылка на доказательство времени готовности.')
    name = job['product']
    if name not in REQUIRED or not isinstance(job['inputs'],dict) or set(job['inputs']) != set(REQUIRED[name]):
        raise ValueError('Отсутствуют необходимые входы либо есть неизвестные поля.')
    root = Path(job_path).absolute().parent
    paths = {k: resolve(root,v) for k,v in job['inputs'].items()}
    fields = {k: load_field(p) for k,p in paths.items()}
    kwargs = dict(available_at=job['available_at'], data_kind=job['data_kind'])
    params = job.get('parameters', {})
    allowed = {'max_solar_zenith'} if name in ('ndvi','ndvi_toa','ndmi','ndsi') else (
        {'max_chi2','max_conditional_sigma'} if name == 'soil_moisture_surface' else set())
    if not isinstance(params,dict) or set(params)-allowed:
        raise ValueError('Неизвестные параметры метода.')
    if name == 'soil_moisture_surface':
        lp = resolve(root,job['lut']); table = read_json(lp)
        lut = SoilEmissivityLUT(np.array(table['moisture']),np.array(table['emissivity_hv']),table['metadata'],sha256(lp))
        p = soil_moisture(**fields,lut=lut,**kwargs,**params)
        if sha256(lp) != job['lut']['sha256']: raise ValueError('Таблица изменилась.')
    elif name in ('ndvi','ndvi_toa','ndmi','ndsi'):
        if 'lut' in job: raise ValueError('Индексу не нужна таблица влажности почвы.')
        p = spectral_index(**fields,product=name,**kwargs,**params)
    else:
        if 'lut' in job: raise ValueError('Неизвестная таблица для этого метода.')
        functions = {'cloud_top_brightness_temperature': cloud_brightness_temperature,
                     'cloud_top_height': cloud_height, 'cloud_liquid_water_path': liquid_water_path,
                     'land_surface_temperature': surface_temperature}
        p = functions[name](**fields,**kwargs)
    metadata = dict(p.metadata,availability_reference=job['availability_reference'],job_sha256=sha256(job_path))
    p = replace(p,metadata=metadata)
    for k,path in paths.items():
        if sha256(path) != job['inputs'][k]['sha256']: raise ValueError('Вход изменился при расчёте.')
    checksum = save_product(output,p)
    return dict(status='product_calculated',product=p.name,method=p.method,
                valid_pixels=int(p.valid.sum()),invalid_pixels=int((~p.valid).sum()),
                data_kind=p.metadata['data_kind'],sha256=checksum,meteorologically_validated=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('catalog')
    c=commands.add_parser('calculate');c.add_argument('--job',required=True);c.add_argument('--output',required=True)
    c=commands.add_parser('export');c.add_argument('--product',required=True);c.add_argument('--geometry',required=True)
    c.add_argument('--output',required=True);c.add_argument('--registry-output',required=True)
    c.add_argument('--history-hours',type=int);c.add_argument('--max-records',type=int,default=100000)
    c=commands.add_parser('from-capsule');c.add_argument('--capsule',required=True);c.add_argument('--output',required=True)
    c.add_argument('--geometry-output');c.add_argument('--data-kind',choices=('real','synthetic'),required=True)
    args=parser.parse_args(argv)
    if args.command=='catalog': report={k: asdict(v) for k,v in CATALOG.items()}
    elif args.command=='calculate': report=calculate(args.job,args.output)
    elif args.command=='from-capsule':
        from .bridge import from_capsule
        report=from_capsule(args.capsule,args.output,data_kind=args.data_kind,geometry_output=args.geometry_output)
    else:
        if Path(args.registry_output).exists() or Path(args.registry_output).is_symlink():
            raise FileExistsError('Выходной реестр уже существует.')
        report=export_product(args.product,args.geometry,args.output,history_hours=args.history_hours,max_records=args.max_records)
        exclusive_bytes(args.registry_output,lambda f:f.write((canonical(report['registry'])+'\n').encode()))
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__': main()
