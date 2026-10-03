"""Bounded operator-invoked acquisition. Never called by the web server."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import re
from pathlib import Path
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlparse
from .noaa import convert_isd
from ..lab.contracts import atomic_json, sha256, now

HOSTS = {'storage.googleapis.com', 'www.ncei.noaa.gov'}


class StrictRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Avoid following signed/cloud or arbitrary redirect destinations silently.
        if urlparse(newurl).netloc != urlparse(req.full_url).netloc or urlparse(newurl).scheme != 'https':
            raise ValueError('Перенаправление на другой источник запрещено.')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, output, *, expected_sha256=None, max_bytes=32*1024*1024):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.netloc not in HOSTS or parsed.username or parsed.query or parsed.fragment:
        raise ValueError('Разрешены только зарегистрированные публичные HTTPS-источники без параметров доступа.')
    if expected_sha256 and not re.fullmatch('[0-9a-f]{64}', expected_sha256): raise ValueError('Неверный SHA256.')
    output = Path(output)
    if output.exists() or output.is_symlink(): raise FileExistsError('Существующий источник нельзя перезаписывать.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix+'.part')
    owned = False
    try:
        total = 0
        with build_opener(StrictRedirect()).open(Request(url, headers={'User-Agent': 'global-ml-weather-research/0.3'}), timeout=30) as response, temp.open('xb') as stream:
            owned = True
            while block := response.read(256*1024):
                total += len(block)
                if total > max_bytes: raise ValueError('Превышен предел загрузки.')
                stream.write(block)
        checksum = sha256(temp)
        if expected_sha256 and checksum != expected_sha256: raise ValueError('Контрольная сумма не совпала.')
        temp.replace(output)
    except Exception:
        if owned: temp.unlink(missing_ok=True)
        raise
    record = dict(source=url, acquired_at=now(), bytes=total, sha256=checksum,
                  expected_hash_verified=expected_sha256 is not None, content_verified=False)
    atomic_json(output.with_suffix(output.suffix+'.provenance.json'), record)
    return record


def era5_request(day, *, pressure_levels=True):
    from ..vertical import PRESSURE_HPA
    date = datetime.strptime(day, '%Y-%m-%d')
    request = dict(product_type=['reanalysis'], year=[date.strftime('%Y')], month=[date.strftime('%m')],
                   day=[date.strftime('%d')], time=[f'{h:02d}:00' for h in range(24)], data_format='netcdf', download_format='unarchived')
    if pressure_levels:
        dataset = 'reanalysis-era5-pressure-levels'
        request.update(variable=['temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'geopotential', 'vertical_velocity'], pressure_level=[str(p) for p in PRESSURE_HPA])
    else:
        dataset = 'reanalysis-era5-single-levels'
        request['variable'] = ['2m_temperature', '2m_dewpoint_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
                              'surface_pressure', 'mean_sea_level_pressure', 'total_precipitation', 'total_cloud_cover',
                              'sea_surface_temperature', 'sea_ice_cover', 'snow_depth']
    return dict(dataset=dataset, request=request, purpose='training_targets_or_frozen_normalization_not_operational_input')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('connector', choices=['graphcast', 'noaa-isd', 'era5-request', 'era5-download'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--network', action='store_true', help='Явно разрешить сетевую загрузку')
    parser.add_argument('--station'); parser.add_argument('--year', type=int)
    parser.add_argument('--date'); parser.add_argument('--surface', action='store_true')
    args = parser.parse_args(argv)
    if args.connector == 'era5-request':
        atomic_json(args.output, era5_request(args.date, pressure_levels=not args.surface)); return
    if not args.network: parser.error('Загрузка требует явного --network; сервер испытаний сеть не открывает.')
    if args.connector == 'graphcast':
        for name in ('mean_by_level.nc', 'stddev_by_level.nc'):
            record = download('https://storage.googleapis.com/dm_graphcast/graphcast/stats/'+name, args.output/name)
            print(json.dumps(record, ensure_ascii=False))
    elif args.connector == 'noaa-isd':
        if not args.station or not re.fullmatch(r'\d{11}', args.station) or not args.year or not 1900 <= args.year <= 2100:
            parser.error('Нужны --station (11 цифр) и --year.')
        url = f'https://www.ncei.noaa.gov/data/global-hourly/access/{args.year}/{args.station}.csv'
        record = download(url, args.output)
        result = convert_isd(args.output, args.output.with_suffix('.jsonl'), acquired_at=record['acquired_at'])
        print(json.dumps(result, ensure_ascii=False))
    else:
        # cdsapi uses operator-managed credentials; never print or copy .cdsapirc.
        import cdsapi
        plan = era5_request(args.date, pressure_levels=not args.surface)
        if args.output.exists(): raise FileExistsError('Файл уже существует.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix('.part')
        try:
            cdsapi.Client().retrieve(plan['dataset'], plan['request'], str(tmp))
            tmp.replace(args.output)
        except Exception:
            tmp.unlink(missing_ok=True); raise
        atomic_json(args.output.with_suffix('.provenance.json'), dict(plan, acquired_at=now(), sha256=sha256(args.output), content_verified=False))


if __name__ == '__main__': main()
