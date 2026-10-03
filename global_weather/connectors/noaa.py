"""ISD CSV adapter. No reconstruction of unavailable historical receipt times."""
import csv
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

GOOD_QC = {'0', '1', '4', '5'}


def decode_number(text, missing, divisor):
    parts = text.split(',')
    if len(parts) < 2 or parts[1] not in GOOD_QC or parts[0].lstrip('+') == missing: return None
    return int(parts[0])/divisor


def convert_isd(source, output, *, acquired_at):
    """Use acquisition time as availability; never backdate archive arrivals.

    Nominal near-surface measurement heights must be reviewed before training.
    Wind is meteorological FROM-direction. Missing direction is allowed for calm.
    """
    import json
    when = datetime.fromisoformat(acquired_at.replace('Z', '+00:00'))
    if when.tzinfo is None: raise ValueError('Время загрузки должно содержать часовой пояс.')
    if Path(output).exists(): raise FileExistsError('Выход уже существует.')
    count = 0; rejected = Counter(); lines = []
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'STATION', 'DATE', 'LATITUDE', 'LONGITUDE'}.issubset(reader.fieldnames or []):
            raise ValueError('Не обнаружена обязательная схема NOAA ISD.')
        for r in reader:
            try:
                measured = datetime.fromisoformat(r['DATE'].replace('Z', '+00:00'))
                if measured.tzinfo is None: measured = measured.replace(tzinfo=timezone.utc)
                if measured > when: raise ValueError('Измерение позже поступления.')
                lat, lon = float(r['LATITUDE']), float(r['LONGITUDE'])
                if not math.isfinite(lat+lon) or abs(lat) > 90 or abs(lon) > 180: raise ValueError()
                values = []
                for field, name, unit, missing, offset in [('TMP', 't2m', 'K', '9999', 273.15), ('DEW', 'td2m', 'K', '9999', 273.15), ('SLP', 'mslp', 'Pa', '99999', 0.)]:
                    value = decode_number(r.get(field, ''), missing, 10.)
                    if value is not None: values.append((name, value*100 if name == 'mslp' else value+offset, unit))
                wind = r.get('WND', '').split(',')
                if len(wind) >= 5 and wind[4] in GOOD_QC and wind[3] != '9999':
                    speed = int(wind[3])/10.
                    direction = int(wind[0]) if wind[0] != '999' and wind[1] in GOOD_QC else None
                    if speed == 0: values.extend([('u10', 0., 'm s-1'), ('v10', 0., 'm s-1')])
                    elif direction is not None and 0 <= direction <= 360:
                        angle = math.radians(direction)
                        values.extend([('u10', -speed*math.sin(angle), 'm s-1'), ('v10', -speed*math.cos(angle), 'm s-1')])
                for name, value, units in values:
                    rec = dict(observation_id=f"ISD/{r['STATION']}/{measured.isoformat()}/{r.get('REPORT_TYPE','')}/{name}",
                               source='station', variable=name, value=value, units=units, latitude=lat, longitude=lon,
                               observed_at=measured.isoformat(), available_at=when.isoformat(), valid=True, revision=0,
                               provider='NOAA_ISD', availability_basis='archive_acquisition_not_historical_receipt',
                               height_reference='nominal_surface_height_requires_station_metadata')
                    elevation = float(r.get('ELEVATION', '9999') or '9999')
                    if math.isfinite(elevation) and elevation != 9999: rec['elevation_m'] = elevation
                    lines.append(json.dumps(rec, ensure_ascii=False, allow_nan=False)); count += 1
            except (ValueError, TypeError, KeyError): rejected['invalid_row'] += 1
    # Duplicate reports must be reconciled explicitly upstream, not silently averaged.
    ids = [json.loads(line)['observation_id'] for line in lines]
    if len(ids) != len(set(ids)): raise ValueError('Повторные сообщения: требуется явное согласование перед импортом.')
    temp = Path(output).with_suffix('.part')
    temp.write_text('\n'.join(lines)+ ('\n' if lines else ''), encoding='utf-8'); temp.replace(output)
    return dict(records=count, rejected=dict(rejected), historical_availability_known=False, model_ready=False)
