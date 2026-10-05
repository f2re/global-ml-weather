"""Нормализация метаданных. Уровень обработки не выводится из роли visual."""
import calendar
import datetime as dt
import hashlib
import json
import re
import unicodedata
from urllib.parse import urlsplit, urljoin, unquote, urlunsplit

UTC = dt.timezone.utc
PLATFORMS = ('ARCM1', 'ARCM2')
MONTHS = ('Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь')
CATEGORIES = [('all', 'Вся продукция'), ('channel', 'ИК-каналы GeoTIFF'),
              ('rgb', 'RGB GeoTIFF / COG'), ('image', 'Картинки PNG / JPEG'),
              ('raw', 'Исходники: кандидаты L0 / L1 / УФД'),
              ('science', 'Числовые продукты'), ('metadata', 'Метаданные / привязка'),
              ('other', 'Прочие файлы')]
CAT = dict(CATEGORIES)
LOW = {'0', 'L0', '0.0', 'L0.0', '1', 'L1', '1.0', 'L1.0', '1.5', 'L1.5'}


def text(value):
    return '' if value is None else str(value)


def safe_text(value):
    """Не пропускаем ESC, управляющие последовательности и bidi в терминал."""
    return ''.join(c if unicodedata.category(c) not in ('Cc', 'Cf', 'Cs') else ' '
                   for c in text(value))


def now():
    return dt.datetime.now(UTC)


def iso(value):
    return value.astimezone(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')


def parse_time(value):
    original = text(value)
    if not original:
        return '', False
    try:
        t = dt.datetime.fromisoformat(original.replace('Z', '+00:00'))
        assumed = t.tzinfo is None
        if assumed:
            t = t.replace(tzinfo=UTC)
        return iso(t), assumed
    except (ValueError, TypeError):
        return '', False


def shift_month(year, month, delta):
    n = year * 12 + month - 1 + delta
    y, m = divmod(n, 12)
    return max(1, min(9998, y)), m + 1


def period(year, month, day=None, clock=None):
    start = dt.datetime(year, month, day or 1, tzinfo=UTC)
    if day:
        stop = start + dt.timedelta(days=1)
    else:
        y, m = shift_month(year, month, 1)
        stop = dt.datetime(y, m, 1, tzinfo=UTC)
    clock = clock or now()
    if start > clock:
        raise ValueError('Дата находится в будущем по UTC.')
    # Сохраняем полуоткрытый интервал [start, stop), запрос может включать границу.
    return start, min(stop, clock)


def level_of(item):
    p = item.get('properties') or item
    for k in ('processing:level', 'processing_level_code', 'level'):
        if p.get(k) is not None:
            return text(p[k])
    return ''


def platform_of(item):
    p = item.get('properties') or item
    value = p.get('platform') or p.get('platform_identifier') or ''
    if value in PLATFORMS:
        return value
    match = re.search(r'(?:^|[._/])ARCM([12])(?:[._/]|$)', text(item.get('id') or item.get('item_id') or item.get('identifier')), re.I)
    return 'ARCM' + match.group(1) if match else text(value)


def infer_channel(name, uri, bands):
    m = re.search(r'(?:_ch|\bch\s*|\bband\s*)(\d{1,2})(?:\D|$)',
                  name + ' ' + unquote(urlsplit(uri).path), re.I)
    if not m:
        m = re.search(r'\.ir\.\d+\.(\d+)$', name)
    if not m and len(bands) == 1:
        m = re.search(r'band\s+(\d+)', text(bands[0].get('description')), re.I)
    return int(m.group(1)) if m else 0


def classify(asset, level, uri, name, channel):
    ext = urlsplit(uri).path.lower()
    roles = asset.get('roles') or []
    roles = roles if isinstance(roles, list) else []
    media = text(asset.get('type')).lower()
    if 'metadata' in roles or ext.endswith(('.json', '.xml', '.txt', '.pngw', '.jgw', '.tfw', '.prj', '.wld', '.xsd')):
        return 'metadata'
    if ext.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')) or 'thumbnail' in roles:
        return 'image'
    lev = level.upper()
    # Явное L2/L3 имеет приоритет над расширением .bin или словом raw в заголовке.
    if lev.startswith(('L2', 'L3', 'L4')):
        if ext.endswith(('.tif', '.tiff')) or 'tiff' in media:
            if lev == 'L2IR' and channel:
                return 'channel'
            if 'RGB' in lev:
                return 'rgb'
        if ext.endswith(('.nc', '.nc4', '.h5', '.hdf5')):
            return 'science'
        return 'other'
    if lev in LOW or lev == 'UFD' or ext.endswith(('.ufd', '.cadu', '.raw')):
        return 'raw'
    if lev in ('L1B', 'L1C', '1B', '1C') or ext.endswith(('.nc', '.nc4', '.h5', '.hdf5')):
        return 'science'
    # .bin само по себе не доказывает уровень данных.
    return 'other'


def normalize_asset(raw, context=None, name=None):
    c = context or {}
    a = dict(raw) if isinstance(raw, dict) else {'href': raw}
    uri = text(a.get('uri') or a.get('href') or a.get('url'))
    if not urlsplit(uri).scheme and c.get('self'):
        uri = urljoin(c['self'], uri)
    if urlsplit(uri).scheme not in ('https', 's3'):
        return None
    name = text(name or a.get('name') or unquote(urlsplit(uri).path).split('/')[-1])
    level = text(a.get('level') or a.get('processing:level') or c.get('level'))
    bands = a.get('eo:bands') or []
    channel = infer_channel(name, uri, bands)
    epsg = a.get('proj:epsg', a.get('epsg'))
    # Для RGB asset EPSG:4326 может не иметь proj:epsg; нельзя наследовать 3857 от item.
    if epsg is None:
        m = re.search(r'(?:\.)(4326|3857)(?:\.|$)', name)
        if not m:
            m = re.search(r'EPSG\s*:\s*(\d+)', text(a.get('title')), re.I)
        epsg = int(m.group(1)) if m else None
    try:
        epsg = int(epsg) if epsg is not None else 0
    except (ValueError, TypeError):
        epsg = 0
    date_original = a.get('time_utc') or a.get('time_original') or a.get('time') or c.get('time_original') or c.get('time') or ''
    date, assumed = parse_time(date_original)
    size = a.get('file:size', a.get('size'))
    try:
        size = int(size) if size is not None and text(size).strip() else None
    except (ValueError, TypeError):
        size = None
    if size is not None and size < 0:
        size = None
    item_id = text(a.get('item_id') or c.get('id'))
    raster_bands=a.get('raster:bands') or a.get('raster_bands') or []
    if not isinstance(raster_bands,list):raster_bands=[]
    p = urlsplit(uri)
    # Обновлённая подписанная ссылка того же объекта не создаёт второй элемент.
    canonical = urlunsplit((p.scheme, p.netloc, p.path, '', ''))
    aid = hashlib.sha256((item_id + '\n' + canonical).encode('utf-8')).hexdigest()[:32]
    return {'id': aid, 'item_id': item_id, 'uri': uri, 'name': name,
            'filename': unquote(p.path).split('/')[-1] or 'object.bin',
            'title': text(a.get('title') or name), 'type': text(a.get('type')),
            'platform': text(a.get('platform') or c.get('platform')),
            'level': level, 'time': date, 'time_original': text(date_original),
            'time_assumed': assumed, 'category': classify(a, level, uri, name, channel),
            'channel': channel, 'epsg': epsg, 'size': size,
            'raster_bands': [{k:b[k] for k in ('unit','scale','offset','nodata','data_type') if k in b} for b in raster_bands if isinstance(b,dict)],
            'bands': bands, 'roles': a.get('roles') or [],
            'region': a.get('storage:region') or a.get('region') or 'ext-dc1',
            'access': text(a.get('access')) if a.get('access') == 'READABLE' else '',
            'source': text(c.get('source') or a.get('source') or 'import'),
            'cog': 'cloud-optimized' in text(a.get('type')) or '.cog.' in uri.lower()}


def normalize_item(item, source='stac'):
    p = item.get('properties') or item
    identity = text(item.get('id') or item.get('identifier'))
    platform = platform_of(item)
    if platform not in PLATFORMS:
        return None, []
    stamp = p.get('datetime') or p.get('start_datetime') or p.get('acquisition_date_begin') or ''
    date, assumed = parse_time(stamp)
    rec = {'id': identity, 'platform': platform, 'level': level_of(item),
           'time': date, 'time_original': stamp, 'time_assumed': assumed,
           'source': source, 'collection': text(item.get('collection')),
           'title': text(p.get('title') or p.get('abstract') or identity),
           'bbox': item.get('bbox'), 'geometry': item.get('geometry'), 'gsd': p.get('gsd')}
    for link in item.get('links') or []:
        if link.get('rel') == 'self':
            rec['self'] = link.get('href', '')
    assets = []
    for name, raw in (item.get('assets') or {}).items():
        a = normalize_asset(raw, rec, name)
        if a:
            assets.append(a)
    # Только прямой URI сеанса. coverage, passport_name и derived_from не исходники.
    if item.get('dataset_uri'):
        a = normalize_asset({'uri': item['dataset_uri']}, rec, 'dataset_uri')
        if a:
            assets.append(a)
    for link in item.get('links') or []:
        if link.get('rel') in ('data', 'download', 'enclosure'):
            a = normalize_asset(link, rec, link.get('title') or link['rel'])
            if a:
                assets.append(a)
    return rec, assets


def size_text(n):
    if n is None:
        return '?'
    value = float(n)
    for suffix in ('Б', 'КиБ', 'МиБ', 'ГиБ', 'ТиБ'):
        if value < 1024 or suffix == 'ТиБ':
            return ('{:.1f} {}'.format(value, suffix)).replace('.', ',')
        value /= 1024


def safe_component(value, limit=100):
    original = safe_text(value)
    value = re.sub(r'[<>:"/\\|?*]', '_', original).strip(' .') or '_'
    if value.split('.')[0].upper() in {'CON','PRN','AUX','NUL'} | {x+str(n) for x in ('COM','LPT') for n in range(1,10)}:
        value = '_' + value
    if len(value) > limit:
        suffix = '.' + value.rsplit('.', 1)[-1] if '.' in value else ''
        suffix = suffix[:12]
        value = value[:limit-12-len(suffix)] + '_' + hashlib.sha256(original.encode()).hexdigest()[:10] + suffix
    return value


def matches(a, filters):
    return (not filters.get('platform') or a['platform'] == filters['platform']) and (
        filters.get('category', 'all') == 'all' or a['category'] == filters['category']) and (
        not filters.get('channel') or a['channel'] == filters['channel']) and (
        not filters.get('epsg') or a['epsg'] == filters['epsg']) and (
        not filters.get('query') or filters['query'].lower() in
        (' '.join(text(a.get(k)) for k in ('name', 'title', 'filename', 'item_id', 'level'))).lower())
