"""Bounded, non-pickle, immutable product and field files."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from .core import Field, Product, canonical, require_hash

LIMIT = 256*1024**2


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def regular(path):
    p = Path(path).absolute()
    if p.is_symlink() or any(q.is_symlink() for q in p.parents) or not p.is_file():
        raise ValueError('Нужен обычный локальный файл без символических ссылок.')
    return p


def read_json(path):
    p = regular(path)
    if p.stat().st_size > 4*1024**2: raise ValueError('Слишком большой JSON.')
    def unique(items):
        out = {}
        for k, v in items:
            if k in out: raise ValueError('Повторный ключ JSON.')
            out[k] = v
        return out
    return json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Неконечный JSON.')))


def arrays(path):
    p = regular(path)
    if p.stat().st_size > LIMIT: raise ValueError('Превышен размер файла.')
    with zipfile.ZipFile(p) as archive:
        info = archive.infolist()
        if not 1 <= len(info) <= 12 or len({x.filename for x in info}) != len(info):
            raise ValueError('Неверные или повторные элементы NPZ.')
        if sum(x.file_size for x in info) > LIMIT: raise ValueError('Превышен объём распакованных данных.')
        for member in info:
            if '/' in member.filename or not member.filename.endswith('.npy'):
                raise ValueError('Неожиданный элемент NPZ.')
            with archive.open(member) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0): shape, _, dtype = np.lib.format.read_array_header_1_0(f, max_header_size=16384)
                elif version == (2, 0): shape, _, dtype = np.lib.format.read_array_header_2_0(f, max_header_size=16384)
                else: raise ValueError('Неподдерживаемый заголовок NPY.')
                size = int(np.prod(shape, dtype=object))*dtype.itemsize
                if dtype.hasobject or dtype.kind not in 'biufUS' or size < 0 or size > LIMIT or size != member.file_size-f.tell():
                    raise ValueError('Небезопасный тип или размер массива.')
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def exclusive_bytes(path, writer):
    p = Path(path).absolute()
    if p.exists() or p.is_symlink() or any(q.is_symlink() for q in p.parents):
        raise FileExistsError('Нельзя перезаписывать исходник или идти через символическую ссылку.')
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+p.name+'-', dir=p.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f); f.flush(); os.fsync(f.fileno())
        os.link(tmp, p)
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_field(path, field):
    meta = dict(quantity=field.quantity, units=field.units, grid_id=field.grid_id,
                observed_at=field.observed_at, available_at=field.available_at,
                metadata=field.metadata, upstream_sha256=field.source_sha256)
    data = dict(values=field.values, valid=field.valid, metadata=np.asarray(canonical(meta)))
    if field.uncertainty is not None: data['uncertainty'] = field.uncertainty
    exclusive_bytes(path, lambda f: np.savez_compressed(f, **data))


def load_field(path):
    a = arrays(path)
    if set(a)-{'values', 'valid', 'metadata', 'uncertainty'} or not {'values','valid','metadata'}.issubset(a):
        raise ValueError('Неизвестный формат поля.')
    m = json.loads(str(a['metadata']))
    return Field(a['values'], a['valid'], m['quantity'], m['units'], m['grid_id'],
                 m['observed_at'], m['available_at'], sha256(path), m.get('metadata', {}), a.get('uncertainty'))


def save_product(path, p):
    exclusive_bytes(path, lambda f: np.savez_compressed(f, values=p.values, valid=p.valid,
                    qc=p.qc, uncertainty=p.uncertainty, metadata=np.asarray(canonical(p.metadata))))
    return sha256(path)


def load_product(path):
    a = arrays(path)
    if set(a) != {'values','valid','qc','uncertainty','metadata'}:
        raise ValueError('Неизвестный формат продукции.')
    m = json.loads(str(a['metadata']))
    return Product(m['product'], m['method'], a['values'], a['valid'], a['qc'], a['uncertainty'], m)


def resolve(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Нужны относительный путь и SHA256.')
    require_hash(ref['sha256'])
    rel = Path(ref['path'])
    if rel.is_absolute() or '..' in rel.parts or not rel.parts: raise ValueError('Выход из каталога запрещён.')
    p = regular(Path(root)/rel)
    if sha256(p) != ref['sha256']: raise ValueError('Файл изменён после подготовки задания.')
    return p
