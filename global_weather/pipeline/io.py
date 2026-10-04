"""Bounded local artifacts. Data never specify commands, imports or remote URLs."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import zipfile
import numpy as np

MAX_JSON = 32 * 1024**2
MAX_ARRAY_BYTES = 512 * 1024**2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f'Повторный ключ JSON: {k}')
        result[k] = v
    return result


def parse_json(text):
    def invalid(value):
        raise ValueError(f'Недопустимое число JSON: {value}')
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON:
        raise ValueError('JSON должен быть ограниченным обычным файлом.')
    return parse_json(path.read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    if path.is_symlink() or tmp.is_symlink():
        raise ValueError('Символическая ссылка вместо файла результата.')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def local_path(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Требуется относительный локальный путь.')
    raw = relative.split('/')
    p = PurePosixPath(relative)
    if p.is_absolute() or any(x in ('', '.', '..') for x in raw):
        raise ValueError('Выход за пределы набора данных запрещён.')
    root = Path(root).resolve()
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Символические ссылки в наборе запрещены.')
    if not current.resolve().is_relative_to(root):
        raise ValueError('Путь находится вне набора данных.')
    return current


def artifact(root, ref, *, limit=None):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Артефакт требует path и sha256.')
    if not isinstance(ref['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', ref['sha256']):
        raise ValueError('Неверная контрольная сумма.')
    path = local_path(root, ref['path'])
    if not path.is_file() or (limit is not None and path.stat().st_size > limit):
        raise ValueError(f'Файл отсутствует или слишком велик: {ref["path"]}')
    if sha256(path) != ref['sha256']:
        raise ValueError(f'Изменён исходный файл: {ref["path"]}')
    return path


def reference(root, path):
    path = Path(path)
    relative = path.relative_to(root).as_posix()
    path = local_path(root, relative)
    return {'path': relative, 'sha256': sha256(path)}


def read_arrays(path, *, limit=MAX_ARRAY_BYTES):
    """Validate headers BEFORE allocation, including huge shapes in short ZIP files."""
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError('NPZ превышает предел или является ссылкой.')
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        if not members or len(members) > 32 or len({m.filename for m in members}) != len(members):
            raise ValueError('Недопустимый состав NPZ.')
        if sum(m.file_size for m in members) > limit:
            raise ValueError('Распакованный NPZ превышает предел.')
        for m in members:
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*\.npy', m.filename):
                raise ValueError('Недопустимое имя массива NPZ.')
            with z.open(m) as f:
                version = np.lib.format.read_magic(f)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError('Неподдерживаемая версия NPY.')
                shape, _, dtype = reader(f, max_header_size=16384)
                size = math.prod(shape) * dtype.itemsize
                if dtype.hasobject or dtype.fields or size > limit or size != m.file_size - f.tell():
                    raise ValueError('Небезопасный тип, размер или заголовок NPY.')
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError('Исходные массивы не перезаписываются.')
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
