"""Bounded, versioned experiments. No arbitrary commands or server paths."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Literal


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def safe_child(root: Path, name: str) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name) or name in ('.', '..'):
        raise ValueError('Недопустимое имя файла или запуска.')
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError('Ссылки и выход за пределы каталога запрещены.')
    return path


class RunSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    kind: Literal['baseline', 'adaptive', 'tests', 'inspect'] = 'adaptive'
    mesh_level: int = Field(default=1, ge=0, le=3)
    hidden: Literal[16, 32] = 16
    latent_slots: Literal[4, 8, 16, 38] = 8
    horizon_hours: int = Field(default=72, ge=0, le=72)
    seed: int = Field(default=17, ge=0, le=2**31-1)
    optimizer_steps: int = Field(default=0, ge=0, le=5)
    remove_source: Literal['none', 'station', 'radiosonde'] = 'none'
    input_file: str = ''

    @field_validator('horizon_hours')
    @classmethod
    def valid_lead(cls, v):
        if v % 3:
            raise ValueError('Заблаговременность должна быть кратна 3 часам.')
        return v

    @field_validator('input_file')
    @classmethod
    def filename(cls, v):
        if v and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', v):
            raise ValueError('Требуется имя файла из входного каталога.')
        return v
