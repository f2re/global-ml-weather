"""Локальные ограниченные файлы; никаких URL из манифестов и pickle."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from .contracts import Sensor, Sequence, fingerprint, hash_value

LIMIT=256*1024**2


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def regular(path):
    p=Path(path).absolute()
    if p.is_symlink() or any(q.is_symlink() for q in p.parents) or not p.is_file():
        raise ValueError('Нужен обычный локальный файл без символических ссылок.')
    return p


def read_json(path):
    p=regular(path)
    if p.stat().st_size>4*1024**2:raise ValueError('Слишком большой JSON.')
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('Повторный ключ JSON.')
            out[k]=v
        return out
    return json.loads(p.read_text(encoding='utf-8'),object_pairs_hook=unique,
                parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Неконечный JSON.')))


def resolve(root,ref):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('Нужны путь и SHA256.')
    hash_value(ref['sha256']);rel=Path(ref['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Выход за каталог запрещён.')
    p=regular(Path(root)/rel)
    if p.stat().st_size>LIMIT or sha256(p)!=ref['sha256']:raise ValueError('Изменённый или слишком большой файл.')
    return p


def reference(root,path):
    p=regular(path)
    return {'path':p.relative_to(Path(root).absolute()).as_posix(),'sha256':sha256(p)}


def exclusive(path,writer):
    p=Path(path).absolute()
    if p.exists() or p.is_symlink() or any(q.is_symlink() for q in p.parents):raise FileExistsError('Перезапись запрещена.')
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:writer(f);f.flush();os.fsync(f.fileno())
        os.link(name,p)
    finally:Path(name).unlink(missing_ok=True)


def write_json(path,obj):
    raw=json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    exclusive(path,lambda f:f.write(raw))


def read_arrays(path):
    p=regular(path)
    if p.stat().st_size>LIMIT:raise ValueError('Превышен предел файла.')
    with zipfile.ZipFile(p) as archive:
        members=archive.infolist()
        if not 1<=len(members)<=32 or len({x.filename for x in members})!=len(members):raise ValueError('Неожиданные элементы NPZ.')
        if sum(x.file_size for x in members)>LIMIT:raise ValueError('Превышен объём распакованного NPZ.')
        for item in members:
            if '/' in item.filename or not item.filename.endswith('.npy'):raise ValueError('Неожиданный файл в NPZ.')
            with archive.open(item) as f:
                v=np.lib.format.read_magic(f)
                if v==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=16384)
                elif v==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(f,max_header_size=16384)
                else:raise ValueError('Неподдерживаемый формат NPY.')
                size=int(np.prod(shape,dtype=object))*dtype.itemsize
                if dtype.hasobject or dtype.kind not in 'biufUS' or size<0 or size>LIMIT or size!=item.file_size-f.tell():
                    raise ValueError('Небезопасный размер или тип NPY.')
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


TENSORS=('values','valid','observed_unix','available_unix','latitude','longitude',
         'view_zenith_deg','solar_zenith_deg','footprint_km','area_m2','link_pixel','link_cell','link_weight')


def save_sequence(path,seq):
    arrays={k:getattr(seq,k).detach().cpu().numpy() for k in TENSORS if getattr(seq,k) is not None}
    meta={k:v for k,v in vars(seq).items() if k not in TENSORS}
    arrays['metadata']=np.array(json.dumps(meta,sort_keys=True,allow_nan=False))
    exclusive(path,lambda f:np.savez_compressed(f,**arrays))


def load_sequence(path,sensor):
    before=sha256(regular(path));data=read_arrays(path)
    meta=json.loads(str(data.pop('metadata')))
    if set(data)-set(TENSORS):raise ValueError('Неизвестное поле последовательности.')
    for k in ('frame_ids','channel_ids'):meta[k]=tuple(meta[k])
    meta['source_sha256']=before
    for k in data:
        v=data[k]
        dtype=torch.bool if k=='valid' else torch.long if k in ('link_pixel','link_cell') else (
              torch.float64 if k in ('observed_unix','available_unix') else torch.float32)
        if k=='valid' and v.dtype!=bool:raise ValueError('Маска должна быть Boolean.')
        if k in ('link_pixel','link_cell') and v.dtype.kind not in 'iu':raise ValueError('Индексы должны быть целыми.')
        meta[k]=torch.as_tensor(v,dtype=dtype)
    seq=Sequence(**meta).validate(sensor)
    if sha256(path)!=before:raise ValueError('Файл изменился при чтении.')
    return seq


def load_sensors(path):
    data=read_json(path)
    if not {'schema','sensors'}.issubset(data) or set(data)-{'schema','sensors','statistics'} or data['schema']!='multimodal-sensors-1':raise ValueError('Неверный реестр приборов.')
    sensors=tuple(Sensor.from_dict(s) for s in data['sensors'])
    if not sensors or len(sensors)>16 or len({s.id for s in sensors})!=len(sensors):raise ValueError('Повторные или избыточные адаптеры.')
    statistics=data.get('statistics',{})
    if not isinstance(statistics,dict):raise ValueError('Неверный реестр файлов норм.')
    for sensor in sensors:
        if sensor.data_kind=='real' and sensor.fit_end is not None and sensor.id not in statistics:
            raise ValueError('Для реальных норм нужен проверяемый исходный файл статистики.')
    for key,ref in statistics.items():
        item=next((s for s in sensors if s.id==key),None)
        if item is None:raise ValueError('Нормы относятся к неизвестному прибору.')
        p=resolve(Path(path).absolute().parent,ref)
        norms=read_json(p)
        if (norms.get('schema')!='multimodal-statistics-1' or norms.get('sensor')!=item.id
                or norms.get('measurement_signature')!=item.measurement_signature
                or norms.get('fit_end')!=item.fit_end or norms.get('data_kind')!=item.data_kind):
            raise ValueError('Происхождение, период или физические каналы норм не совпадают.')
        if sha256(p)!=item.normalization_sha256 or norms['mean']!=[c.mean for c in item.channels] or norms['std']!=[c.std for c in item.channels]:
            raise ValueError('Статистики не совпадают с реестром.')
    return sensors


def save_sensors(path,sensors):
    write_json(path,{'schema':'multimodal-sensors-1','sensors':[asdict(s) for s in sensors]})
