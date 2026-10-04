"""Read-only producer adapters and training-only channel statistics. No network calls."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import numpy as np
from .frames import sensor_registry, read_scene, utc


def attach(dataset_path, plan_path, output):
    """Seal a local sensor plan into a new dataset, preserving scalar inputs/targets."""
    from ..pipeline.io import read_json, atomic_json, artifact, reference, local_path
    root=Path(dataset_path).absolute().parent
    target=Path(output).absolute()
    if target.parent!=root or target.exists():
        raise ValueError('Новый манифест должен находиться рядом с набором.')
    data=read_json(dataset_path);plan=read_json(plan_path)
    if set(plan)!={'multimodal','samples'} or not isinstance(plan['samples'],dict):
        raise ValueError('План должен описывать приборы и кадры примеров.')
    registry=sensor_registry(plan['multimodal'])
    if set(plan['samples'])-{s['id'] for s in data['samples']}:
        raise ValueError('В плане есть неизвестный пример.')
    for s in data['samples']:
        groups=plan['samples'].get(s['id'],{})
        if not isinstance(groups,dict) or set(groups)-set(registry['sensors']):
            raise ValueError('Неизвестный прибор в плане.')
        s['satellite']={}
        for name,paths in groups.items():
            if not isinstance(paths,list) or len(paths)>32:
                raise ValueError('Допускается не более 32 кадров на прибор.')
            s['satellite'][name]=[reference(root,local_path(root,p)) for p in paths]
    data['multimodal']=registry
    atomic_json(target,data)
    return {'status':'multimodal_attached','normalization_required':True,'dataset':str(target)}


from .normalizers import fit_channels
from .capsules import from_capsules, tile_scene
