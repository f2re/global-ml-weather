"""Frozen per-channel statistics from the training partition only."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import numpy as np
from .frames import read_scene, utc


def fit_channels(dataset_path, output):
    """Keep existing scalar statistics; fit distinct raster channels on train only."""
    from .dataset import TrainingDataset
    from ..pipeline.io import artifact, reference, atomic_json, sha256
    from ..pipeline.fit import Moments
    from ..normalization import NormalizationBundle
    ds=TrainingDataset(dataset_path,require_raster_norm=False)
    if ds.multimodal is None:raise ValueError('Нет многоканального реестра.')
    target=Path(output).absolute(); norm_path=target.with_suffix('.channels.json')
    if target.parent!=ds.root or target.exists() or norm_path.exists():
        raise ValueError('Нужны новые файлы в каталоге выборки.')
    moments={ch['variable']:Moments() for s in ds.multimodal['sensors'].values() for ch in s['channels']}
    units={ch['variable']:ch['units'] for s in ds.multimodal['sensors'].values() for ch in s['channels']}
    if set(moments)&set(ds.norm.stats):
        raise ValueError('Нормы этих каналов уже существуют. Не меняйте замороженный эксперимент.')
    hashes={};seen=set();start=[];end=[]
    for sample in ds.subset('train'):
        for name,refs in ds._satellite[sample.id].items():
            spec=ds.multimodal['sensors'][name]
            for ref in refs:
                path=artifact(ds.root,ref,limit=128*1024**2)
                a,meta=read_scene(path,spec,kind=ds.kind,issue=sample.issue)
                # Unique native channel/pixel identity; partial time windows do not duplicate samples.
                for k,ch in enumerate(spec['channels']):
                    indices=np.flatnonzero(a['valid'][k]); fresh=[]
                    for pixel in indices:
                        key=(name,ref['sha256'],k,int(pixel))
                        if key not in seen:seen.add(key);fresh.append(pixel)
                    if fresh:
                        values=a['values'][k].ravel()[fresh]
                        moments[ch['variable']].add(values,np.ones(len(values),bool),1.)
                        times=a['observed_utc_s'].ravel()[fresh];start.append(float(times.min()));end.append(float(times.max()))
                hashes[ref['path']]=ref['sha256']
    payload=deepcopy(ds.norm._payload)
    for name,moment in moments.items():
        mu,sd=moment.finish()
        payload['variables'][name]={'units':units[name],'mean':[float(mu)],'std':[float(sd)],'pressure_pa':[],'interval_hours':None}
    from datetime import datetime, timezone
    provenance=payload['provenance'];old_period=provenance.get('fit_period')
    period={'start':datetime.fromtimestamp(min(start),timezone.utc).isoformat(),
            'end':datetime.fromtimestamp(max(end),timezone.utc).isoformat()}
    provenance['components']=[{'kind':'base_scalar_normalization','provenance':deepcopy(ds.norm._payload['provenance'])},
                               {'kind':'unique_training_pixels','period':period,'weighting':'equal_native_pixel; not global area climatology',
                                'sha256':hashes}]
    provenance['artifact_sha256']={**provenance['artifact_sha256'],**{'raster:'+k:v for k,v in hashes.items()}}
    if old_period and old_period.get('start') and old_period.get('end'):
        provenance['fit_period']={'start':min(utc(old_period['start']),utc(period['start'])).isoformat(),
                                  'end':max(utc(old_period['end']),utc(period['end'])).isoformat()}
    else:provenance['fit_period']=None
    provenance['revision']='multimodal-pixel-normalization-1'
    bundle=NormalizationBundle(payload)
    for split in ('validation','test'):
        times=[s.issue for s in ds.samples if s.split==split]
        if times:bundle.assert_independent_test((min(times)-timedelta(hours=ds.history_hours)).isoformat())
    ds.assert_unchanged()
    for sample in ds.subset('train'):
        for refs in ds._satellite[sample.id].values():
            for ref in refs:artifact(ds.root,ref,limit=128*1024**2)
    bundle.save(norm_path)
    manifest=dict(ds.manifest,normalization=reference(ds.root,norm_path))
    atomic_json(target,manifest)
    return {'status':'raster_norms_frozen','channels':sorted(moments),'dataset':str(target),
            'normalization':bundle.fingerprint,'test_pixels_read':False}

