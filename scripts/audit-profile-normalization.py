"""Read-only CPU audit of observation normalization and frozen R4 decoding.

GraphCast is inspected as a reference, never supplied to the trained model.
Bounded train samples are diagnostic examples, not replacement statistics.
"""
from __future__ import annotations
import argparse
from datetime import timedelta, datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
import xarray as xr
from global_weather.grid import build_pyramid
from global_weather.import_climatology import import_graphcast, PINNED_HASHES
from global_weather.observation_training import checkpoint_path, digest, save
from global_weather.observations import utc
from global_weather.profile_training import ProfileObservationModel, bounded_records, VARIABLES
from global_weather.vertical import PRESSURE_HPA


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args(); started=time.monotonic(); root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    dataset=args.parent/'dataset'; training=args.parent/'training'
    manifest=json.loads((dataset/'dataset.json').read_text())
    ref=json.loads((training/'best.json').read_text()); checkpoint=checkpoint_path(training,ref); before=digest(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True); config=state['identity']['config']
    mean=[row['mean'] for row in manifest['statistics']]; std=[row['std'] for row in manifest['statistics']]
    model=ProfileObservationModel(build_pyramid(config['mesh_level'])[0],mean,std,config['hidden'],
          [[row['minimum_pressure_pa'],row['maximum_pressure_pa']] for row in manifest['statistics']])
    model.load_state_dict(state['model'],strict=True); model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert torch.equal(model.mean,torch.tensor(mean,dtype=torch.float32))
    assert torch.equal(model.std,torch.tensor(std,dtype=torch.float32))
    directory=root/'assets/normalization/graphcast'
    bundle=import_graphcast(directory/'mean_by_level.nc',directory/'stddev_by_level.nc',expected_hashes=PINNED_HASHES)
    attrs={}
    for name in PINNED_HASHES:
        with xr.open_dataset(directory/name) as data: attrs[name]=dict(data.attrs)
    roundtrips=[]
    for row in manifest['statistics']:
        values=np.array([row['mean']-row['std'],row['mean'],row['mean']+row['std']],dtype=np.float64)
        restored=((values-row['mean'])/row['std'])*row['std']+row['mean']
        roundtrips.append({'variable':row['variable'],'max_error':float(np.max(np.abs(values-restored)))})
    issue=utc('2022-08-01T00:00:00Z'); uri=(dataset/'observations.sqlite').resolve().as_uri()+'?mode=ro'
    samples={}; input_rows=[]
    with sqlite3.connect(uri,uri=True) as db:
        for variable,name in enumerate(VARIABLES):
            rows=[json.loads(row[0]) for row in db.execute(
                "SELECT record FROM records WHERE split='train' AND variable=? ORDER BY observed,id LIMIT 10000",(variable,))]
            samples[name]=[]
            for pressure in (100000,70000,50000,20000,10000):
                values=[row['value'] for row in rows if abs(np.log(row['pressure_pa']/pressure))<=.025]
                samples[name].append({'pressure_pa':pressure,'count':len(values),'mean':float(np.mean(values)) if values else None})
        input_rows=[json.loads(row[0]) for row in db.execute(
            "SELECT record FROM records WHERE observed>? AND observed<=? AND available<=? AND split='test' ORDER BY observed,id",
            ((issue-timedelta(hours=12)).isoformat(),issue.isoformat(),issue.isoformat()))]
    inputs=bounded_records(input_rows,config['max_records_per_window']); heads=[]
    hook=model.head.register_forward_hook(lambda module,arguments,result: heads.append(result.detach().clone()))
    with torch.no_grad():frames=model(inputs,issue)
    hook.remove(); decode_checks=[]
    for index,frame in enumerate(frames):
        linear=heads[index]*model.std+model.mean
        finite=frame.profile_variable_mask[...,:5]
        errors={name:float((frame.profiles[...,i][finite[...,i]]-linear[...,i][finite[...,i]]).abs().max())
                for i,name in enumerate(VARIABLES)}
        decode_checks.append({'lead_hours':frame.lead_hours,'linear_inverse_max_error':errors})
    with np.load(args.reference,allow_pickle=False) as arrays:
        reference=arrays['profiles']; reference_mask=arrays['profile_mask']
    comparison=[]
    for index in (0,24):
        for pressure in (700,500,200):
            level=list(PRESSURE_HPA).index(pressure)
            prediction=frames[index].profiles[:,level,0].numpy()
            valid=reference_mask[index,:,level,0]&np.isfinite(prediction)
            comparison.append({'lead_hours':frames[index].lead_hours,'pressure_hpa':pressure,'cells':int(valid.sum()),
                 'prediction_mean_k':float(prediction[valid].mean()),'era5_mean_k':float(reference[index,valid,level,0].mean()),
                 'r4_norm_mean_k':mean[0],'graphcast_level_mean_k':bundle.get('temperature','K').at(pressure*100)[0]})
    q=np.array([0.,1e-6,1e-4,1e-3,.01],dtype=np.float32)
    q_restored=(std[1]*F.softplus(torch.from_numpy(q)/std[1])).numpy()
    assert digest(checkpoint)==before
    report={'schema':'profile-normalization-audit-1','utc':datetime.now(timezone.utc).isoformat(),
       'audit_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
       'checkpoint_training_commit':state['identity']['commit'],'checkpoint_sha256':before,
       'dataset_manifest_sha256':digest(dataset/'dataset.json'),'script_sha256':digest(Path(__file__)),
       'device':'CPU, one thread; read-only diagnostic, not production numerical replay','seconds':time.monotonic()-started,
       'active_norms':manifest['statistics'],'norm_period':manifest['norm_period'],
       'checkpoint_buffers_match_manifest':True,'graphcast_used_by_profile_model':False,
       'graphcast_provenance':bundle._payload['provenance'],'graphcast_attributes':attrs,
       'affine_roundtrip':roundtrips,'decode_checks':decode_checks,
       'humidity_post_affine_transform':{'physical_input_q':q.tolist(),'decoded_q':q_restored.tolist(),
            'zero_maps_to':float(q_restored[0]),'head_zero_maps_to':float(std[1]*F.softplus(torch.tensor(mean[1]/std[1])))},
       'bounded_train_samples':samples,'sample_limit_per_variable':10000,'input_records_before_bound':len(input_rows),
       'input_records_used':len(inputs),'temperature_comparison':comparison,
       'checkpoint_unchanged':True,'scientific_acceptance':False}
    save(args.output,report); print(json.dumps({'seconds':report['seconds'],'comparison':comparison,'humidity':report['humidity_post_affine_transform']},indent=2))


if __name__=='__main__':main()
