"""Fresh pressure-normalized S1/S2 research; measured targets only, no ERA5."""
from __future__ import annotations
import argparse
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .analysis.observation_operator import _prediction
from .grid import build_pyramid
from .observation_training import checkpoint_path,digest,save,sync_directory
from .observations import utc
from .profile_model_v2 import PressureProfileModel
from .profile_normalization import PressureNormalization
from .profile_training import ProfileDataset,VARIABLES,bounded_records,configuration,identity,_restore_rng


def objective(model: PressureProfileModel,frames: list,targets: list[dict]) -> tuple[torch.Tensor,list[int]]:
    terms=[[] for _ in VARIABLES]
    for record in targets:
        variable=VARIABLES.index(record['variable']); _,std,supported=model.normalization.at(variable,record['pressure_pa'])
        if not supported: continue
        result=_prediction(frames,model.grid,record,model.pressure_pa,3)
        if result is None: continue
        prediction=result[0]
        if not torch.isfinite(prediction): raise FloatingPointError('Nonfinite physical prediction.')
        if variable==1:
            difference=torch.log1p(prediction/model.normalization.q_scale)-np.log1p(record['value']/model.normalization.q_scale)
        else: difference=prediction-record['value']
        terms[variable].append((difference/float(std)).square())
    if not any(terms): raise ValueError('No jointly supported observed targets.')
    return torch.stack([torch.stack(rows).mean() for rows in terms if rows]).mean(),[len(rows) for rows in terms]


def reconstruction(model: PressureProfileModel,dataset: ProfileDataset,targets: list[dict],limit: int,split: str) -> torch.Tensor:
    # A separate S1 origin at the actual held-out measurement time. It never
    # feeds observations after the S2 origin into that already-started forecast.
    admitted=[r for r in targets if r['variable']=='temperature' and model.normalization.at(r['variable'],r['pressure_pa'])[2]]
    if not admitted: raise ValueError('No held-out temperature profile for S1 reconstruction.')
    first=admitted[0]; when=utc(first['observed_at']); group=first['profile_id']
    held=[r for r in targets if r['profile_id']==group and utc(r['observed_at'])==when]
    inputs=dataset.records(when-timedelta(hours=12),when,issue=when,split=split)
    inputs=bounded_records([r for r in inputs if r['profile_id']!=group],limit)
    return objective(model,model(inputs,when)[:1],held)[0]


def score(model: PressureProfileModel,dataset: ProfileDataset,config: dict,split: str) -> dict[str,float]:
    model.eval(); forecast=[]; analysis=[]
    with torch.no_grad():
        for issue in dataset.issues(split,config['max_'+split+'_issues']):
            inputs,targets=dataset.sample(issue,config['max_records_per_window'])
            if not targets: continue
            forecast.append(float(objective(model,model(inputs,issue),targets)[0]))
            analysis.append(float(reconstruction(model,dataset,targets,config['max_records_per_window'],split)))
    if not forecast: raise ValueError('No observed validation support.')
    return {'forecast':float(np.mean(forecast)),'reconstruction':float(np.mean(analysis))}


def train(dataset_path: str|Path,norm_path: str|Path,output: str|Path,config: dict) -> None:
    weight=config.get('reconstruction_weight',.25)
    if type(weight) not in (int,float) or not 0<=weight<=1: raise ValueError('Invalid reconstruction weight.')
    config=configuration({k:v for k,v in config.items() if k!='reconstruction_weight'}); config['reconstruction_weight']=weight
    dataset=ProfileDataset(dataset_path); norm_path=Path(norm_path); norm_hash=digest(norm_path)
    norms=PressureNormalization(json.loads(norm_path.read_text()))
    sources={'dataset_manifest_sha256':dataset.manifest_sha256,'database_sha256':dataset.manifest['database_sha256'],
             'source_sha256':dataset.manifest['source_sha256'],'admission_sha256':dataset.manifest['admission_sha256']}
    if norms.payload['source_identity']!=sources:
        raise ValueError('Pressure normalization dataset differs.')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with (output/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(config['threads'])
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        model=PressureProfileModel(build_pyramid(config['mesh_level'])[0],norms,config['hidden']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        expected={**identity(dataset,config,device),'architecture':'pressure-profile-v2','norm_sha256':norm_hash,'norm_path':str(norm_path.resolve())}
        completed=0; best=float('inf'); best_epoch=None; stale=0
        if (output/'latest.json').exists():
            ref=json.loads((output/'latest.json').read_text()); state=torch.load(checkpoint_path(output,ref),map_location=device,weights_only=True)
            if state['identity']!=expected: raise ValueError('Incompatible R6 source/data/norm/numerical resume.')
            model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer']); _restore_rng(state,device)
            completed,best,best_epoch,stale=state['epoch'],state['best'],state['best_epoch'],state['stale']
            best_ref=ref if best_epoch==completed else state['best_ref']; checkpoint_path(output,best_ref); save(output/'best.json',best_ref)
        if any(output.glob('.epoch-*')) or (output/f'epoch-{completed+1:04d}').exists():
            raise ValueError('Ambiguous interrupted R6 epoch; preserve for inspection.')
        for epoch in range(completed+1,config['epochs']+1):
            dataset.verify()
            if digest(norm_path)!=norm_hash: raise ValueError('Pressure norms changed.')
            model.train(); started=time.monotonic(); losses=[]; gradients={}; coverage=np.zeros(5,dtype=int)
            issues=dataset.issues('train',config['max_train_issues']); random.shuffle(issues)
            for issue in issues:
                inputs,targets=dataset.sample(issue,config['max_records_per_window'])
                if not targets: continue
                optimizer.zero_grad(set_to_none=True)
                forecast,counts=objective(model,model(inputs,issue),targets)
                analysis=reconstruction(model,dataset,targets,config['max_records_per_window'],'train')
                loss=forecast+weight*analysis
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite S1/S2 loss.')
                loss.backward(); coverage+=counts
                for name,parameter in model.named_parameters():
                    if parameter.grad is None: gradients.setdefault(name,False); continue
                    if not torch.isfinite(parameter.grad).all(): raise FloatingPointError('Nonfinite gradient: '+name)
                    gradients[name]=gradients.get(name,False) or bool(parameter.grad.abs().sum()>0)
                for channel,name in enumerate(VARIABLES):
                    active=bool(model.head.weight.grad[channel].abs().sum()>0) and bool(model.head.bias.grad[channel].abs()>0)
                    key='physical_head.'+name; gradients[key]=gradients.get(key,False) or active
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                optimizer.step(); losses.append(float(loss.detach()))
                if len(losses)==1 or len(losses)%12==0:
                    save(output/'progress.json',{'epoch':epoch,'optimizer_steps':len(losses),'train_issues':len(issues),
                         'loss':losses[-1],'forecast_loss':float(forecast.detach()),'reconstruction_loss':float(analysis.detach()),
                         'seconds':time.monotonic()-started,'source_commit':expected['commit'],'device':str(device)})
            if not losses or not all(gradients.values()) or not np.all(coverage>0): raise ValueError('Incomplete branch/variable gradients.')
            validation=score(model,dataset,config,'validation'); value=validation['forecast']; improved=value<best
            if improved: best,best_epoch,stale=value,epoch,0
            else: stale+=1
            dataset.verify()
            if digest(norm_path)!=norm_hash: raise ValueError('Pressure norms changed during epoch.')
            temporary=output/f'.epoch-{epoch:04d}'; temporary.mkdir(); rng=np.random.get_state()
            state={'identity':expected,'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,
                   'best_ref':None if improved else json.loads((output/'best.json').read_text()),
                   'model':model.state_dict(),'optimizer':optimizer.state_dict(),'python_rng':random.getstate(),
                   'numpy_rng':(rng[0],rng[1].tolist(),*rng[2:]),'torch_rng':torch.get_rng_state(),
                   'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
            torch.save(state,temporary/'state.pt')
            with (temporary/'state.pt').open('rb') as file: os.fsync(file.fileno())
            metrics={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':validation,'seconds':time.monotonic()-started,
                     'nonzero_gradients':gradients,'measured_variable_counts':coverage.tolist(),'scientific_acceptance':False}
            save(temporary/'metrics.json',metrics); sync_directory(temporary)
            final=output/f'epoch-{epoch:04d}'; temporary.rename(final); sync_directory(output)
            ref={'directory':final.name,'epoch':epoch,'sha256':digest(final/'state.pt')}; save(output/'latest.json',ref)
            if improved: save(output/'best.json',ref)
            print(json.dumps(metrics),flush=True)
            if stale>=config['patience']: break
        save(output/'complete.json',{'identity':expected,'best_epoch':best_epoch,'status':'measured_pressure_profile_research_trained',
             'scientific_acceptance':False,'test_independence':'old periods already seen; new independent acceptance required'})


def load_frozen(dataset_path: str|Path,training: str|Path,device: str='auto') -> tuple[PressureProfileModel,ProfileDataset]:
    training=Path(training);completion=json.loads((training/'complete.json').read_text())
    if completion.get('status')!='measured_pressure_profile_research_trained': raise ValueError('Complete R6 training first.')
    ref=json.loads((training/'best.json').read_text());state=torch.load(checkpoint_path(training,ref),map_location='cpu',weights_only=True)
    if state['identity']!=completion['identity'] or ref['epoch']!=completion['best_epoch']: raise ValueError('R6 completion differs.')
    previous=state['identity']; norm_path=Path(previous['norm_path']);dataset=ProfileDataset(dataset_path)
    if digest(norm_path)!=previous['norm_sha256']: raise ValueError('R6 pressure norms changed.')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    config=previous['config'];torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    expected={**identity(dataset,config,device),'architecture':'pressure-profile-v2','norm_sha256':digest(norm_path),'norm_path':str(norm_path.resolve())}
    if previous!=expected: raise ValueError('R6 frozen identity differs.')
    model=PressureProfileModel(build_pyramid(config['mesh_level'])[0],PressureNormalization(json.loads(norm_path.read_text())),config['hidden']).to(device)
    model.load_state_dict(state['model'],strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,dataset


def lead_bin(observed_at,issue) -> int:
    hours=(utc(observed_at)-utc(issue)).total_seconds()/3600
    if not 0<hours<=72: raise ValueError('Observation outside future 72-hour support.')
    return math.ceil(hours/3)-1


def evaluate(dataset_path: str|Path,training: str|Path,output: str|Path) -> None:
    from types import SimpleNamespace
    model,dataset=load_frozen(dataset_path,training); completion=json.loads((Path(training)/'complete.json').read_text());config=completion['identity']['config']
    sums=np.zeros((24,5,5)); rejected=np.zeros(5,dtype=int)
    with torch.no_grad():
        for issue in dataset.issues('test',config['max_test_issues']):
            inputs,targets=dataset.sample(issue,config['max_records_per_window']);frames=model(inputs,issue)
            persistence=[SimpleNamespace(profiles=frames[0].profiles,profile_mask=frames[0].profile_mask,
                         profile_variable_mask=frames[0].profile_variable_mask,wind_basis=frames[0].wind_basis,
                         valid_time=f.valid_time,lead_hours=f.lead_hours) for f in frames]
            for record in targets:
                variable=VARIABLES.index(record['variable']); a=_prediction(frames,model.grid,record,model.pressure_pa,3)
                b=_prediction(persistence,model.grid,record,model.pressure_pa,3)
                if a is None or b is None:rejected[variable]+=1;continue
                error=float(a[0])-record['value'];control=float(b[0])-record['value']
                lead=lead_bin(record['observed_at'],issue)
                sums[lead,variable]+=[abs(error),error**2,error,control**2,1]
    rows=[]
    for lead in range(24):
        for variable in range(5):
            mae,mse,bias,control,count=sums[lead,variable]
            if count:rows.append({'lead_bin_hours':(lead+1)*3,'variable':VARIABLES[variable],'count':int(count),
                'mae':mae/count,'rmse':float(np.sqrt(mse/count)),'bias':bias/count,'analysis_persistence_rmse':float(np.sqrt(control/count))})
    if not rows:raise ValueError('No admitted diagnostic test predictions.')
    save(output,{'split':'test','test_independence':'previously seen periods; diagnostic only','metrics':rows,
         'unsupported_records':rejected.tolist(),'scientific_acceptance':False})


def main(argv: list[str]|None=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True)
    parser.add_argument('--norms');parser.add_argument('--output',required=True);parser.add_argument('--config');parser.add_argument('--training')
    args=parser.parse_args(argv)
    if args.training:evaluate(args.dataset,args.training,args.output)
    else:
        if not args.norms or not args.config:parser.error('Training requires --norms and --config.')
        train(args.dataset,args.norms,args.output,json.loads(Path(args.config).read_text()))


if __name__=='__main__': main()
