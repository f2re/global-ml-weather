"""Fresh pressure-normalized S1/S2 research; measured targets; explicit frozen R6/R7 norm schemas."""
from __future__ import annotations
import argparse
from datetime import timedelta
import fcntl
import hashlib
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
from .profile_normalization import load_normalization
from .profile_training import ProfileDataset,VARIABLES,bounded_records,configuration,identity,_restore_rng
from .vertical import PRESSURE_HPA, hydrostatic_residual

MIN_HUMIDITY_PRESSURE_PA = 30000.0


def vertical_pressure_weights(pressure_pa=None) -> np.ndarray:
    if pressure_pa is None:
        p = np.asarray(PRESSURE_HPA, dtype=float) * 100.0
    elif isinstance(pressure_pa, torch.Tensor):
        p = pressure_pa.detach().cpu().numpy().astype(float)
    else:
        p = np.asarray(pressure_pa, dtype=float)
    n = len(p)
    half = np.zeros(n + 1)
    half[0] = p[0] + (p[0] - p[1]) / 2.0
    half[1:-1] = (p[:-1] + p[1:]) / 2.0
    half[-1] = 0.0
    dp = half[:-1] - half[1:]
    return dp * (float(n) / float(dp.sum()))



def objective(model: PressureProfileModel,frames: list,targets: list[dict],
              min_humidity_pressure_pa: float=MIN_HUMIDITY_PRESSURE_PA,
              use_vertical_weights: bool=True) -> tuple[torch.Tensor,list[int]]:
    terms=[[] for _ in VARIABLES]
    weights=vertical_pressure_weights(model.pressure_pa) if use_vertical_weights else None
    log_p_model=np.log(model.pressure_pa.detach().cpu().numpy()) if use_vertical_weights else None
    for record in targets:
        variable=VARIABLES.index(record['variable'])
        if variable==1 and record['pressure_pa']<min_humidity_pressure_pa:
            continue
        _,std,supported=model.normalization.at(variable,record['pressure_pa'])
        if not supported: continue
        result=_prediction(frames,model.grid,record,model.pressure_pa,3)
        if result is None: continue
        prediction=result[0]
        if not torch.isfinite(prediction): raise FloatingPointError('Nonfinite physical prediction.')
        if variable==1 and model.normalization.humidity_transform!='identity':
            difference=torch.log1p(prediction/model.normalization.q_scale)-np.log1p(record['value']/model.normalization.q_scale)
        else: difference=prediction-record['value']
        loss_val=(difference/float(std)).square()
        if use_vertical_weights:
            level=int(np.abs(log_p_model-np.log(record['pressure_pa'])).argmin())
            loss_val=float(weights[level])*loss_val
        terms[variable].append(loss_val)
    if not any(terms): raise ValueError('No jointly supported observed targets.')
    return torch.stack([torch.stack(rows).mean() for rows in terms if rows]).mean(),[len(rows) for rows in terms]


def reconstruction(model: PressureProfileModel,dataset: ProfileDataset,targets: list[dict],limit: int,split: str,
                   min_humidity_pressure_pa: float=MIN_HUMIDITY_PRESSURE_PA,
                   use_vertical_weights: bool=True) -> torch.Tensor:
    # A separate S1 origin at the actual held-out measurement time. It never
    # feeds observations after the S2 origin into that already-started forecast.
    admitted=[r for r in targets if r['variable']=='temperature' and model.normalization.at(r['variable'],r['pressure_pa'])[2]]
    if not admitted: raise ValueError('No held-out temperature profile for S1 reconstruction.')
    first=admitted[0]; when=utc(first['observed_at']); group=first['profile_id']
    held=[r for r in targets if r['profile_id']==group and utc(r['observed_at'])==when]
    inputs=dataset.records(when-timedelta(hours=12),when,issue=when,split=split)
    inputs=bounded_records([r for r in inputs if r['profile_id']!=group],limit)
    try:
        return objective(model,model(inputs,when)[:1],held,
                         min_humidity_pressure_pa=min_humidity_pressure_pa,
                         use_vertical_weights=use_vertical_weights)[0]
    except TypeError:
        return objective(model,model(inputs,when)[:1],held)[0]


def score(model: PressureProfileModel,dataset: ProfileDataset,config: dict,split: str) -> dict[str,float]:
    model.eval(); forecast=[]; analysis=[]
    min_humidity=config.get('min_humidity_pressure_pa',MIN_HUMIDITY_PRESSURE_PA)
    use_weights=config.get('use_vertical_weights',True)
    with torch.no_grad():
        for issue in dataset.issues(split,config['max_'+split+'_issues']):
            inputs,targets=dataset.sample(issue,config['max_records_per_window'])
            if not targets: continue
            try:
                f_val=objective(model,model(inputs,issue),targets,
                                min_humidity_pressure_pa=min_humidity,
                                use_vertical_weights=use_weights)[0]
            except TypeError:
                f_val=objective(model,model(inputs,issue),targets)[0]
            forecast.append(float(f_val))
            try:
                a_val=reconstruction(model,dataset,targets,config['max_records_per_window'],split,
                                     min_humidity_pressure_pa=min_humidity,
                                     use_vertical_weights=use_weights)
            except TypeError:
                a_val=reconstruction(model,dataset,targets,config['max_records_per_window'],split)
            analysis.append(float(a_val))
    if not forecast: raise ValueError('No observed validation support.')
    return {'forecast':float(np.mean(forecast)),'reconstruction':float(np.mean(analysis))}


def _load_fixed_state(model: PressureProfileModel, state: dict) -> None:
    """Checkpoint weights cannot replace the declared fixed physical buffers."""
    buffers=dict(model.named_buffers())
    for name in ('mean','std','norm_support','pressure_pa','xyz','log_pressure'):
        if name not in state or not torch.equal(state[name].to(buffers[name].device),buffers[name]):
            raise ValueError('Checkpoint fixed normalization or geometry buffer differs: '+name)
    model.load_state_dict(state,strict=True)


def _build_model(norms, config: dict, device, climatology_path=None):
    grid = build_pyramid(config['mesh_level'])[0]
    if climatology_path is None:
        return PressureProfileModel(grid, norms, config['hidden']).to(device)
    from .seasonal_climatology import SeasonalClimatology
    from .profile_seasonal_model import SeasonalProfileModel
    climate = SeasonalClimatology(climatology_path, grid=grid)
    return SeasonalProfileModel(grid, norms, climate, config['hidden']).to(device)


def _model_identity(model, norm_path: Path, norm_hash: str) -> dict:
    result = {'architecture': model.normalization.architecture,
              'norm_sha256': norm_hash, 'norm_path': str(norm_path.resolve())}
    if hasattr(model, 'climatology'):
        from .profile_seasonal_model import ARCHITECTURE
        result.update(architecture=ARCHITECTURE,
                      climatology_path=str(model.climatology.root.resolve()),
                      climatology_sha256=model.climatology.fingerprint)
    return result


def _verify_climate(model) -> None:
    if hasattr(model, 'climatology'):
        model.climatology.verify_sources()


def _training_status(model) -> str:
    if hasattr(model, 'climatology'):
        from .profile_seasonal_model import STATUS
        return STATUS
    return model.normalization.status


def train(dataset_path: str|Path,norm_path: str|Path,output: str|Path,config: dict,
          climatology_path: str|Path|None=None) -> None:
    v2_keys = {'reconstruction_weight', 'min_humidity_pressure_pa', 'use_vertical_weights', 'hydrostatic_weight'}
    weight = config.get('reconstruction_weight', .25)
    if type(weight) not in (int, float) or not 0 <= weight <= 1:
        raise ValueError('Invalid reconstruction weight.')
    min_humidity = float(config.get('min_humidity_pressure_pa', MIN_HUMIDITY_PRESSURE_PA))
    use_weights = bool(config.get('use_vertical_weights', True))
    hydro_weight = float(config.get('hydrostatic_weight', 0.0))
    config = configuration({k: v for k, v in config.items() if k not in v2_keys})
    config.update(
        reconstruction_weight=weight,
        min_humidity_pressure_pa=min_humidity,
        use_vertical_weights=use_weights,
        hydrostatic_weight=hydro_weight,
    )
    dataset=ProfileDataset(dataset_path); norm_path=Path(norm_path); norm_hash=digest(norm_path)
    norms=load_normalization(json.loads(norm_path.read_text()))
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
        model=_build_model(norms,config,device,climatology_path)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        expected={**identity(dataset,config,device),**_model_identity(model,norm_path,norm_hash)}
        completed=0; best=float('inf'); best_epoch=None; stale=0
        if (output/'latest.json').exists():
            ref=json.loads((output/'latest.json').read_text()); state=torch.load(checkpoint_path(output,ref),map_location=device,weights_only=True)
            if state['identity']!=expected: raise ValueError('Incompatible R6 source/data/norm/numerical resume.')
            _load_fixed_state(model,state['model']); optimizer.load_state_dict(state['optimizer']); _restore_rng(state,device)
            completed,best,best_epoch,stale=state['epoch'],state['best'],state['best_epoch'],state['stale']
            best_ref=ref if best_epoch==completed else state['best_ref']; checkpoint_path(output,best_ref); save(output/'best.json',best_ref)
        if any(output.glob('.epoch-*')) or (output/f'epoch-{completed+1:04d}').exists():
            raise ValueError('Ambiguous interrupted R6 epoch; preserve for inspection.')
        for epoch in range(completed+1,config['epochs']+1):
            dataset.verify()
            if hasattr(norms,'verify_sources'): norms.verify_sources()
            _verify_climate(model)
            if digest(norm_path)!=norm_hash: raise ValueError('Pressure norms changed.')
            model.train(); started=time.monotonic(); losses=[]; gradients={}; coverage=np.zeros(5,dtype=int)
            issues=dataset.issues('train',config['max_train_issues']); random.shuffle(issues)
            min_humidity=config.get('min_humidity_pressure_pa',MIN_HUMIDITY_PRESSURE_PA)
            use_weights=config.get('use_vertical_weights',True)
            hydro_weight=config.get('hydrostatic_weight',0.0)
            for issue in issues:
                inputs,targets=dataset.sample(issue,config['max_records_per_window'])
                if not targets: continue
                optimizer.zero_grad(set_to_none=True)
                frames=model(inputs,issue)
                forecast,counts=objective(model,frames,targets,min_humidity_pressure_pa=min_humidity,use_vertical_weights=use_weights)
                analysis=reconstruction(model,dataset,targets,config['max_records_per_window'],'train',min_humidity_pressure_pa=min_humidity,use_vertical_weights=use_weights)
                loss=forecast+weight*analysis
                if hydro_weight>0:
                    hydro_res=hydrostatic_residual(frames[0].profiles[...,:5],model.pressure_pa)
                    loss=loss+hydro_weight*(hydro_res/1000.0).square().mean()
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
            if hasattr(norms,'verify_sources'): norms.verify_sources()
            _verify_climate(model)
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
        save(output/'complete.json',{'identity':expected,'best_epoch':best_epoch,'status':_training_status(model),
             'source_roles':{'input':'IGRA','target':'IGRA','normalization':norms.payload['schema'],
                             'climatology':'fixed NOAA monthly means as masked context' if hasattr(model,'climatology') else None,
                             'external_verification':'ERA5 frozen-model only'},
             'scientific_acceptance':False,'test_independence':'old periods already seen; new independent acceptance required'})


def load_frozen(dataset_path: str|Path,training: str|Path,device: str='auto') -> tuple[PressureProfileModel,ProfileDataset]:
    training=Path(training);completion=json.loads((training/'complete.json').read_text())
    if completion.get('status') not in ('measured_pressure_profile_research_trained','measured_graphcast_profile_research_trained','measured_seasonal_profile_research_trained'): raise ValueError('Complete R6 training first.')
    ref=json.loads((training/'best.json').read_text());state=torch.load(checkpoint_path(training,ref),map_location='cpu',weights_only=True)
    if state['identity']!=completion['identity'] or ref['epoch']!=completion['best_epoch']: raise ValueError('R6 completion differs.')
    previous=state['identity']; norm_path=Path(previous['norm_path']);dataset=ProfileDataset(dataset_path)
    if digest(norm_path)!=previous['norm_sha256']: raise ValueError('R6 pressure norms changed.')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    norms=load_normalization(json.loads(norm_path.read_text()))
    sources={'dataset_manifest_sha256':dataset.manifest_sha256,'database_sha256':dataset.manifest['database_sha256'],
             'source_sha256':dataset.manifest['source_sha256'],'admission_sha256':dataset.manifest['admission_sha256']}
    if norms.payload['source_identity']!=sources:
        raise ValueError('Frozen normalization data or schema differs.')
    config=previous['config'];torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    model=_build_model(norms,config,device,previous.get('climatology_path'))
    expected={**identity(dataset,config,device),**_model_identity(model,norm_path,digest(norm_path))}
    if previous!=expected or completion['status']!=_training_status(model): raise ValueError('Frozen profile identity differs.')
    _verify_climate(model)
    _load_fixed_state(model,state['model']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,dataset


def lead_bin(observed_at,issue) -> int:
    hours=(utc(observed_at)-utc(issue)).total_seconds()/3600
    if not 0<hours<=72: raise ValueError('Observation outside future 72-hour support.')
    return math.ceil(hours/3)-1


def _frozen_training_hashes(training: Path) -> dict[str, str]:
    completion_hash = digest(training / 'complete.json')
    best_hash = digest(training / 'best.json')
    reference = json.loads((training / 'best.json').read_text())
    state_path = checkpoint_path(training, reference)
    return {'complete_json': completion_hash, 'best_json': best_hash, 'state_pt': digest(state_path)}


def evaluate(dataset_path: str|Path,training: str|Path,output: str|Path) -> None:
    from types import SimpleNamespace
    training=Path(training); frozen_hashes=_frozen_training_hashes(training)
    model,dataset=load_frozen(dataset_path,training); completion=json.loads((training/'complete.json').read_text());config=completion['identity']['config']
    if _frozen_training_hashes(training)!=frozen_hashes:raise ValueError('Frozen training changed during diagnostic loading.')
    sums=np.zeros((24,5,5)); rejected=np.zeros(5,dtype=int)
    min_humidity=config.get('min_humidity_pressure_pa',MIN_HUMIDITY_PRESSURE_PA)
    with torch.no_grad():
        for issue in dataset.issues('test',config['max_test_issues']):
            inputs,targets=dataset.sample(issue,config['max_records_per_window']);frames=model(inputs,issue)
            persistence=[SimpleNamespace(profiles=frames[0].profiles,profile_mask=frames[0].profile_mask,
                         profile_variable_mask=frames[0].profile_variable_mask,wind_basis=frames[0].wind_basis,
                         valid_time=f.valid_time,lead_hours=f.lead_hours) for f in frames]
            for record in targets:
                variable=VARIABLES.index(record['variable'])
                if variable==1 and record['pressure_pa']<min_humidity:
                    continue
                a=_prediction(frames,model.grid,record,model.pressure_pa,3)
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
    dataset.verify();_verify_climate(model)
    if hasattr(model.normalization,'verify_sources'):model.normalization.verify_sources()
    if digest(completion['identity']['norm_path'])!=completion['identity']['norm_sha256'] or _frozen_training_hashes(training)!=frozen_hashes:
        raise ValueError('Frozen training or normalization changed during diagnostic test.')
    save(output,{'split':'test','test_independence':'previously seen periods; diagnostic only','metrics':rows,
         'unsupported_records':rejected.tolist(),'identity':completion['identity'],
         'training_artifact_sha256':frozen_hashes,'scientific_acceptance':False})



def forecast(dataset_path: str|Path, training: str|Path, issue: str, output: str|Path) -> None:
    """Frozen causal forecast; verified immutable retries, no future targets."""
    from .vertical import PROFILE_UNITS,PROFILE_VARIABLES
    training=Path(training)
    def training_hashes():
        return _frozen_training_hashes(training)
    frozen_hashes=training_hashes()
    completion=json.loads((training/'complete.json').read_text())
    frozen_reference=json.loads((training/'best.json').read_text())
    model,dataset=load_frozen(dataset_path,training)
    if training_hashes()!=frozen_hashes:raise ValueError('Frozen training artifacts changed during loading.')
    config=completion['identity']['config']; origin=utc(issue)
    destination=Path(output)
    inputs=bounded_records(dataset.records(origin-timedelta(hours=12),origin,issue=origin),
                           config['max_records_per_window'])
    dataset.verify()
    if hasattr(model.normalization,'verify_sources'):model.normalization.verify_sources()
    _verify_climate(model)
    if digest(completion['identity']['norm_path'])!=completion['identity']['norm_sha256']:
        raise ValueError('Forecast normalization changed.')
    expected={'schema':'measured-profile-frozen-forecast-1','issue_time':origin.isoformat(),
         'lead_hours':list(range(0,73,3)),
         'valid_times':[(origin+timedelta(hours=lead)).isoformat() for lead in range(0,73,3)],
         'profile_variables':list(PROFILE_VARIABLES),'profile_units':list(PROFILE_UNITS),
         'input_count':len(inputs),'input_window_hours':12,'future_targets_read':False,
         'input_sha256':hashlib.sha256(json.dumps(inputs,sort_keys=True,allow_nan=False).encode()).hexdigest(),
         'wind_basis':'local_enu_vector','identity':completion['identity'],
         'checkpoint':frozen_reference,'training_artifact_sha256':frozen_hashes,'scientific_acceptance':False}
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError('Forecast output symlinks are forbidden.')
    if destination.exists():
        if not (destination/'forecast.json').is_file() or not (destination/'forecast.npz').is_file():
            raise ValueError('Incomplete immutable forecast; preserve for inspection.')
        metadata=json.loads((destination/'forecast.json').read_text())
        stored_hash=metadata.pop('artifact_sha256',None)
        if metadata!=expected or digest(destination/'forecast.npz')!=stored_hash:
            raise ValueError('Immutable forecast identity or artifact changed.')
        if training_hashes()!=frozen_hashes:raise ValueError('Frozen training artifacts changed during reuse.')
        return
    stage=destination.with_name('.'+destination.name+'.incomplete')
    if stage.exists() or stage.is_symlink():
        raise ValueError('Ambiguous incomplete forecast; preserve for inspection.')
    with torch.no_grad(): frames=model(inputs,origin)
    dataset.verify()
    if hasattr(model.normalization,'verify_sources'):model.normalization.verify_sources()
    _verify_climate(model)
    if digest(completion['identity']['norm_path'])!=completion['identity']['norm_sha256']:
        raise ValueError('Forecast normalization changed during inference.')
    if training_hashes()!=frozen_hashes:raise ValueError('Frozen training artifacts changed during inference.')
    destination.parent.mkdir(parents=True,exist_ok=True);stage.mkdir()
    with (stage/'forecast.npz').open('xb') as file:
        np.savez_compressed(file,profiles=np.stack([f.profiles.cpu().numpy() for f in frames]),
                            profile_variable_mask=np.stack([f.profile_variable_mask.cpu().numpy() for f in frames]),
                            pressure_pa=model.pressure_pa.cpu().numpy(),xyz=model.xyz.cpu().numpy(),
                            surface=np.stack([f.surface.cpu().numpy() for f in frames]),
                            surface_mask=np.stack([f.surface_mask.cpu().numpy() for f in frames]))
        file.flush();os.fsync(file.fileno())
    if training_hashes()!=frozen_hashes:raise ValueError('Frozen training artifacts changed during publication.')
    save(stage/'forecast.json',{**expected,'artifact_sha256':digest(stage/'forecast.npz')})
    sync_directory(stage);stage.rename(destination);sync_directory(destination.parent)


def main(argv: list[str]|None=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True)
    parser.add_argument('--norms');parser.add_argument('--output',required=True);parser.add_argument('--config');parser.add_argument('--training');parser.add_argument('--issue');parser.add_argument('--climatology')
    args=parser.parse_args(argv)
    if args.issue:
        if not args.training:parser.error('Frozen forecast requires --training and --issue.')
        forecast(args.dataset,args.training,args.issue,args.output)
    elif args.training:evaluate(args.dataset,args.training,args.output)
    else:
        if not args.norms or not args.config:parser.error('Training requires --norms and --config.')
        train(args.dataset,args.norms,args.output,json.loads(Path(args.config).read_text()),args.climatology)


if __name__=='__main__': main()
