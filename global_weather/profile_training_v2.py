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
LEGACY_LOSS_KEYS = frozenset(('min_humidity_pressure_pa', 'use_vertical_weights', 'hydrostatic_weight'))


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


class _ScreenedProfileDataset(ProfileDataset):
    """R9 filtering is an immutable view, never a rewritten archive or norm."""
    def __init__(self, path, policy):
        super().__init__(path)
        self.physical_policy = policy
        self.qc_reports = {}

    def records(self, start, end, *, issue=None, split=None):
        from .profile_physics import screen_humidity
        raw = super().records(start, end, issue=issue, split=split)
        rows, report = screen_humidity(raw, self.physical_policy)
        key = (utc(start).isoformat(), utc(end).isoformat(),
               utc(issue).isoformat() if issue is not None else None, split)
        self.qc_reports[key] = dict(report, start=key[0], end=key[1], issue=key[2], split=split)
        return rows


def _build_dataset(path, config):
    if config.get('physical_policy') is None:
        return ProfileDataset(path)
    from .profile_physics import parse_policy
    return _ScreenedProfileDataset(path, parse_policy(config['physical_policy']))


def _configuration(config):
    config = dict(config)
    value = config.pop('physical_policy', None)
    supplied_legacy = LEGACY_LOSS_KEYS.intersection(config)
    if value is not None and supplied_legacy:
        raise ValueError('Do not mix physical_policy with legacy loss options: ' + ', '.join(sorted(supplied_legacy)))
    weight = config.pop('reconstruction_weight', .25)
    if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Invalid reconstruction weight.')
    if value is None:
        # Preserve the separately published main R9 configuration and defaults.
        minimum = config.pop('min_humidity_pressure_pa', MIN_HUMIDITY_PRESSURE_PA)
        use_weights = config.pop('use_vertical_weights', True)
        hydro_weight = config.pop('hydrostatic_weight', 0.0)
        for name, number in (('min_humidity_pressure_pa', minimum), ('hydrostatic_weight', hydro_weight)):
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError('Invalid legacy loss option: ' + name)
        if minimum > 100000 or type(use_weights) is not bool:
            raise ValueError('Invalid legacy pressure threshold or vertical weighting flag.')
        result = configuration(config)
        result.update(min_humidity_pressure_pa=float(minimum), use_vertical_weights=use_weights,
                      hydrostatic_weight=float(hydro_weight))
    else:
        from .profile_physics import parse_policy
        policy = parse_policy(value)
        mesh = config.get('mesh_level', 1)
        if type(mesh) is not int or not 0 <= mesh <= 4:
            raise ValueError('R9 mesh_level must be an integer from 0 to 4.')
        # Existing pilot limits are not silently raised for historical configs.
        result = configuration(dict(config, mesh_level=min(mesh, 2)))
        result['mesh_level'] = mesh
        if result['hidden'] > 128:
            raise ValueError('R9 hidden exceeds the reviewed research range.')
        estimate = (10*4**mesh+2)*37*result['hidden']*4*25*20
        if estimate > policy.memory_budget_mib*1024**2:
            raise ValueError('R9 estimated activations exceed the declared memory budget.')
        result['physical_policy'] = policy.payload()
    result['reconstruction_weight'] = weight
    return result


def _objective_options(config):
    """Dispatch an explicit experiment; never apply two weighting rules at once."""
    if config.get('physical_policy') is not None:
        if LEGACY_LOSS_KEYS.intersection(config):
            raise ValueError('Do not mix physical_policy with legacy loss options.')
        return {}
    return {key: config[key] for key in ('min_humidity_pressure_pa', 'use_vertical_weights') if key in config}


def objective(model: PressureProfileModel,frames: list,targets: list[dict],
              min_humidity_pressure_pa: float|None=None,
              use_vertical_weights: bool|None=None) -> tuple[torch.Tensor,list[int]]:
    policy = getattr(model, 'physical_policy', None)
    if policy is not None:
        if min_humidity_pressure_pa is not None or use_vertical_weights is not None:
            raise ValueError('Do not mix physical_policy with legacy objective arguments.')
        from .profile_physics import physical_objective
        loss, counts, report = physical_objective(model, frames, targets, policy)
        model.last_objective_report = report
        return loss, counts
    if min_humidity_pressure_pa is None:
        min_humidity_pressure_pa = MIN_HUMIDITY_PRESSURE_PA
    if use_vertical_weights is None:
        use_vertical_weights = True
    if (type(min_humidity_pressure_pa) not in (int, float) or not math.isfinite(min_humidity_pressure_pa)
            or not 0 <= min_humidity_pressure_pa <= 100000 or type(use_vertical_weights) is not bool):
        raise ValueError('Invalid legacy objective arguments.')
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
                   min_humidity_pressure_pa: float|None=None,
                   use_vertical_weights: bool|None=None) -> torch.Tensor:
    # A separate S1 origin at the actual held-out measurement time. It never
    # feeds observations after the S2 origin into that already-started forecast.
    admitted=[r for r in targets if r['variable']=='temperature' and model.normalization.at(r['variable'],r['pressure_pa'])[2]]
    if not admitted: raise ValueError('No held-out temperature profile for S1 reconstruction.')
    first=admitted[0]; when=utc(first['observed_at']); group=first['profile_id']
    held=[r for r in targets if r['profile_id']==group and utc(r['observed_at'])==when]
    inputs=dataset.records(when-timedelta(hours=12),when,issue=when,split=split)
    inputs=bounded_records([r for r in inputs if r['profile_id']!=group],limit)
    frames = model(inputs,when,horizon_hours=0) if isinstance(model,PressureProfileModel) else model(inputs,when)[:1]
    options = {}
    if min_humidity_pressure_pa is not None:
        options['min_humidity_pressure_pa'] = min_humidity_pressure_pa
    if use_vertical_weights is not None:
        options['use_vertical_weights'] = use_vertical_weights
    return objective(model,frames,held,**options)[0]


def score(model: PressureProfileModel,dataset: ProfileDataset,config: dict,split: str) -> dict[str,float]:
    model.eval(); forecast=[]; analysis=[]
    options = _objective_options(config)
    with torch.no_grad():
        for issue in dataset.issues(split,config['max_'+split+'_issues']):
            inputs,targets=dataset.sample(issue,config['max_records_per_window'])
            if not targets: continue
            forecast.append(float(objective(model,model(inputs,issue),targets,**options)[0]))
            analysis.append(float(reconstruction(model,dataset,targets,config['max_records_per_window'],split,**options)))
    if not forecast: raise ValueError('No observed validation support.')
    return {'forecast':float(np.mean(forecast)),'reconstruction':float(np.mean(analysis))}


def _load_fixed_state(model: PressureProfileModel, state: dict) -> None:
    """Checkpoint weights cannot replace the declared fixed physical buffers."""
    buffers=dict(model.named_buffers())
    for name in ('mean','std','norm_support','pressure_pa','xyz','log_pressure'):
        if name not in state or not torch.equal(state[name].to(buffers[name].device),buffers[name]):
            raise ValueError('Checkpoint fixed normalization or geometry buffer differs: '+name)
    for name in ('edge_src','edge_dst','edge_direction','edge_length_m','enu_basis','humidity_bias'):
        if name in buffers and (name not in state or not torch.equal(state[name].to(buffers[name].device),buffers[name])):
            raise ValueError('Checkpoint physical buffer differs: '+name)
    model.load_state_dict(state,strict=True)


def _build_model(norms, config: dict, device, climatology_path=None):
    from .profile_physics import parse_policy
    grid = build_pyramid(config['mesh_level'])[0]
    policy = parse_policy(config['physical_policy']) if config.get('physical_policy') is not None else None
    climate = None
    if climatology_path is not None:
        from .seasonal_climatology import SeasonalClimatology
        climate = SeasonalClimatology(climatology_path, grid=grid)
    if policy is not None and policy.architecture == 'hydrostatic_flow':
        from .profile_hydrostatic_model import HydrostaticFlowProfileModel
        model = HydrostaticFlowProfileModel(grid, norms, config['hidden'], climate)
    elif climate is not None:
        from .profile_seasonal_model import SeasonalProfileModel
        model = SeasonalProfileModel(grid, norms, climate, config['hidden'])
    else:
        model = PressureProfileModel(grid, norms, config['hidden'])
    if policy is not None:
        if norms.humidity_transform != 'identity':
            raise ValueError('R9 requires unchanged physical GraphCast normalization.')
        model.physical_policy = policy
    return model.to(device)


def _model_identity(model, norm_path: Path, norm_hash: str) -> dict:
    result = {'architecture': model.normalization.architecture,
              'norm_sha256': norm_hash, 'norm_path': str(norm_path.resolve())}
    if hasattr(model, 'climatology'):
        from .profile_seasonal_model import ARCHITECTURE
        result.update(architecture=ARCHITECTURE,
                      climatology_path=str(model.climatology.root.resolve()),
                      climatology_sha256=model.climatology.fingerprint)
    if getattr(model, 'physical_policy', None) is not None:
        result['physical_policy'] = model.physical_policy.payload()
        if model.physical_policy.architecture == 'hydrostatic_flow':
            from .profile_hydrostatic_model import ARCHITECTURE
            result['architecture'] = ARCHITECTURE
    return result


def _verify_climate(model) -> None:
    if hasattr(model, 'climatology'):
        model.climatology.verify_sources()


def _training_status(model) -> str:
    if getattr(model, 'physical_policy', None) is not None and model.physical_policy.architecture == 'hydrostatic_flow':
        from .profile_hydrostatic_model import STATUS
        return STATUS
    if hasattr(model, 'climatology'):
        from .profile_seasonal_model import STATUS
        return STATUS
    return model.normalization.status


def train(dataset_path: str|Path,norm_path: str|Path,output: str|Path,config: dict,
          climatology_path: str|Path|None=None) -> None:
    config=_configuration(config); weight=config['reconstruction_weight']
    options = _objective_options(config)
    hydro_weight = config.get('hydrostatic_weight', 0.0)
    dataset=_build_dataset(dataset_path,config); norm_path=Path(norm_path); norm_hash=digest(norm_path)
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
            for issue in issues:
                inputs,targets=dataset.sample(issue,config['max_records_per_window'])
                if not targets: continue
                optimizer.zero_grad(set_to_none=True)
                frames=model(inputs,issue)
                forecast,counts=objective(model,frames,targets,**options)
                objective_report = getattr(model, 'last_objective_report', None)
                analysis=reconstruction(model,dataset,targets,config['max_records_per_window'],'train',**options)
                loss=forecast+weight*analysis
                if hydro_weight > 0:
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
                         'seconds':time.monotonic()-started,'source_commit':expected['commit'],'device':str(device),
                         'forecast_objective':objective_report})
            if not losses or not all(gradients.values()) or not np.all(coverage>0): raise ValueError('Incomplete branch/variable gradients.')
            validation=score(model,dataset,config,'validation'); value=validation['forecast']; improved=value<best
            if hasattr(dataset, 'qc_reports'):
                save(output/'qc-queries.json', {'schema':'profile-qc-queries-1','policy':config['physical_policy'],
                     'count_scope':'per_query_not_unique_observations','queries':list(dataset.qc_reports.values())})
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
    if completion.get('status') not in ('measured_pressure_profile_research_trained','measured_graphcast_profile_research_trained','measured_seasonal_profile_research_trained','measured_hydrostatic_flow_profile_research_trained'): raise ValueError('Complete R6 training first.')
    ref=json.loads((training/'best.json').read_text());state=torch.load(checkpoint_path(training,ref),map_location='cpu',weights_only=True)
    if state['identity']!=completion['identity'] or ref['epoch']!=completion['best_epoch']: raise ValueError('R6 completion differs.')
    previous=state['identity']; norm_path=Path(previous['norm_path']);dataset=_build_dataset(dataset_path,previous['config'])
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
    from .vertical import PRESSURE_HPA
    by_pressure=np.zeros((24,37,5,5))
    min_humidity = 0.0 if config.get('physical_policy') is not None else config.get('min_humidity_pressure_pa', MIN_HUMIDITY_PRESSURE_PA)
    humidity_target_masked = 0
    with torch.no_grad():
        for issue in dataset.issues('test',config['max_test_issues']):
            inputs,targets=dataset.sample(issue,config['max_records_per_window']);frames=model(inputs,issue)
            persistence=[SimpleNamespace(profiles=frames[0].profiles,profile_mask=frames[0].profile_mask,
                         profile_variable_mask=frames[0].profile_variable_mask,wind_basis=frames[0].wind_basis,
                         valid_time=f.valid_time,lead_hours=f.lead_hours) for f in frames]
            for record in targets:
                variable=VARIABLES.index(record['variable'])
                if variable == 1 and record['pressure_pa'] < min_humidity:
                    humidity_target_masked += 1
                    continue
                a=_prediction(frames,model.grid,record,model.pressure_pa,3)
                b=_prediction(persistence,model.grid,record,model.pressure_pa,3)
                if a is None or b is None:rejected[variable]+=1;continue
                error=float(a[0])-record['value'];control=float(b[0])-record['value']
                lead=lead_bin(record['observed_at'],issue)
                values=[abs(error),error**2,error,control**2,1]
                sums[lead,variable]+=values
                pressure_bin=int(np.abs(np.log(np.array(PRESSURE_HPA)*100)-np.log(record['pressure_pa'])).argmin())
                by_pressure[lead,pressure_bin,variable]+=values
    rows=[]
    for lead in range(24):
        for variable in range(5):
            mae,mse,bias,control,count=sums[lead,variable]
            if count:rows.append({'lead_bin_hours':(lead+1)*3,'variable':VARIABLES[variable],'count':int(count),
                'mae':mae/count,'rmse':float(np.sqrt(mse/count)),'bias':bias/count,'analysis_persistence_rmse':float(np.sqrt(control/count))})
    pressure_rows=[]
    for lead,pidx,var in zip(*np.nonzero(by_pressure[...,4])):
        mae,mse,bias,control,count=by_pressure[lead,pidx,var]
        pressure_rows.append({'lead_bin_hours':int((lead+1)*3),'pressure_bin_hpa':int(PRESSURE_HPA[pidx]),
            'variable':VARIABLES[var],'count':int(count),'mae':float(mae/count),'rmse':float(np.sqrt(mse/count)),
            'bias':float(bias/count),'analysis_persistence_rmse':float(np.sqrt(control/count))})
    if not rows:raise ValueError('No admitted diagnostic test predictions.')
    dataset.verify();_verify_climate(model)
    if hasattr(model.normalization,'verify_sources'):model.normalization.verify_sources()
    if digest(completion['identity']['norm_path'])!=completion['identity']['norm_sha256'] or _frozen_training_hashes(training)!=frozen_hashes:
        raise ValueError('Frozen training or normalization changed during diagnostic test.')
    save(output,{'split':'test','test_independence':'previously seen periods; diagnostic only','metrics':rows,
         'unsupported_records':rejected.tolist(),'humidity_target_masked':humidity_target_masked,
         'identity':completion['identity'],
         'training_artifact_sha256':frozen_hashes,'metrics_by_pressure':pressure_rows,
         'physical_policy':config.get('physical_policy'),'scientific_acceptance':False})



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
