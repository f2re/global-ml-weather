"""S2 measured upper-air pilot, with observation-space losses and no ERA5 reads.

This is a geometry-only research ablation. Five physical profile channels are
learned from measured train records. Omega, surface fields and terrain remain
unsupported and explicitly masked; full atmospheric acceptance is blocked.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sqlite3
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .analysis.observation_operator import _prediction, predict_observations
from .grid import build_pyramid, unit_xyz
from .model import ForecastFrame, GraphOps
from .observation_training import checkpoint_path, digest, save, sync_directory
from .observations import utc
from .vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS

VARIABLES = PROFILE_VARIABLES[:5]
START = utc('2021-01-01T00:00:00Z')
TRAIN_END = utc('2022-01-01T00:00:00Z')
VAL_END = utc('2022-07-01T00:00:00Z')
END = utc('2023-01-01T00:00:00Z')
DEFAULT_CONFIG = {'mesh_level': 1, 'hidden': 16, 'epochs': 20, 'patience': 5,
                  'threads': 4, 'seed': 29, 'learning_rate': 1e-4,
                  'weight_decay': .01, 'gradient_clip': 1.,
                  'max_train_issues': 48, 'max_validation_issues': 16,
                  'max_test_issues': 16, 'max_records_per_window': 2000}


def checked(record):
    if (not isinstance(record, dict) or record.get('source') != 'radiosonde'
            or record.get('provider') != 'NOAA_IGRA2'
            or record.get('data_kind') not in (None,'real')
            or record.get('valid') is not True):
        raise ValueError('Only admitted real IGRA radiosonde scalars are supported.')
    name = record.get('variable')
    if name not in VARIABLES:
        raise ValueError('Unsupported S2 measurement variable; omega is not derived.')
    index = VARIABLES.index(name)
    if record.get('units') != PROFILE_UNITS[index]:
        raise ValueError('Profile measurement units differ.')
    values = [record.get(key, np.nan) for key in ('value', 'pressure_pa', 'latitude', 'longitude')]
    if not np.isfinite(values).all() or not 1 <= values[1] <= 120000 or abs(values[2]) > 90 or abs(values[3]) > 180:
        raise ValueError('Invalid actual pressure, geometry or value.')
    observed, available = utc(record['observed_at']), utc(record['available_at'])
    if available < observed or not record.get('observation_id'):
        raise ValueError('Invalid measurement identity or availability.')
    if record.get('group_split') not in ('train', 'validation', 'val', 'test') or not record.get('profile_id'):
        raise ValueError('Need full-profile split admission and profile identity.')
    if (type(record.get('revision',0)) is not int or record.get('revision', 0) != 0 or record.get('withdrawn', False) is not False
            or record.get('retracted', False) is not False):
        raise ValueError('Pilot admits only immutable revision zero; historical revision replay is unsupported.')
    if (not isinstance(record.get('provider_qc'), dict)
            or any(not isinstance(record.get(key), str) or len(record[key]) != 64
                   for key in ('archive_sha256', 'format_sha256'))):
        raise ValueError('Need actual IGRA QC and archive/format provenance.')
    if record.get('time_basis') not in ('reported_level_time','launch_plus_elapsed_seconds','launch_time_fallback','nominal_time_fallback'):
        raise ValueError('Need explicit actual or assumed level-time basis.')
    return index, observed, available


def bounded_records(records, limit):
    """Fixed bounded-pilot sampling by variable, actual pressure and identity.

    This computational limit is declared before model fitting and does not use
    errors, validation or ERA5. Original measured records remain in the archive.
    """
    if len(records) <= limit:
        return records
    groups = {}
    for record in records:
        key = (record['variable'], int(np.log(record['pressure_pa']) * 3))
        groups.setdefault(key, []).append(record)
    for rows in groups.values():
        rows.sort(key=lambda row: hashlib.sha256(row['observation_id'].encode()).hexdigest())
    result = []
    ordered = sorted(groups)
    index = 0
    while len(result) < limit:
        added = False
        for key in ordered:
            if index < len(groups[key]):
                result.append(groups[key][index]); added = True
                if len(result) == limit: break
        if not added: break
        index += 1
    return result


def prepare(source, output):
    source, output = Path(source), Path(output)
    if source.is_symlink() or source.suffix != '.jsonl':
        raise ValueError('Need immutable normalized IGRA JSONL.')
    source_hash = digest(source)
    admission_path=source.parent/'manifest.json'
    if not admission_path.is_file() or admission_path.is_symlink():
        raise ValueError('Need companion immutable IGRA admission manifest.')
    admission=json.loads(admission_path.read_text())
    if (admission.get('schema')!='igra-observation-archive-1' or admission.get('provider')!='NOAA_IGRA2'
            or admission.get('data_kind')!='real' or admission.get('observations_sha256')!=source_hash):
        raise ValueError('IGRA admission receipt does not match actual normalized bytes.')
    admission_hash=digest(admission_path)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'observations.sqlite'
    manifest_path = output / 'dataset.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if (old['source_sha256'] != source_hash or old['database_sha256'] != digest(destination)
                or old.get('admission_sha256')!=admission_hash):
            raise ValueError('Prepared profile data changed.')
        return old
    if destination.exists() or (output / 'observations.tmp.sqlite').exists():
        raise ValueError('Ambiguous unfinished profile preparation; preserve for inspection.')
    temporary = output / 'observations.tmp.sqlite'
    count = 0
    with sqlite3.connect(temporary) as database:
        database.execute('CREATE TABLE records(id TEXT PRIMARY KEY, observed TEXT, available TEXT, variable INTEGER, value REAL, revision INTEGER, split TEXT, record TEXT)')
        database.execute('CREATE TABLE profile_groups(id TEXT PRIMARY KEY, split TEXT, earliest TEXT, latest TEXT)')
        with source.open() as stream:
            for line in stream:
                if not line.strip(): continue
                record = json.loads(line)
                variable, observed, available = checked(record)
                if not START <= observed < END: continue
                text = json.dumps(record, sort_keys=True, allow_nan=False)
                revision = record.get('revision', 0)
                if type(revision) is not int or revision < 0: raise ValueError('Invalid measurement revision.')
                old = database.execute('SELECT revision,record FROM records WHERE id=?', (record['observation_id'],)).fetchone()
                if old and old[0] == revision and old[1] != text: raise ValueError('Conflicting scalar revision.')
                if old is None or revision > old[0]:
                    split='validation' if record['group_split']=='val' else record['group_split']
                    expected_split='train' if observed<TRAIN_END else 'validation' if observed<VAL_END else 'test'
                    if split!=expected_split: raise ValueError('Profile group crosses chronological split.')
                    group=database.execute('SELECT split,earliest,latest FROM profile_groups WHERE id=?',(record['profile_id'],)).fetchone()
                    earliest=min(observed,utc(group[1])) if group else observed
                    latest=max(observed,utc(group[2])) if group else observed
                    if group and (group[0]!=split or latest-earliest>timedelta(hours=24)):
                        raise ValueError('Full-profile identity has inconsistent split or time support.')
                    if ((split=='train' and latest>=TRAIN_END-timedelta(hours=42))
                            or (split=='validation' and (earliest<TRAIN_END+timedelta(hours=42) or latest>=VAL_END-timedelta(hours=42)))
                            or (split=='test' and earliest<VAL_END+timedelta(hours=42))):
                        raise ValueError('Full-profile group violates admission guard.')
                    database.execute('INSERT OR REPLACE INTO profile_groups VALUES(?,?,?,?)',
                                     (record['profile_id'],split,earliest.isoformat(),latest.isoformat()))
                    database.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?,?,?,?,?)',
                        (record['observation_id'], observed.isoformat(), available.isoformat(), variable, record['value'], revision, split, text))
                count += 1
        database.execute('CREATE INDEX observed_index ON records(observed)')
        database.execute('CREATE INDEX available_index ON records(available)')
        statistics = []
        for variable in range(5):
            # Each admitted identity contributes once; no duplicate forecast windows.
            rows = database.execute("SELECT value,record FROM records WHERE variable=? AND split='train' AND observed>=? AND observed<?",
                                    (variable, START.isoformat(), TRAIN_END.isoformat()))
            n, mean, m2 = 0, 0., 0.
            minimum_pressure,maximum_pressure=float('inf'),0.
            for value,text in rows:
                n += 1; delta = value - mean; mean += delta / n; m2 += delta * (value - mean)
                pressure=json.loads(text)['pressure_pa']
                minimum_pressure=min(minimum_pressure,pressure); maximum_pressure=max(maximum_pressure,pressure)
            if n < 2 or m2 <= 0: raise ValueError('Missing empirical train statistics for ' + VARIABLES[variable])
            statistics.append({'variable': VARIABLES[variable], 'units': PROFILE_UNITS[variable],
                               'count': n, 'mean': mean, 'std': float(np.sqrt(m2 / n)),
                               'minimum_pressure_pa':minimum_pressure,'maximum_pressure_pa':maximum_pressure})
        unique_count = database.execute('SELECT count(*) FROM records').fetchone()[0]
    if digest(source) != source_hash or digest(admission_path)!=admission_hash:
        raise ValueError('Source or admission changed while preparing profiles.')
    temporary.rename(destination)
    manifest = {'schema': 'measured-profile-pilot-1', 'data_kind': 'real', 'source': str(source.resolve()),
                'source_sha256': source_hash, 'database_sha256': digest(destination), 'records': unique_count,
                'admission_manifest':str(admission_path.resolve()),'admission_sha256':admission_hash,
                'norm_period': [START.isoformat(), TRAIN_END.isoformat()], 'statistics': statistics,
                'norm_interpretation': 'pooled actual train pressures; not empirical statistics on 37 standard levels',
                'source_roles': {'input': 'IGRA', 'target': 'IGRA', 'normalization': 'train IGRA', 'static': None},
                'scientific_acceptance': False, 'geometry_only_ablation': True, 'omega_supervision': False}
    save(manifest_path, manifest)
    return manifest


class ProfileDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest_sha256=digest(self.root/'dataset.json')
        self.manifest = json.loads((self.root / 'dataset.json').read_text())
        if self.manifest.get('schema') != 'measured-profile-pilot-1' or self.manifest.get('data_kind')!='real':
            raise ValueError('Wrong profile dataset schema or source kind.')
        if (self.manifest.get('source_roles')!={'input':'IGRA','target':'IGRA','normalization':'train IGRA','static':None}
                or self.manifest.get('norm_period')!=[START.isoformat(),TRAIN_END.isoformat()]):
            raise ValueError('Profile source roles or norm period differ.')
        statistics=self.manifest.get('statistics')
        if (not isinstance(statistics,list) or len(statistics)!=5
                or any(item.get('variable')!=VARIABLES[i] or item.get('units')!=PROFILE_UNITS[i]
                       or type(item.get('count')) is not int or item['count']<2 for i,item in enumerate(statistics))):
            raise ValueError('Profile train statistics schema differs.')
        self.mean = np.array([item['mean'] for item in self.manifest['statistics']], dtype=np.float32)
        self.std = np.array([item['std'] for item in self.manifest['statistics']], dtype=np.float32)
        self.pressure_bounds=np.array([[item['minimum_pressure_pa'],item['maximum_pressure_pa']]
                                      for item in self.manifest['statistics']],dtype=np.float32)
        if not np.isfinite(self.pressure_bounds).all() or (self.pressure_bounds[:,0]>self.pressure_bounds[:,1]).any():
            raise ValueError('Invalid empirical train pressure coverage.')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std<=0).any():
            raise ValueError('Invalid profile train statistics.')
        self.verify()

    def verify(self):
        if self.root.is_symlink() or Path(self.manifest['source']).is_symlink() or (self.root/'observations.sqlite').is_symlink():
            raise ValueError('Profile source/dataset symlinks are forbidden.')
        if (digest(self.root/'dataset.json')!=self.manifest_sha256
                or digest(self.manifest['admission_manifest'])!=self.manifest['admission_sha256']
                or digest(self.manifest['source']) != self.manifest['source_sha256']
                or digest(self.root / 'observations.sqlite') != self.manifest['database_sha256']):
            raise ValueError('Measured profile sources or prepared data changed.')

    def records(self, start, end, *, issue=None, split=None):
        uri = (self.root / 'observations.sqlite').resolve().as_uri() + '?mode=ro'
        query = 'SELECT record FROM records WHERE observed>? AND observed<=?'
        parameters = [utc(start).isoformat(), utc(end).isoformat()]
        if issue is not None: query += ' AND available<=?'; parameters.append(utc(issue).isoformat())
        if split is not None: query += ' AND split=?'; parameters.append(split)
        query += ' ORDER BY observed,id'
        with sqlite3.connect(uri, uri=True) as database:
            return [json.loads(row[0]) for row in database.execute(query, parameters)]

    def issues(self, split, limit):
        boundaries = {'train': (START, TRAIN_END), 'validation': (TRAIN_END, VAL_END), 'test': (VAL_END, END)}
        start, end = boundaries[split]
        # All 12-hour inputs and +72-hour targets stay within their own part.
        rows = []; issue = start + timedelta(hours=54)
        while issue + timedelta(hours=114) < end:
            rows.append(issue); issue += timedelta(days=1)
        if len(rows) > limit:
            rows = [rows[i] for i in np.linspace(0, len(rows)-1, limit, dtype=int)]
        return rows

    def sample(self, issue, limit):
        split='train' if issue<TRAIN_END else 'validation' if issue<VAL_END else 'test'
        return (bounded_records(self.records(issue-timedelta(hours=12), issue, issue=issue, split=split), limit),
                bounded_records(self.records(issue, issue+timedelta(hours=72), split=split), limit))


class ProfileObservationModel(nn.Module):
    def __init__(self, grid, mean, std, hidden=16, pressure_bounds=None):
        super().__init__()
        self.grid = grid; self.hidden = hidden
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        if self.mean.shape != (5,) or self.std.shape != (5,) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Need five actual-measurement train statistics.')
        self.register_buffer('pressure_pa', torch.tensor(PRESSURE_HPA, dtype=torch.float32) * 100)
        if pressure_bounds is None: raise ValueError('Empirical train pressure support is required.')
        bounds=torch.as_tensor(pressure_bounds,dtype=torch.float32).clone()
        if bounds.shape!=(5,2) or not torch.isfinite(bounds).all() or (bounds[:,0]>bounds[:,1]).any():
            raise ValueError('Need empirical pressure support for each profile variable.')
        self.register_buffer('pressure_bounds',bounds)
        self.register_buffer('xyz', torch.tensor(grid.xyz, dtype=torch.float32))
        self.geometry = nn.Linear(3, hidden); self.levels = nn.Parameter(torch.randn(37, hidden)*.01)
        self.variable = nn.Embedding(5, hidden)
        self.token = nn.Sequential(nn.Linear(hidden+6, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.ingest = nn.GRUCell(hidden, hidden); self.graph = GraphOps(grid)
        self.dynamic = nn.Linear(hidden*3, hidden); self.step = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 5)

    def _decode(self, state, issue, lead):
        values = self.head(state) * self.std + self.mean
        # Humidity is positive without substituting any missing measured value.
        humidity = self.std[1] * F.softplus(values[..., 1] / self.std[1])
        values = torch.stack([values[..., 0], humidity, values[..., 2], values[..., 3], values[..., 4]], -1)
        profiles = torch.cat([values, values.new_full((*values.shape[:-1], 1), float('nan'))], -1)
        support=(self.pressure_pa[:,None]>=self.pressure_bounds[:,0]) & (self.pressure_pa[:,None]<=self.pressure_bounds[:,1])
        mask=torch.cat([support,torch.zeros(37,1,dtype=torch.bool,device=values.device)],-1)[None].expand_as(profiles)
        profiles=torch.where(mask,profiles,torch.full_like(profiles,float('nan')))
        frame = ForecastFrame(lead, issue+timedelta(hours=lead), profiles,
                              values.new_full((self.grid.n_cells, 8), float('nan')),
                              mask[..., :5].any(-1), torch.zeros(self.grid.n_cells, 8, dtype=torch.bool, device=values.device))
        frame.profile_variable_mask = mask
        return frame

    def forward(self, inputs, issue):
        issue = utc(issue); n, d = self.grid.n_cells, self.hidden
        state = self.geometry(self.xyz)[:, None] + self.levels[None]
        for hour in range(12):
            rows = [r for r in inputs if issue-timedelta(hours=12-hour) < utc(r['observed_at']) <= issue-timedelta(hours=11-hour)]
            rows = [r for r in rows if utc(r['available_at']) <= issue]
            if not rows: continue
            variables = torch.tensor([VARIABLES.index(r['variable']) for r in rows], dtype=torch.long, device=state.device)
            pressures = np.array([r['pressure_pa'] for r in rows])
            levels = np.abs(np.log(pressures[:, None]) - np.log(np.array(PRESSURE_HPA)[None]*100)).argmin(1)
            cells = self.grid.locate([r['latitude'] for r in rows], [r['longitude'] for r in rows])
            indices = torch.tensor(cells*37+levels, dtype=torch.long, device=state.device)
            values = torch.tensor([r['value'] for r in rows], dtype=state.dtype, device=state.device)
            xyz = unit_xyz([r['latitude'] for r in rows], [r['longitude'] for r in rows])
            extra = torch.tensor(np.column_stack([np.log(pressures/100000),
                    [(issue-utc(r['observed_at'])).total_seconds()/43200 for r in rows], xyz]), dtype=state.dtype, device=state.device)
            features = torch.cat([((values-self.mean[variables])/self.std[variables])[:, None], extra], -1)
            tokens = self.token(torch.cat([features, self.variable(variables)], -1))
            sums = state.new_zeros(n*37, d).index_add(0, indices, tokens)
            counts = state.new_zeros(n*37).index_add(0, indices, torch.ones_like(values))
            updated = self.ingest(sums/counts.clamp_min(1)[:, None], state.reshape(-1, d))
            state = torch.where((counts>0)[:, None], updated, state.reshape(-1,d)).reshape(n,37,d)
        frames = [self._decode(state, issue, 0)]
        for lead in range(3, 73, 3):
            context = torch.cat([state, self.graph.neighbours(state), state.mean(1, keepdim=True).expand_as(state)], -1)
            state = self.step(torch.tanh(self.dynamic(context)).reshape(-1,d), state.reshape(-1,d)).reshape(n,37,d)
            frames.append(self._decode(state, issue, lead))
        return frames


def profile_loss(model, frames, targets):
    terms = [[] for _ in VARIABLES]
    for record in targets:
        if record['variable'] not in VARIABLES: continue
        result = _prediction(frames, model.grid, record, model.pressure_pa, 3)
        if result is None: continue
        variable = VARIABLES.index(record['variable'])
        predicted = result[0]
        if not torch.isfinite(predicted): raise FloatingPointError('Nonfinite observation-space prediction.')
        terms[variable].append(((predicted-record['value'])/model.std[variable]).square())
    available = [torch.stack(rows).mean() for rows in terms if rows]
    if not available: raise ValueError('No admitted future profile targets.')
    return torch.stack(available).mean(), [len(rows) for rows in terms]


def configuration(config):
    result = dict(DEFAULT_CONFIG); result.update(config)
    if set(result) != set(DEFAULT_CONFIG): raise ValueError('Unknown profile training configuration.')
    for key in ('hidden','epochs','patience','threads','max_train_issues','max_validation_issues','max_test_issues','max_records_per_window'):
        if type(result[key]) is not int or result[key] <= 0: raise ValueError('Invalid profile training bound: '+key)
    if result['epochs'] > 100 or result['mesh_level'] not in (0,1,2) or result['max_records_per_window'] > 20000:
        raise ValueError('Profile pilot exceeds declared research limits.')
    return result


def identity(dataset, config, device):
    return {'dataset_sha256': digest(dataset.root/'dataset.json'), 'source_sha256': dataset.manifest['source_sha256'],
            'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
            'config': config, 'device': str(device), 'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'cuda': torch.version.cuda,
            'hardware': torch.cuda.get_device_name(device) if device.type=='cuda' else platform.machine(),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'deterministic':True, 'tf32':False}


def score(model, dataset, config, split):
    values=[]; model.eval()
    with torch.no_grad():
        for issue in dataset.issues(split, config['max_'+split+'_issues']):
            inputs, targets = dataset.sample(issue, config['max_records_per_window'])
            if not inputs or not targets: continue
            objective, _ = profile_loss(model, model(inputs,issue), targets)
            values.append(float(objective))
    if not values: raise ValueError('No observed profile '+split+' examples.')
    return float(np.mean(values))


def train(dataset_path, output, config):
    config=configuration(config); dataset=ProfileDataset(dataset_path); output=Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output/'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(config['threads'])
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        model=ProfileObservationModel(build_pyramid(config['mesh_level'])[0],dataset.mean,dataset.std,config['hidden'],dataset.pressure_bounds).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        expected=identity(dataset,config,device); completed=0; best=float('inf'); stale=0; best_epoch=None
        if (output/'latest.json').exists():
            ref=json.loads((output/'latest.json').read_text()); path=checkpoint_path(output,ref)
            state=torch.load(path,map_location=device,weights_only=True)
            if state['identity']!=expected: raise ValueError('Incompatible profile resume source/data/numerics.')
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            completed,best,stale,best_epoch=state['epoch'],state['best'],state['stale'],state['best_epoch']
            random.setstate(state['python_rng']); rng=state['numpy_rng']; np.random.set_state((rng[0],np.asarray(rng[1],dtype=np.uint32),*rng[2:]))
            torch.set_rng_state(state['torch_rng'].cpu())
            if device.type=='cuda': torch.cuda.set_rng_state_all([row.cpu() for row in state['cuda_rng']])
            if state['best_epoch']==completed:
                save(output/'best.json',ref)
            elif state.get('best_ref') is not None:
                checkpoint_path(output,state['best_ref']); save(output/'best.json',state['best_ref'])
            else: raise ValueError('Missing compatible best profile checkpoint reference.')
        if any(output.glob('.epoch-*')) or (output/f'epoch-{completed+1:04d}').exists():
            raise ValueError('Ambiguous interrupted profile epoch; preserve for inspection.')
        for epoch in range(completed+1,config['epochs']+1):
            dataset.verify(); started=time.monotonic(); model.train(); objectives=[]; gradient={}; coverage=np.zeros(5,dtype=int)
            issues=dataset.issues('train',config['max_train_issues']); random.shuffle(issues)
            for issue in issues:
                inputs,targets=dataset.sample(issue,config['max_records_per_window'])
                if not inputs or not targets: continue
                optimizer.zero_grad(set_to_none=True); frames=model(inputs,issue); objective,counts=profile_loss(model,frames,targets)
                if not torch.isfinite(objective): raise FloatingPointError('Nonfinite profile loss.')
                objective.backward(); coverage+=counts
                for name,parameter in model.named_parameters():
                    if parameter.grad is None: gradient.setdefault(name,False); continue
                    if not torch.isfinite(parameter.grad).all(): raise FloatingPointError('Nonfinite profile gradient: '+name)
                    gradient[name]=gradient.get(name,False) or bool(parameter.grad.abs().sum()>0)
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                optimizer.step(); objectives.append(float(objective.detach()))
            if not objectives or not all(gradient.values()) or not np.all(coverage>0):
                raise ValueError('Incomplete measured profile gradient/variable coverage; acceptance blocked.')
            validation=score(model,dataset,config,'validation'); improved=validation<best
            if improved: best,best_epoch,stale=validation,epoch,0
            else: stale+=1
            dataset.verify(); temporary=output/f'.epoch-{epoch:04d}'; temporary.mkdir()
            rng=np.random.get_state(); state={'identity':expected,'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,
                'best_ref':None if improved else json.loads((output/'best.json').read_text()),
                'model':model.state_dict(),'optimizer':optimizer.state_dict(),'python_rng':random.getstate(),
                'numpy_rng':(rng[0],rng[1].tolist(),*rng[2:]),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
            torch.save(state,temporary/'state.pt')
            with (temporary/'state.pt').open('rb') as file: os.fsync(file.fileno())
            report={'epoch':epoch,'train_loss':float(np.mean(objectives)),'validation_loss':validation,
                    'seconds':time.monotonic()-started,'nonzero_gradients':gradient,'measured_variable_counts':coverage.tolist(),
                    'scientific_acceptance':False,'omega_supervision':False}
            save(temporary/'metrics.json',report); sync_directory(temporary)
            final=output/f'epoch-{epoch:04d}'; temporary.rename(final); sync_directory(output)
            ref={'directory':final.name,'sha256':digest(final/'state.pt'),'epoch':epoch}; save(output/'latest.json',ref)
            if improved: save(output/'best.json',ref)
            print(json.dumps(report),flush=True)
            if stale>=config['patience']: break
        save(output/'complete.json',{'identity':expected,'best_epoch':best_epoch,'scientific_acceptance':False,
             'status':'measured_upper_air_research_trained','limitations':['omega unobserved','terrain absent','surface unsupported','pooled train pressure statistics']})


def load_model(dataset_path, training, device='auto'):
    dataset=ProfileDataset(dataset_path); training=Path(training); ref=json.loads((training/'best.json').read_text())
    path=checkpoint_path(training,ref)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    state=torch.load(path,map_location=device,weights_only=True)
    config=state['identity']['config']; torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
    if state['identity']!=identity(dataset,config,device): raise ValueError('Profile evaluation identity differs.')
    model=ProfileObservationModel(build_pyramid(config['mesh_level'])[0],dataset.mean,dataset.std,config['hidden'],dataset.pressure_bounds).to(device)
    model.load_state_dict(state['model']); model.eval()
    return dataset,model,config,ref


def load_frozen(dataset_path, training, device='auto'):
    """Public frozen-model interface for a separate external verification job."""
    dataset,model,_,_=load_model(dataset_path,training,device=device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model,dataset


def evaluate(dataset_path, training, output):
    dataset,model,config,ref=load_model(dataset_path,training); sums=np.zeros((24,5,4)); rejections={}; attempted=np.zeros(5,dtype=int)
    with torch.no_grad():
        for issue in dataset.issues('test',config['max_test_issues']):
            inputs,targets=dataset.sample(issue,config['max_records_per_window'])
            if not inputs or not targets: continue
            predictions,rejected=predict_observations(model(inputs,issue),model.grid,model.pressure_pa,targets)
            if any(reason!='outside_supported_forecast' for reason in rejected):
                raise ValueError('Invalid final-test observations: '+json.dumps(rejected))
            for reason,count in rejected.items(): rejections[reason]=rejections.get(reason,0)+count
            for row in targets: attempted[VARIABLES.index(row['variable'])]+=1
            for row in predictions:
                lead=min(23,int((utc(row.observed_at)-issue).total_seconds()/10800)); variable=VARIABLES.index(row.variable)
                error=row.predicted-row.observed; sums[lead,variable]+=[abs(error),error**2,error,1]
    metrics=[[None if not cell[3] else {'mae':cell[0]/cell[3],'rmse':float(np.sqrt(cell[1]/cell[3])),
              'bias':cell[2]/cell[3],'count':int(cell[3])} for cell in rows] for rows in sums]
    if not sums[...,3].sum(): raise ValueError('No supported measured final-test observations.')
    save(output,{'split':'test','checkpoint':ref,'variables':list(VARIABLES),'units':list(PROFILE_UNITS[:5]),
                 'lead_bins_hours':list(range(3,73,3)),'metrics':metrics,'attempted_per_variable':attempted.tolist(),
                 'rejected':rejections,'scientific_acceptance':False})


def forecast(dataset_path, training, issue, output):
    dataset,model,config,ref=load_model(dataset_path,training); issue=utc(issue)
    # Deliberately query inputs only: inference never loads future test targets.
    inputs=bounded_records(dataset.records(issue-timedelta(hours=12),issue,issue=issue),config['max_records_per_window'])
    with torch.no_grad(): frames=model(inputs,issue)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    arrays={'profiles':np.stack([frame.profiles.cpu().numpy() for frame in frames]),
            'profile_variable_mask':np.stack([frame.profile_variable_mask.cpu().numpy() for frame in frames]),
            'pressure_pa':model.pressure_pa.cpu().numpy(),'xyz':model.grid.xyz,'lead_hours':np.arange(0,73,3)}
    np.savez_compressed(output/'forecast.npz',**arrays)
    save(output/'forecast.json',{'issue':issue.isoformat(),'checkpoint':ref,'sha256':digest(output/'forecast.npz'),
         'scientific_acceptance':False,'omega_output':'NaN with false variable mask','future_targets_read':False})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('train'); p.add_argument('--dataset',required=True); p.add_argument('--output',required=True); p.add_argument('--config')
    for command in ('test','forecast'):
        p=sub.add_parser(command); p.add_argument('--dataset',required=True); p.add_argument('--training',required=True); p.add_argument('--output',required=True)
        if command=='forecast': p.add_argument('--issue',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare': prepare(args.source,args.output)
    elif args.command=='train': train(args.dataset,args.output,{} if not args.config else json.loads(Path(args.config).read_text()))
    elif args.command=='test': evaluate(args.dataset,args.training,args.output)
    else: forecast(args.dataset,args.training,args.issue,args.output)


if __name__=='__main__': main()
