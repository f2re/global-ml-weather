"""CPU reference trainer, immutable epoch snapshots, explicit held-out evaluation.

The trainer is not an operational weather service or proof of forecast skill.
No download, arbitrary command, pickle payload or test-set model selection occurs.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
import fcntl
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from .dataset import PreparedDataset, integer
from .io import atomic_json, artifact, digest, local_path, read_json, reference, sha256, write_arrays
from ..vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    hidden: int = 16
    latent_slots: int = 8
    horizon_hours: int = 3
    seed: int = 17
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    physics_weight: float = 0.0
    patience: int = 5
    threads: int = 1
    memory_budget_mib: int = 4096
    curriculum: tuple = ()

    def validate(self, ds):
        integer(self.epochs, 1, 10000); integer(self.hidden, 8, 512)
        if self.hidden % 4 or self.latent_slots not in (4, 8, 16, 38):
            raise ValueError('Неверные размеры скрытого состояния.')
        integer(self.seed, 0, 2**31-1); integer(self.threads, 1, 64)
        integer(self.patience, 1, 10000); integer(self.memory_budget_mib, 128, 1024*1024)
        integer(self.horizon_hours, ds.step, ds.horizon)
        if self.horizon_hours % ds.step:
            raise ValueError('Горизонт не кратен шагу.')
        for v in (self.learning_rate, self.weight_decay, self.grad_clip, self.physics_weight):
            if type(v) not in (float, int) or not math.isfinite(v):
                raise ValueError('Неконечный параметр обучения.')
        if not 0 < self.learning_rate <= 0.1 or not 0 <= self.weight_decay <= 1 or not 0 < self.grad_clip <= 100 or not 0 <= self.physics_weight <= 10:
            raise ValueError('Параметр обучения вне разрешённого диапазона.')
        last, horizon = 0, 0
        for item in self.curriculum:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError('Этап обучения требует конечную эпоху и горизонт.')
            end, h = item
            integer(end, last+1, 10000); integer(h, max(ds.step, horizon), self.horizon_hours)
            if h % ds.step:
                raise ValueError('Горизонт этапа не кратен шагу.')
            last, horizon = end, h
        if self.curriculum and self.epochs <= (self.curriculum[-2][0] if len(self.curriculum) > 1 else 0):
            raise ValueError('Число эпох не достигает последнего этапа обучения.')
        if self.curriculum and horizon != self.horizon_hours:
            raise ValueError('Последний этап должен достигать полного горизонта.')
        # Screening estimate, not a hard allocator memory guarantee.
        estimate = ds.n_cells*38*self.hidden*4*(2+self.horizon_hours//ds.step)*16
        if estimate > self.memory_budget_mib*1024**2:
            raise ValueError('Оценка активаций превышает заданный бюджет памяти.')
        return estimate

    def training_horizon(self, epoch):
        for end, h in self.curriculum:
            if epoch <= end:
                return h
        return self.horizon_hours

    def identity(self):
        return {k: v for k, v in asdict(self).items() if k != 'epochs'}


def config_from_json(value):
    if not isinstance(value, dict):
        raise ValueError('Конфигурация должна быть объектом JSON.')
    allowed = set(TrainConfig.__dataclass_fields__)
    if set(value)-allowed:
        raise ValueError('Неизвестные параметры обучения.')
    value = dict(value)
    value['curriculum'] = tuple(tuple(x) for x in value.get('curriculum', ()))
    return TrainConfig(**value)


def software():
    package = Path(__file__).resolve().parents[1]
    root = package.parent
    import sys
    return {'torch': str(torch.__version__), 'numpy': np.__version__, 'python': sys.version.split()[0],
            'scipy': importlib.metadata.version('scipy'), 'device': 'cpu',
            'source_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in sorted(package.rglob('*.py'))}}


def event(stage, **values):
    print(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'stage': stage, **values},
                     ensure_ascii=False, allow_nan=False), flush=True)


def make_model(ds, cfg):
    from ..adaptive import AdaptiveWeatherModel
    from ..grid import build_pyramid
    observations = ds.packed(ds.samples[0])
    return AdaptiveWeatherModel(build_pyramid(ds.level), observations.vocabulary,
                                observation_schema=observations.schema_fingerprint, hidden=cfg.hidden,
                                latent_slots=cfg.latent_slots, step_hours=ds.step, normalization=ds.norm)


def target_tensors(ds, sample, horizon):
    from ..training import Targets
    d = ds.targets(sample); k = horizon//ds.step+1
    indices = [i for i in range(k) if d['profile_mask'][i].any() or d['surface_mask'][i].any()]
    if not indices:
        raise ValueError('Нет пригодных целей у примера.')
    return Targets(tuple(int(d['lead_hours'][i]) for i in indices),
                   torch.as_tensor(d['profiles'][indices], dtype=torch.float32),
                   torch.as_tensor(d['profile_mask'][indices], dtype=torch.bool),
                   torch.as_tensor(d['surface'][indices], dtype=torch.float32),
                   torch.as_tensor(d['surface_mask'][indices], dtype=torch.bool),
                   ds.grid_fingerprint, torch.tensor(np.array(PRESSURE_HPA)*100, dtype=torch.float32))


class ScoreAccumulator:
    def __init__(self):
        self.rows = {}

    def add(self, lead, name, units, pred, truth, mask, area, baseline=None, pressure=None):
        valid = mask & np.isfinite(truth)
        if not np.isfinite(pred[valid]).all():
            raise FloatingPointError('Неконечный прогноз в независимой оценке.')
        if not valid.any():
            return
        e = np.asarray(pred[valid], np.float64)-np.asarray(truth[valid], np.float64)
        w = np.broadcast_to(area, pred.shape)[valid]
        key = (lead, name, pressure)
        row = self.rows.setdefault(key, {'lead_hours': lead, 'variable': name, 'units': units,
                                         'pressure_hpa': pressure, 'count': 0, 'weight': 0.,
                                         'squared_error': 0., 'absolute_error': 0., 'signed_error': 0.,
                                         'control_squared_error': 0. if baseline is not None else None})
        row['count'] += int(valid.sum()); row['weight'] += float(w.sum())
        row['squared_error'] += float((w*e**2).sum())
        row['absolute_error'] += float((w*np.abs(e)).sum()); row['signed_error'] += float((w*e).sum())
        if baseline is not None:
            b = baseline[valid].astype(np.float64)-truth[valid]
            if not np.isfinite(b).all():
                raise FloatingPointError('Неконечный контрольный прогноз.')
            row['control_squared_error'] += float((w*b*b).sum())

    def finish(self):
        result = []
        for key in sorted(self.rows, key=str):
            row = self.rows[key]; w = row['weight']
            rmse = math.sqrt(row['squared_error']/w)
            base = row['control_squared_error']
            control = math.sqrt(base/w) if base is not None else None
            result.append({k: row[k] for k in ('lead_hours', 'variable', 'units', 'pressure_hpa', 'count')} |
                          {'rmse': rmse, 'mae': row['absolute_error']/w, 'bias': row['signed_error']/w,
                           'control_rmse': control,
                           'rmse_skill': 1-rmse/control if control is not None and control > 0 else None})
        return result


def evaluate_model(ds, model, split, horizon, *, detailed=True):
    ds.assert_unchanged()
    from ..losses import forecast_loss
    samples = ds.subset(split); values = []; scores = ScoreAccumulator()
    elevation = torch.as_tensor(ds.elevation, dtype=torch.float32)
    land = torch.as_tensor(ds.land, dtype=torch.float32)
    area = torch.tensor(ds.grid().areas_m2/ds.grid().areas_m2.mean(), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            obs = ds.packed(sample); targets = target_tensors(ds, sample, horizon)
            targets.validate(model)
            initial = None
            lookup = {lead: i for i, lead in enumerate(targets.lead_hours)}
            for frame in model(obs, elevation, land, horizon_hours=horizon):
                if initial is None:
                    initial = (frame.profiles.numpy().copy(), frame.surface.numpy().copy())
                if frame.lead_hours == 0 or frame.lead_hours not in lookup:
                    continue
                i = lookup[frame.lead_hours]
                if not (targets.profile_mask[i].any() or targets.surface_mask[i].any()):
                    continue
                loss = forecast_loss(frame, targets.profiles[i], targets.profile_mask[i], targets.surface[i],
                                     targets.surface_mask[i], area, normalization=ds.norm,
                                     pressure_pa=model.pressure_pa, step_hours=ds.step)
                values.append(float(loss))
                if detailed:
                    pp, ss = frame.profiles.numpy(), frame.surface.numpy()
                    tp, ts = targets.profiles[i].numpy(), targets.surface[i].numpy()
                    pm, sm = targets.profile_mask[i].numpy(), targets.surface_mask[i].numpy()
                    for k, (name, units) in enumerate(zip(PROFILE_VARIABLES, PROFILE_UNITS)):
                        for j, pressure in enumerate(PRESSURE_HPA):
                            scores.add(frame.lead_hours, name, units, pp[:, j, k], tp[:, j, k], pm[:, j, k],
                                       area.numpy(), initial[0][:, j, k], pressure)
                    for k, (name, units) in enumerate(zip(SURFACE_VARIABLES, SURFACE_UNITS)):
                        scores.add(frame.lead_hours, name, units, ss[:, k], ts[:, k], sm[:, k], area.numpy(),
                                   initial[1][:, k] if name != 'precipitation_step' else None)
    if not values or not np.isfinite(values).all():
        raise ValueError('Нет пригодных будущих целей для оценки.')
    return {'split': split, 'data_kind': ds.kind, 'normalized_forecast_loss': float(np.mean(values)),
            'sample_ids': [s.id for s in samples], 'scores': scores.finish(),
            'control': 'persistence_of_same_model_initial_analysis; precipitation_excluded',
            'dataset_fingerprint': ds.fingerprint, 'normalization_fingerprint': ds.norm.fingerprint,
            'meteorologically_validated': False, 'independence': 'time_windows_checked; source_claims_not_certified'}


def _torch_save(path, value):
    with Path(path).open('xb') as f:
        torch.save(value, f)
        f.flush()
        import os
        os.fsync(f.fileno())


def _finite_state(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError('Неконечное состояние оптимизатора.')
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_state(item)
    elif isinstance(value, dict):
        for item in value.values():
            _finite_state(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Неконечный параметр оптимизатора.')


def _load_training_state(run, ds, cfg):
    pointer = read_json(run/'latest.json')
    state = read_json(artifact(run, pointer))
    if state.get('schema') != 'weather-training-state-1':
        raise ValueError('Неизвестная схема продолжения.')
    if state.get('dataset_fingerprint') != ds.fingerprint or state.get('config_fingerprint') != digest(cfg.identity()):
        raise ValueError('Продолжение требует той же выборки и конфигурации.')
    if state['software'] != software():
        raise ValueError('Изменились исходники или численная среда. Создайте новый эксперимент.')
    if state['data_kind'] != ds.kind:
        raise ValueError('Нельзя сменить происхождение данных при продолжении.')
    return state


def train(dataset_path, output, cfg, *, resume=False, progress=event):
    from ..training import train_step
    from ..checkpoints import save_checkpoint, load_checkpoint
    ds = PreparedDataset(dataset_path); estimate = cfg.validate(ds)
    environment = software()
    ds.subset('train'); ds.subset('validation')
    torch.set_num_threads(cfg.threads); torch.manual_seed(cfg.seed)
    for s in ds.samples:
        if s.split in ('train', 'validation'):
            ds.targets(s); ds.packed(s)
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError('Ссылка вместо каталога эксперимента.')
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError('Каталог продолжения не существует.')
    with (output/'trainer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Эксперимент уже исполняется.') from exc
        model = make_model(ds, cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        previous = 0; best = None; best_score = math.inf; stale = 0
        history = []
        if resume:
            state = _load_training_state(output, ds, cfg)
            load_checkpoint(artifact(output, state['weights'], limit=512*1024**2), model)
            opt = torch.load(artifact(output, state['optimizer'], limit=512*1024**2), map_location='cpu', weights_only=True)
            _finite_state(opt)
            optimizer.load_state_dict(opt['optimizer']); torch.set_rng_state(opt['torch_rng'])
            previous, best, best_score, stale = state['epoch'], state['best'], state['best_score'], state['stale']
            history = state['history']
            if previous >= cfg.epochs:
                raise ValueError('Нет новых эпох для продолжения.')
            if stale >= cfg.patience:
                raise ValueError('Сработала ранняя остановка; продолжение требует нового эксперимента.')
        else:
            atomic_json(output/'setup.json', {'config': asdict(cfg), 'dataset_fingerprint': ds.fingerprint,
                                             'data_kind': ds.kind, 'software': software(),
                                             'estimated_activation_bytes': estimate, 'status': 'research_training'})
        elevation, land = (torch.as_tensor(a, dtype=torch.float32) for a in (ds.elevation, ds.land))
        for epoch in range(previous+1, cfg.epochs+1):
            ds.assert_unchanged()
            if software() != environment:
                raise ValueError('Исходники изменились во время обучения.')
            begin = time.monotonic(); losses = []
            horizon = cfg.training_horizon(epoch)
            order = list(ds.subset('train')); random.Random(cfg.seed+epoch).shuffle(order)
            for index, sample in enumerate(order):
                result = train_step(model, optimizer, ds.packed(sample), elevation, land,
                                    target_tensors(ds, sample, horizon), grad_clip=cfg.grad_clip,
                                    physics_weight=cfg.physics_weight)
                losses.append(result['loss'])
                progress('training', epoch=epoch, sample=index+1, samples=len(order), loss=result['loss'])
            validation = evaluate_model(ds, model, 'validation', cfg.horizon_hours, detailed=False)
            score = validation['normalized_forecast_loss']
            improved = score < best_score
            stale = 0 if improved else stale+1
            entry = {'epoch': epoch, 'training_horizon_hours': horizon, 'validation_horizon_hours': cfg.horizon_hours,
                     'train_loss': float(np.mean(losses)), 'validation_loss': score,
                     'elapsed_seconds': time.monotonic()-begin}
            history.append(entry)
            if software() != environment:
                raise ValueError('Исходники изменились во время эпохи.')
            epochs = output/'epochs'; epochs.mkdir(exist_ok=True)
            temporary = epochs/('pending-'+uuid.uuid4().hex); temporary.mkdir()
            destination = epochs/f'{epoch:06d}'
            if destination.exists():
                raise FileExistsError('Неоднозначная эпоха после прерывания. Сохранённые данные не перезаписываются.')
            try:
                save_checkpoint(temporary/'weights.pt', model)
                _torch_save(temporary/'optimizer.pt', {'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state()})
                atomic_json(temporary/'metrics.json', entry)
                temporary.rename(destination)
            except Exception:
                raise
            if improved:
                best_score = score; best = reference(output, destination/'weights.pt')
            state = {'schema': 'weather-training-state-1', 'epoch': epoch, 'history': history,
                     'config': asdict(cfg), 'config_fingerprint': digest(cfg.identity()),
                     'dataset_fingerprint': ds.fingerprint, 'data_kind': ds.kind, 'software': software(),
                     'weights': reference(output, destination/'weights.pt'),
                     'optimizer': reference(output, destination/'optimizer.pt'),
                     'best': best, 'best_score': best_score, 'stale': stale}
            atomic_json(destination/'state.json', state)
            atomic_json(output/'latest.json', reference(output, destination/'state.json'))
            atomic_json(output/'best.json', {'weights': best, 'config': asdict(cfg), 'data_kind': ds.kind,
                                             'dataset_fingerprint': ds.fingerprint,
                                             'selection': 'validation_only', 'best_score': best_score,
                                             'software': environment,
                                             'selection_end_utc': max(
                                                 s.issue + timedelta(hours=cfg.horizon_hours)
                                                 for s in ds.samples if s.split in ('train', 'validation')).isoformat(),
                                             'meteorologically_validated': False})
            atomic_json(output/'history.json', history)
            progress('epoch_completed', **entry)
            if stale >= cfg.patience:
                break
        ds.assert_unchanged()
        report = {'schema': 'weather-training-report-1', 'status': 'trained_research', 'data_kind': ds.kind,
                  'epochs_completed': history[-1]['epoch'], 'early_stopped': stale >= cfg.patience,
                  'best_validation_loss': best_score, 'history': history,
                  'dataset_fingerprint': ds.fingerprint, 'normalization_fingerprint': ds.norm.fingerprint,
                  'test_set_used_for_selection': False, 'meteorologically_validated': False}
        atomic_json(output/'report.json', report)
        return report


def load_trained(ds, run):
    from ..checkpoints import load_checkpoint
    run = Path(run).absolute(); info = read_json(run/'best.json')
    if info.get('selection') != 'validation_only' or info.get('data_kind') != ds.kind:
        raise ValueError('Несовместимый тип обученных данных или неподдерживаемая контрольная точка.')
    if info.get('software') != software():
        raise ValueError('Исходники или численная среда отличаются от обучения.')
    cfg = config_from_json(info['config']); cfg.validate(ds)
    torch.set_num_threads(cfg.threads)
    model = make_model(ds, cfg)
    load_checkpoint(artifact(run, info['weights'], limit=512*1024**2), model)
    model.eval()
    return model, cfg, info


def evaluate(dataset_path, run, output, *, split='test'):
    if split not in ('validation', 'test'):
        raise ValueError('Оценка требует validation или test.')
    ds = PreparedDataset(dataset_path); model, cfg, trained = load_trained(ds, run)
    if split == 'test':
        from .dataset import utc
        end = utc(trained['selection_end_utc'])
        if any(s.issue-timedelta(hours=12) <= end for s in ds.subset('test')):
            raise ValueError('Итоговый тест пересекается с периодом выбора контрольной точки.')
    result = evaluate_model(ds, model, split, cfg.horizon_hours)
    result['weights'] = read_json(Path(run)/'best.json')['weights']
    result['selection_dataset_fingerprint'] = read_json(Path(run)/'best.json')['dataset_fingerprint']
    output = Path(output)
    if output.exists():
        raise FileExistsError('Отчёт не перезаписывается.')
    atomic_json(output, result)
    return result


def forecast(dataset_path, run, sample_id, output, *, horizon_hours=72):
    ds = PreparedDataset(dataset_path, inference=True)
    model, cfg, info = load_trained(ds, run)
    integer(horizon_hours, ds.step, cfg.horizon_hours)
    if horizon_hours % ds.step:
        raise ValueError('Срок не кратен обученному шагу.')
    sample = next((s for s in ds.samples if s.id == sample_id), None)
    if sample is None:
        raise ValueError('Неизвестный пример.')
    obs = ds.packed(sample)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    ds.grid().save(output/'grid.npz')
    elevation, land = (torch.as_tensor(a, dtype=torch.float32) for a in (ds.elevation, ds.land))
    leads = []; diagnostics = []
    from ..lab.metrics import physical_diagnostics
    with torch.inference_mode():
        for frame in model(obs, elevation, land, horizon_hours=horizon_hours):
            if not torch.isfinite(frame.profiles).all() or not torch.isfinite(frame.surface).all():
                raise FloatingPointError('Неконечный прогноз.')
            write_arrays(output/f'frame_{frame.lead_hours:03d}.npz', profiles=frame.profiles.numpy(),
                         surface=frame.surface.numpy(), profile_mask=frame.profile_mask.numpy(),
                         surface_mask=frame.surface_mask.numpy(), pressure_hpa=np.array(PRESSURE_HPA),
                         profile_variables=np.array(PROFILE_VARIABLES), profile_units=np.array(PROFILE_UNITS),
                         surface_variables=np.array(SURFACE_VARIABLES), surface_units=np.array(SURFACE_UNITS),
                         issue_time=sample.issue.isoformat(), valid_time=frame.valid_time.isoformat())
            leads.append(frame.lead_hours)
            diagnostics.append({'lead_hours': frame.lead_hours, **physical_diagnostics(
                frame.profiles.numpy(), frame.surface.numpy(), model.pressure_pa.numpy(), ds.grid().areas_m2)})
    ds.assert_unchanged()
    report = {'status': 'research_forecast', 'data_kind': ds.kind, 'sample_id': sample.id,
              'issue_time': sample.issue.isoformat(), 'lead_hours': leads,
              'spec': {'kind': 'forecast_dataset', 'horizon_hours': horizon_hours, 'mesh_level': ds.level,
                       'hidden': cfg.hidden, 'latent_slots': cfg.latent_slots},
              'cells': ds.n_cells, 'pressure_hpa': list(PRESSURE_HPA), 'diagnostics': diagnostics,
              'profile_variables': list(PROFILE_VARIABLES), 'profile_units': list(PROFILE_UNITS),
              'surface_variables': list(SURFACE_VARIABLES), 'surface_units': list(SURFACE_UNITS),
              'dataset_fingerprint': ds.fingerprint, 'weights': info['weights'],
              'selection_end_utc': info['selection_end_utc'],
              'retrospective_overlap_with_selection': sample.issue.isoformat() <= info['selection_end_utc'],
              'normalization_fingerprint': ds.norm.fingerprint, 'targets_read': False,
              'meteorologically_validated': False}
    atomic_json(output/'report.json', report)
    atomic_json(output/'artifacts.json', {p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    return report
