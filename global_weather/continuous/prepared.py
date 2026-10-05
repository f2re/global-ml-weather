"""Use C2 step persistence with the existing weather model on a local block.

This bridge is for admitted prepared data, not the automatic C3-C5 provider
pipeline. It performs no independent evaluation or satellite migration.
"""
from __future__ import annotations

from dataclasses import asdict
import random
import numpy as np
import torch

from .checkpointing import StepTrainer
from .contracts import fingerprint
from ..devices import select_device
from ..pipeline.dataset import PreparedDataset
from ..pipeline.io import artifact
from ..pipeline.runner import TrainConfig, make_model, target_tensors
from ..training import train_step


def train_prepared_block(store, dataset_path, *, config=None, sample_ids=None, pass_number=0,
                         max_steps=None, cancelled=None, failpoint=None):
    ds = PreparedDataset(dataset_path)
    state = store.state()
    if state['campaign'] is None:
        raise ValueError('Сначала зарегистрируйте диапазон постоянной программы.')
    contract = state['campaign']['contract']
    cfg = config or TrainConfig(hidden=contract['hidden'], latent_slots=contract['latent_slots'],
                                horizon_hours=contract['horizon_hours'])
    cfg.validate(ds)
    if (ds.level != contract['mesh_level'] or ds.step != contract['step_hours']
            or ds.horizon != contract['horizon_hours'] or cfg.horizon_hours != contract['horizon_hours']
            or cfg.hidden != contract['hidden'] or cfg.latent_slots != contract['latent_slots']):
        raise ValueError('Подготовленный блок и модель не совпадают с постоянной программой.')
    if ds.manifest.get('multimodal') is not None:
        raise ValueError('Миграция спутниковых ветвей требует C6–C7.')
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError('Предел шагов должен быть положительным целым числом.')
    train = ds.subset('train')
    if sample_ids is not None:
        if not isinstance(sample_ids, list) or not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError('Нужен непустой список разных обучающих примеров.')
        by_id = {s.id: s for s in train}
        if any(i not in by_id for i in sample_ids):
            raise ValueError('Разрешены только примеры обучающей части.')
        train = [by_id[i] for i in sample_ids]
    train.sort(key=lambda s: s.issue)
    # The runtime identity is fixed across blocks, not tied to date boundaries.
    identity = {'schema': 'prepared-continuous-1', 'data_kind': ds.kind,
                'grid_fingerprint': ds.grid_fingerprint, 'normalization_fingerprint': ds.norm.fingerprint,
                'static_sha256': ds.manifest['static']['sha256'],
                'registry_sha256': ds.manifest['registry']['sha256'],
                'optimizer': {k: v for k, v in asdict(cfg).items() if k not in ('epochs', 'patience', 'curriculum')}}
    use_ids = []
    for sample in train:
        ds.targets(sample)
        ds.packed(sample)
        def source(ref, kind):
            artifact(ds.root, ref)
            return {'provider': 'prepared_'+ds.kind, 'object_id': kind+'/'+ref['sha256'],
                    'revision': '1', 'sha256': ref['sha256']}
        row = store.register_sample({'issue_time': sample.issue.isoformat(),
            'inputs': [source(sample.observations, 'observations')],
            'targets': [source(sample.targets, 'targets')], 'modalities': ['prepared_atmosphere'],
            'transform_sha256': fingerprint(identity)})
        use_ids.append(store.plan_use(row['id'], pass_number=pass_number)['id'])
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device(cfg.device)
    model = make_model(ds, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elevation = torch.as_tensor(ds.elevation, dtype=torch.float32, device=device)
    land = torch.as_tensor(ds.land, dtype=torch.float32, device=device)
    completed, skipped = 0, 0
    with StepTrainer(store, model, optimizer, identity=identity, failpoint=failpoint) as trainer:
        for index, (sample, use_id) in enumerate(zip(train, use_ids)):
            if cancelled and cancelled():
                break
            def run():
                ds.assert_unchanged()
                result = train_step(model, optimizer, ds.packed(sample).to(device), elevation, land,
                                    target_tensors(ds, sample, cfg.horizon_hours).to(device),
                                    grad_clip=cfg.grad_clip, physics_weight=cfg.physics_weight)
                ds.assert_unchanged()
                return result
            result = trainer.step([use_id], run, cursor={'block_id': ds.fingerprint, 'next_index': index+1,
                                  'order_sha256': fingerprint(use_ids), 'pass_number': pass_number})
            if result['status'] == 'committed':
                completed += 1
            else:
                skipped += 1
            if max_steps is not None and completed >= max_steps:
                break
        current = store.state()['checkpoint']
    return {'status': 'prepared_block_processed' if completed+skipped == len(train) else 'paused_at_saved_step',
            'data_kind': ds.kind, 'new_steps': completed, 'already_committed': skipped,
            'checkpoint': current, 'automatic_provider_pipeline': False,
            'independent_evaluation_performed': False, 'meteorologically_validated': False}
