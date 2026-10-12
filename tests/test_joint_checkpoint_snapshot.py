"""Completed epoch snapshots preserve a running parent, not weather acceptance."""
import json

import numpy as np
import pytest
import torch

from global_weather.joint_checkpoint_snapshot import freeze_best_completed_epoch, STATUS
from global_weather.observation_training import checkpoint_path, digest


def parent_fixture(tmp_path):
    parent = tmp_path / 'parent'
    folder = parent / 'epoch-0001'
    folder.mkdir(parents=True)
    identity = {'architecture': 'joint-measured-hydrostatic-v1', 'config': {'epochs': 12},
                'joint_fingerprint': {'horizon_hours': 72}}
    state = {'identity': identity, 'epoch': 1, 'best_epoch': 1, 'best': .4,
             'model': {'head.weight': torch.ones(5, 2)},
             'trained_variable_mask': np.ones((37, 5), dtype=bool).tolist()}
    metrics = {'epoch': 1, 'nonzero_gradients': {'head.weight': True},
               'profile_target_counts': [10]*5, 'native_target_counts': [10]*6,
               'train_loss': .5, 'validation': {'joint': .4}}
    torch.save(state, folder / 'state.pt')
    (folder / 'metrics.json').write_text(json.dumps(metrics))
    ref = {'epoch': 1, 'directory': 'epoch-0001', 'sha256': digest(folder / 'state.pt')}
    for name in ('best.json', 'latest.json'):
        (parent / name).write_text(json.dumps(ref))
    # A running subsequent epoch does not invalidate the finalized best epoch.
    partial = parent / '.epoch-0002'
    partial.mkdir()
    (partial / 'state.pt').write_bytes(b'incomplete')
    return parent, state, metrics, ref


def rewrite_state(parent, state, ref):
    torch.save(state, parent / ref['directory'] / 'state.pt')
    ref['sha256'] = digest(parent / ref['directory'] / 'state.pt')
    for name in ('best.json', 'latest.json'):
        (parent / name).write_text(json.dumps(ref))


def test_snapshot_freezes_completed_epoch_leaves_running_parent_untouched(tmp_path):
    parent, state, metrics, ref = parent_fixture(tmp_path)
    before = {str(p.relative_to(parent)): digest(p) for p in parent.rglob('*') if p.is_file()}
    output = tmp_path / 'snapshot'
    report = freeze_best_completed_epoch(parent, output)
    after = {str(p.relative_to(parent)): digest(p) for p in parent.rglob('*') if p.is_file()}
    assert after == before
    assert report['parent_training_completed'] is False
    assert report['epochs_planned'] == 12
    assert report['completed_at_freeze'] == report['selected_epoch'] == 1
    complete = json.loads((output / 'complete.json').read_text())
    assert complete['status'] == STATUS
    assert complete['identity'] == state['identity']
    assert complete['parent_training_completed'] is False
    assert complete['scientific_acceptance'] is False
    assert digest(checkpoint_path(output, json.loads((output / 'best.json').read_text()))) == ref['sha256']
    assert not (output / '.epoch-0002').exists()
    with pytest.raises(FileExistsError):
        freeze_best_completed_epoch(parent, output)


@pytest.mark.parametrize('damage', ['missing_gradients', 'false_gradient', 'empty_gradients',
                                  'profile_count', 'surface_count', 'nonfinite_loss'])
def test_incomplete_epoch_evidence_is_rejected(tmp_path, damage):
    parent, _, metrics, _ = parent_fixture(tmp_path)
    if damage == 'missing_gradients': metrics.pop('nonzero_gradients')
    elif damage == 'false_gradient': metrics['nonzero_gradients']['head.weight'] = False
    elif damage == 'empty_gradients': metrics['nonzero_gradients'] = {}
    elif damage == 'profile_count': metrics['profile_target_counts'][4] = 0
    elif damage == 'surface_count': metrics['native_target_counts'][5] = 0
    else: metrics['train_loss'] = float('nan')
    (parent / 'epoch-0001' / 'metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(ValueError):
        freeze_best_completed_epoch(parent, tmp_path / 'snapshot')
    assert not (tmp_path / 'snapshot').exists()


def test_checkpoint_hash_mismatch_and_partial_reference_rejected(tmp_path):
    parent, _, _, ref = parent_fixture(tmp_path)
    (parent / 'epoch-0001' / 'state.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        freeze_best_completed_epoch(parent, tmp_path / 'bad-hash')
    ref.update(directory='.epoch-0002', epoch=2)
    for name in ('best.json', 'latest.json'):
        (parent / name).write_text(json.dumps(ref))
    with pytest.raises(ValueError, match='Invalid checkpoint reference'):
        freeze_best_completed_epoch(parent, tmp_path / 'partial')


@pytest.mark.parametrize('damage', ['missing_parameter', 'extra_parameter'])
def test_gradient_evidence_must_cover_exactly_every_learned_parameter(tmp_path, damage):
    parent, state, metrics, ref = parent_fixture(tmp_path)
    state['model'].update({'ingest.weight_ih': torch.ones(6, 2),
                           'ingest.bias_hh': torch.ones(6), 'levels': torch.ones(37, 2),
                           'xyz': torch.ones(12, 3)})
    metrics['nonzero_gradients'].update({'ingest.weight_ih': True,
                                       'ingest.bias_hh': True, 'levels': True})
    rewrite_state(parent, state, ref)
    (parent / 'epoch-0001' / 'metrics.json').write_text(json.dumps(metrics))
    freeze_best_completed_epoch(parent, tmp_path / 'valid-snapshot')
    if damage == 'missing_parameter':
        del metrics['nonzero_gradients']['ingest.weight_ih']
    else:
        metrics['nonzero_gradients']['xyz'] = True
    (parent / 'epoch-0001' / 'metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match='exactly the learned parameters'):
        freeze_best_completed_epoch(parent, tmp_path / 'snapshot')
    assert not (tmp_path / 'snapshot').exists()


def test_state_epoch_and_nonfinite_weights_rejected(tmp_path):
    parent, state, _, ref = parent_fixture(tmp_path)
    state['epoch'] = 2
    rewrite_state(parent, state, ref)
    with pytest.raises(ValueError, match='epoch identity'):
        freeze_best_completed_epoch(parent, tmp_path / 'wrong-epoch')
    state['epoch'] = 1
    state['model']['head.weight'][0, 0] = float('nan')
    rewrite_state(parent, state, ref)
    with pytest.raises(ValueError, match='Nonfinite model'):
        freeze_best_completed_epoch(parent, tmp_path / 'wrong-weights')


def test_completed_parent_is_reported_separately_and_no_child_output(tmp_path):
    parent, state, _, _ = parent_fixture(tmp_path)
    (parent / 'complete.json').write_text(json.dumps({'status': STATUS, 'identity': state['identity']}))
    with pytest.raises(ValueError, match='outside'):
        freeze_best_completed_epoch(parent, parent / 'snapshot')
    report = freeze_best_completed_epoch(parent, tmp_path / 'snapshot')
    assert report['parent_training_completed'] is True
    assert report['parent_completion_sha256'] == digest(parent / 'complete.json')
