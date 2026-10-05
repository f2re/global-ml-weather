"""C2 durability tests; toy inputs are never represented as weather evidence."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import shutil
import sqlite3
import subprocess
import sys

import pytest
import torch

from global_weather.continuous.store import CampaignStore, APPLICATION_ID
from global_weather.continuous.checkpointing import StepTrainer, CheckpointError
from helpers.continuous_step_worker import setup, operation

IDENTITY = {'data_kind': 'synthetic', 'purpose': 'crash_test'}


def load_current(root):
    store = CampaignStore(root)
    current = store.state()['checkpoint']
    return torch.load(store.root/'checkpoints'/current['id']/'state.pt', weights_only=True), store.state()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


def test_commit_is_atomic_with_usage_and_cursor(tmp_path):
    s, m, o, lr, uses = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as trainer:
        assert s.state()['checkpoint']['step_number'] == 0
        r = trainer.step(uses[:2], operation(m, o), cursor={'block': 'one', 'next_index': 2})
        assert r['status'] == 'committed'
        assert s.state()['committed_uses'] == 2
        assert s.state()['checkpoint']['cursor']['next_index'] == 2
        assert len(s.checkpoints()['items']) == 2
        assert s.state()['training_ready'] is False  # automatic acquisition is not implemented
        assert trainer.step(uses[:2], lambda: pytest.fail('duplicate gradient'), cursor={})['status']=='already_committed'
        with pytest.raises(CheckpointError, match='Часть'):
            trainer.step(uses[1:], operation(m, o), cursor={})
        assert s.state()['committed_uses'] == 2


@pytest.mark.parametrize('point', ['after_optimizer', 'after_state_write', 'after_publish', 'before_commit', 'after_commit', 'truncated_write'])
def test_actual_process_crash_and_resume_matches_uninterrupted(tmp_path, point):
    script = Path(__file__).parent/'helpers/continuous_step_worker.py'
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), OMP_NUM_THREADS='1')
    uninterrupted, resumed = tmp_path/'full', tmp_path/'resume'
    full = subprocess.run([sys.executable, str(script), str(uninterrupted), 'none'], env=env, capture_output=True, timeout=30)
    assert full.returncode == 0, full.stderr.decode()
    broken = subprocess.run([sys.executable, str(script), str(resumed), point], env=env, capture_output=True, timeout=30)
    assert broken.returncode == 77, broken.stderr.decode()
    status = CampaignStore(resumed).state()
    assert status['committed_uses'] == (1 if point == 'after_commit' else 0)
    again = subprocess.run([sys.executable, str(script), str(resumed), 'none'], env=env, capture_output=True, timeout=30)
    assert again.returncode == 0, again.stderr.decode()
    a, sa = load_current(uninterrupted)
    b, sb = load_current(resumed)
    equal(a, b)  # model, optimizer, scheduler, Python/NumPy/Torch RNG, modes
    assert sa['committed_uses'] == sb['committed_uses'] == 3
    assert sa['checkpoint']['cursor'] == sb['checkpoint']['cursor']
    assert sb['checkpoint']['step_number'] == 3
    assert len(list((resumed/'continuous/checkpoints').iterdir())) == 2


def test_one_writer_and_lost_fence(tmp_path):
    s, m, o, lr, uses = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        with pytest.raises(CheckpointError, match='Другой'):
            with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr): pass
        with s.transaction(write=True) as db:
            db.execute("UPDATE trainer_owner SET token='new-owner'")
        with pytest.raises(CheckpointError, match='утратил'):
            t.step([uses[0]], operation(m, o), cursor={})
        assert s.state()['committed_uses'] == 0


def test_different_thread_cannot_apply_gradient(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    s, m, o, lr, uses = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t, ThreadPoolExecutor(1) as pool:
        future = pool.submit(t.step, [uses[0]], operation(m, o), cursor={})
        with pytest.raises(CheckpointError): future.result()
        assert s.state()['committed_uses'] == 0


def test_retention_preserves_two_generations_and_explicit_pin(tmp_path):
    s, m, o, lr, uses = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        initial = t.current['id']
        t.pin()
        for use in uses: t.step([use], operation(m, o), cursor={})
        assert len(list(t.directory.iterdir())) == 3
        assert (t.directory/initial).exists()
        assert len(s.checkpoints()['items']) == 4
        assert s.state()['committed_uses'] == 3


@pytest.mark.parametrize('bad', ['nan', 'zero_steps', 'two_steps', 'raised_error'])
def test_failed_step_never_appears_in_ledger_and_requires_reopen(tmp_path, bad):
    s, m, o, lr, uses = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        initial, _ = load_current(tmp_path)
        def run():
            if bad == 'zero_steps': return {}
            result = operation(m, o)()
            if bad == 'nan': next(m.parameters()).data.fill_(float('nan'))
            if bad == 'two_steps': operation(m, o)()
            if bad == 'raised_error': raise RuntimeError('simulated failure')
            return result
        with pytest.raises((ValueError, RuntimeError)): t.step([uses[0]], run, cursor={})
        assert s.state()['committed_uses'] == 0
        with pytest.raises(CheckpointError): t.step([uses[0]], operation(m, o), cursor={})
    _, new_m, new_o, new_lr, _ = setup(tmp_path)
    with StepTrainer(s, new_m, new_o, identity=IDENTITY, scheduler=new_lr):
        equal(new_m.state_dict(), initial['model'])
        equal(new_o.state_dict(), initial['optimizer'])


@pytest.mark.parametrize('file', ['state.pt', 'manifest.json'])
def test_corrupt_current_does_not_roll_back_cursor_or_reinitialize(tmp_path, file):
    s, m, o, lr, uses = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        t.step([uses[0]], operation(m, o), cursor={'next_index': 1})
        current = t.current['id']
    (s.root/'checkpoints'/current/file).write_bytes(b'CORRUPTED')
    _, m, o, lr, _ = setup(tmp_path)
    with pytest.raises(CheckpointError):
        with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr): pass
    assert s.state()['committed_uses'] == 1
    assert s.state()['checkpoint']['id'] == current


def test_consistent_backup_recovers_after_source_workspace_loss(tmp_path):
    origin, backup = tmp_path/'origin', tmp_path/'backup'
    s, m, o, lr, uses = setup(origin)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        t.step([uses[0]], operation(m, o), cursor={'next_index': 1})
        result = t.backup(backup)
        assert not result['source_weather_arrays_included']
    before, _ = load_current(origin)
    shutil.rmtree(origin)
    s, m, o, lr, uses = setup(backup)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        equal(m.state_dict(), before['model'])
        equal(o.state_dict(), before['optimizer'])
        assert s.state()['committed_uses'] == 1
        assert t.step([uses[0]], lambda: pytest.fail('repeated'), cursor={})['status']=='already_committed'
        t.step([uses[1]], operation(m, o), cursor={'next_index': 2})
    assert s.state()['committed_uses'] == 2


def test_changed_identity_or_optimizer_is_rejected_before_weights_change(tmp_path):
    s, m, o, lr, uses = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr): pass
    expected = {k: v.clone() for k, v in m.state_dict().items()}
    with pytest.raises(CheckpointError, match='Изменились'):
        with StepTrainer(s, m, o, identity=dict(IDENTITY, normalization='new'), scheduler=lr): pass
    equal(expected, m.state_dict())


def test_old_v1_schema_migrates_preserving_campaign(tmp_path):
    from global_weather.continuous.store import DDL, IMMUTABLE
    root = tmp_path/'continuous'
    root.mkdir()
    path = root/'learning.sqlite3'
    with sqlite3.connect(path) as db:
        for sql in DDL: db.execute(sql)
        for table in IMMUTABLE:
            for operation in ('UPDATE', 'DELETE'):
                db.execute(f"CREATE TRIGGER immutable_{table}_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'immutable campaign history'); END")
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute('PRAGMA user_version=1')
    s = CampaignStore(tmp_path)
    s.add_range('2000-01-01', '2005-12-31')
    assert s.state()['requested_days'] == 2192
    with s.transaction() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 2


def test_checkpoint_paths_and_writer_lock_reject_symlinks(tmp_path):
    s, m, o, lr, _ = setup(tmp_path)
    elsewhere = tmp_path/'outside'; elsewhere.mkdir()
    (s.root/'checkpoints').symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(CheckpointError):
        with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr): pass
    assert not list(elsewhere.iterdir())


def test_unknown_use_and_control_sample_do_not_reach_callback(tmp_path):
    s, m, o, lr, _ = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        with pytest.raises(CheckpointError):
            t.step(['f'*64], lambda: pytest.fail('unknown sample'), cursor={})
        assert s.state()['committed_uses'] == 0


def test_size_limit_leaves_no_false_training_event(tmp_path):
    s, m, o, lr, _ = setup(tmp_path)
    with pytest.raises(CheckpointError, match='размер'):
        with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr, max_checkpoint_bytes=100): pass
    assert s.state()['checkpoint'] is None


def retimed_fixture(root):
    """Analytic fixture retimed into C1 train months; not an observed archive."""
    from datetime import timedelta
    from global_weather.pipeline.fixture import create_fixture
    from global_weather.pipeline.io import read_json, read_arrays, write_arrays, atomic_json, reference
    from global_weather.continuous.contracts import utc_time
    ds_path = create_fixture(root, mesh_level=2, horizon_hours=72)
    manifest = read_json(ds_path)
    for sample in manifest['samples']:
        issue = utc_time(sample['issue_time']) + timedelta(days=45)
        sample['issue_time'] = issue.isoformat()
        data = read_arrays(root/sample['targets']['path'])
        data['issue_time'] = issue.isoformat()
        target = root/(sample['id']+'-retimed.npz')
        write_arrays(target, **data)
        rows = []
        for text in (root/sample['observations']['path']).read_text().splitlines():
            row = json.loads(text)
            for key in ('observed_at', 'available_at'):
                row[key] = (utc_time(row[key])+timedelta(days=45)).isoformat()
            rows.append(row)
        obs = root/(sample['id']+'-retimed.jsonl')
        obs.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        sample['observations'] = reference(root, obs)
        sample['targets'] = reference(root, target)
    out = root/'retimed.json'
    atomic_json(out, manifest)
    return out


def test_actual_adaptive_weather_step_uses_c2_and_resumes_across_block_calls(tmp_path):
    from global_weather.continuous.prepared import train_prepared_block
    from global_weather.pipeline.runner import TrainConfig
    ds = retimed_fixture(tmp_path/'synthetic')
    cfg = TrainConfig(device='cpu', horizon_hours=72, epochs=1, threads=1)
    sample_ids = ['sample-0', 'sample-1']
    full, resumed = CampaignStore(tmp_path/'full'), CampaignStore(tmp_path/'resumed')
    for s in (full, resumed): s.add_range('2020-02-01', '2020-03-31')
    complete = train_prepared_block(full, ds, config=cfg, sample_ids=sample_ids)
    partial = train_prepared_block(resumed, ds, config=cfg, sample_ids=sample_ids, max_steps=1)
    assert partial['status'] == 'paused_at_saved_step'
    continued = train_prepared_block(resumed, ds, config=cfg, sample_ids=sample_ids)
    assert continued['new_steps'] == continued['already_committed'] == 1
    assert complete['status'] == continued['status'] == 'prepared_block_processed'
    a, _ = load_current(tmp_path/'full')
    b, _ = load_current(tmp_path/'resumed')
    # Sparse topology buffers need their coalesced components compared.
    for key in a['model']:
        if isinstance(a['model'][key], torch.Tensor) and a['model'][key].is_sparse:
            equal(a['model'][key].coalesce().indices(), b['model'][key].coalesce().indices())
            equal(a['model'][key].coalesce().values(), b['model'][key].coalesce().values())
        else:
            equal(a['model'][key], b['model'][key])
    equal(a['optimizer'], b['optimizer'])
    equal(a['rng'], b['rng'])
    assert resumed.state()['committed_uses'] == 2
    assert continued['data_kind'] == 'synthetic'
    assert not continued['automatic_provider_pipeline']
    assert not continued['independent_evaluation_performed']


def test_read_only_checkpoint_api_does_not_expose_a_commit_method(tmp_path):
    from fastapi.testclient import TestClient
    from global_weather.lab.app import create_app
    with TestClient(create_app(tmp_path, testing=True)) as client:
        csrf = {'X-Lab-CSRF': client.get('/api/bootstrap').json()['csrf']}
        assert client.get('/api/learning/checkpoints').json() == {'items': [], 'next_after': None}
        assert client.post('/api/learning/checkpoints', headers=csrf, json={}).status_code == 405


def test_old_head_is_not_accepted_with_new_training_ledger(tmp_path):
    s, m, o, lr, uses = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        initial = t.current['id']
        t.step([uses[0]], operation(m, o), cursor={'next_index': 1})
    with s.transaction(write=True) as db:
        db.execute('UPDATE checkpoint_head SET generation_id=?', (initial,))
    _, m, o, lr, _ = setup(tmp_path)
    with pytest.raises(CheckpointError, match='не согласованы'):
        with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr): pass
    assert s.state()['committed_uses'] == 1


def test_missing_head_is_not_permission_to_initialize_new_weights(tmp_path):
    s, m, o, lr, _ = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr): pass
    with s.transaction(write=True) as db: db.execute('DELETE FROM checkpoint_head')
    _, m, o, lr, _ = setup(tmp_path)
    with pytest.raises(CheckpointError, match='Утрачен'):
        with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr): pass


def test_day_progress_is_derived_from_committed_events_not_registration(tmp_path):
    s, m, o, lr, uses = setup(tmp_path)
    before = next(r for r in s.blocks()['items'] if r['date']=='2000-02-10')
    assert before['training_status'] == 'not_started'
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        t.step([uses[0]], operation(m, o), cursor={'next_index': 1})
        after = next(r for r in s.blocks()['items'] if r['date']=='2000-02-10')
        assert after['training_status'] == 'registered_uses_committed'
        assert after['committed_uses'] == 1
        assert after['catalog_status'] == 'not_checked'
        sample = s.samples()['items'][0]
        assert s.plan_use(sample['id'])['status'] == 'committed'
        assert s.state()['committed_uses'] == 1


def test_backup_rejects_destination_inside_active_workspace(tmp_path):
    s, m, o, lr, _ = setup(tmp_path)
    with StepTrainer(s, m, o, identity=IDENTITY, scheduler=lr) as t:
        with pytest.raises(CheckpointError, match='вне активного'):
            t.backup(tmp_path/'backup')
