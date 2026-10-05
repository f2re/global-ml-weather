"""C1 persistence and admission boundaries, not real continuous weather training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from global_weather.continuous.store import CampaignStore
from global_weather.continuous.contracts import (default_contract, fingerprint, month_role,
                                                 temporal_role, utc_time, sample_identity)

NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)


def store(root):
    return CampaignStore(root, clock=lambda: NOW)


def descriptor(issue='2000-01-15T12:00:00Z'):
    return {'issue_time': issue, 'inputs': [dict(provider='ghcnh', object_id='station/year', revision='1', sha256='a' * 64)],
            'targets': [dict(provider='era5', object_id='analysis/time', revision='1', sha256='b' * 64)],
            'modalities': ['station'], 'transform_sha256': 'c' * 64}


def issue_for(role):
    for month in range(1, 13):
        when = datetime(2000, month, 15, 12, tzinfo=timezone.utc)
        if temporal_role(when)[1] == role:
            return when.isoformat()
    raise AssertionError('Test-year selection must contain each required role')


def test_multiyear_range_persists_without_materializing_days(tmp_path):
    s = store(tmp_path)
    assert s.state()['campaign'] is None
    first = s.add_range('2000-01-01', '2005-12-31')
    assert first['requested_days'] == first['new_days_at_registration'] == 2192
    second = store(tmp_path).add_range('2004-01-01', '2010-12-31')
    assert second['campaign_id'] == first['campaign_id']
    assert second['existing_days_at_registration'] == 731
    assert second['new_days_at_registration'] == 1826
    assert s.state()['requested_days'] == 4018
    with s.transaction() as db:
        assert db.execute('SELECT count(*) FROM intervals').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM samples').fetchone()[0] == 0
    assert first['training_started'] is False
    assert s.state()['committed_uses'] == 0


def test_large_range_is_interval_metadata_not_a_million_rows(tmp_path):
    s = store(tmp_path)
    s.add_range('0002-01-01', '2020-12-31')
    result = s.blocks(limit=3)
    assert len(result['items']) == 3
    assert s.path.stat().st_size < 512 * 1024


def test_retries_and_key_reuse(tmp_path):
    s = store(tmp_path)
    first = s.add_range('2000-01-01', '2005-12-31', request_key='click-1')
    event_cursor = s.state()['last_event']
    repeat = s.add_range('2000-01-01', '2005-12-31', request_key='click-1')
    assert first['id'] == repeat['id'] and repeat['replayed']
    assert s.state()['last_event'] == event_cursor
    with pytest.raises(ValueError, match='другими датами'):
        s.add_range('2001-01-01', '2005-12-31', request_key='click-1')
    assert s.state()['requests'] == 1
    default = s.add_range('2001-01-01', '2002-12-31')
    assert s.add_range('2001-01-01', '2002-12-31')['id'] == default['id']
    assert default['new_days_at_registration'] == 0


@pytest.mark.parametrize('start,end', [('2000-1-01','2001-01-01'), ('2000-02-30','2000-03-01'),
    ('2005-01-01','2000-01-01'), ('2026-10-06','2026-10-06'), ('0001-01-01','2000-01-01'),
    ('2000-01-01T00:00Z','2001-01-01'), (20000101,'2001-01-01')])
def test_invalid_range_is_not_partially_registered(tmp_path, start, end):
    s = store(tmp_path)
    with pytest.raises(ValueError):
        s.add_range(start, end)
    assert s.state()['requests'] == 0
    assert s.state()['campaign'] is None


def test_transaction_rollback_on_event_failure(tmp_path, monkeypatch):
    s = store(tmp_path)
    s.add_range('2000-01-01','2000-01-31')
    before = s.state()
    def fail(*args):
        raise RuntimeError('injected interruption')
    monkeypatch.setattr(s, '_event', fail)
    with pytest.raises(RuntimeError):
        s.add_range('1999-01-01','2001-01-01')
    assert s.state() == before


def test_concurrent_retry_does_not_duplicate_campaign_or_request(tmp_path):
    def add(_):
        return store(tmp_path).add_range('2000-01-01','2005-12-31')
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(add, range(8)))
    assert len({r['id'] for r in rows}) == 1
    assert sum(not r['replayed'] for r in rows) == 1
    assert store(tmp_path).state()['requests'] == 1


def test_day_pagination_crosses_gaps_and_includes_last_day(tmp_path):
    s = store(tmp_path)
    s.add_range('2000-02-28','2000-03-01')
    s.add_range('2020-12-31','2020-12-31')
    first = s.blocks(limit=2)
    assert [r['date'] for r in first['items']] == ['2000-02-28','2000-02-29']
    second = s.blocks(limit=2, after_date=first['next_after_date'])
    assert [r['date'] for r in second['items']] == ['2000-03-01','2020-12-31']
    assert second['next_after_date'] is None
    assert all(x['catalog_status']=='not_checked' for x in first['items'])


def test_interval_union_bridging_adjacent_and_contained_ranges(tmp_path):
    s = store(tmp_path)
    s.add_range('2000-01-01','2000-01-03')
    s.add_range('2000-01-05','2000-01-08')
    bridge = s.add_range('2000-01-03','2000-01-05')
    assert bridge['new_days_at_registration']==1
    assert s.add_range('2000-01-02','2000-01-07')['new_days_at_registration']==0
    assert s.state()['requested_days']==8


def test_month_roles_and_full_dependency_windows_are_disjoint():
    beginning = datetime(1999,12,1,tzinfo=timezone.utc)
    intervals = {'train':[],'validation':[],'test':[]}
    guards = 0
    for offset in range(400*4):
        issue = beginning + timedelta(hours=6*offset)
        nominal, role = temporal_role(issue)
        if role=='guard':
            guards += 1
        else:
            intervals[role].append((issue-timedelta(hours=12),issue+timedelta(hours=72)))
            assert role==nominal
    assert guards>0
    # All pairs close enough to overlap must be of the same role.
    rows = sorted((lo,hi,role) for role,group in intervals.items() for lo,hi in group)
    for index,(lo,hi,role) in enumerate(rows):
        for other in rows[index+1:index+20]:
            if other[0] > hi:
                break
            assert other[2]==role


def test_sample_identity_order_time_zone_and_revision(tmp_path):
    s = store(tmp_path)
    s.add_range('2000-01-01','2000-12-31')
    d = descriptor(issue_for('train'))
    d['inputs'].append(dict(provider='ghcnh',object_id='other/year',revision='1',sha256='d'*64))
    original = s.register_sample(d)
    alternate = json.loads(json.dumps(d))
    alternate['inputs'].reverse()
    when = utc_time(alternate['issue_time']).astimezone(timezone(timedelta(hours=3)))
    alternate['issue_time']=when.isoformat()
    assert s.register_sample(alternate)['id']==original['id']
    alternate['inputs'][0]['revision']='2'
    changed=s.register_sample(alternate)
    assert changed['id']!=original['id']
    assert s.state()['samples']==2
    assert changed['physically_admitted'] is False


def test_roles_do_not_change_when_adding_past_and_future_ranges(tmp_path):
    s=store(tmp_path)
    s.add_range('2000-01-01','2000-12-31')
    old=[s.register_sample(descriptor(issue_for(role))) for role in ('train','validation','test')]
    contract=s.state()['campaign']['contract_hash']
    s.add_range('1990-01-01','2010-12-31')
    new=[s.register_sample(descriptor(issue_for(role))) for role in ('train','validation','test')]
    assert [(r['id'],r['role']) for r in old]==[(r['id'],r['role']) for r in new]
    assert s.state()['campaign']['contract_hash']==contract
    assert s.state()['historical_independence_verified'] is False


def test_intended_passes_are_separate_from_committed_training(tmp_path):
    s=store(tmp_path)
    s.add_range('2000-01-01','2000-12-31')
    sample=s.register_sample(descriptor(issue_for('train')))
    first=s.plan_use(sample['id'])
    assert s.plan_use(sample['id'])['replayed'] is True
    second=s.plan_use(sample['id'],pass_number=1)
    assert first['id']!=second['id']
    assert s.state()['planned_uses']==2 and s.state()['committed_uses']==0
    assert s.samples()['items'][0]['committed_uses']==0
    assert not hasattr(s,'mark_trained')


@pytest.mark.parametrize('role',['validation','test'])
def test_holdouts_cannot_enter_usage_plan(tmp_path, role):
    s=store(tmp_path)
    s.add_range('2000-01-01','2000-12-31')
    sample=s.register_sample(descriptor(issue_for(role)))
    with pytest.raises(ValueError,match='Контрольные'):
        s.plan_use(sample['id'])
    with s.transaction(write=True) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO usage_plan(id,sample_id,stage,pass_number,created_at) VALUES (?,?,'base',0,'test')",('x',sample['id']))
    assert s.state()['planned_uses']==0


def test_immutable_history_and_fk_constraints(tmp_path):
    s=store(tmp_path)
    s.add_range('2000-01-01','2000-12-31')
    sample=s.register_sample(descriptor(issue_for('train')))
    with s.transaction(write=True) as db:
        assert db.execute('PRAGMA foreign_keys').fetchone()[0]==1
        assert db.execute('PRAGMA synchronous').fetchone()[0]==2
        for statement in ("DELETE FROM samples", "UPDATE assignments SET role='test'", "DELETE FROM campaign"):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(statement)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO training_events(use_id,generation_id) VALUES ('unknown','unpublished')")
    assert s.samples()['items'][0]['id']==sample['id']


def test_sample_outside_range_duplicate_sources_and_unknown_fields(tmp_path):
    s=store(tmp_path)
    s.add_range('2000-01-01','2000-12-31')
    with pytest.raises(ValueError):
        s.register_sample(descriptor('2001-01-15T12:00:00Z'))
    d=descriptor();d['inputs']*=2
    with pytest.raises(ValueError):
        s.register_sample(d)
    d=descriptor();d['command']='echo bad'
    with pytest.raises(ValueError):
        s.register_sample(d)
    assert s.state()['samples']==0


@pytest.mark.parametrize('name',['learning.sqlite3','learning.sqlite3-wal','learning.sqlite3-shm'])
def test_database_and_sidecar_symlinks_rejected(tmp_path,name):
    root=tmp_path/'continuous';root.mkdir()
    target=tmp_path/'outside';target.write_text('keep')
    (root/name).symlink_to(target)
    with pytest.raises(ValueError,match='ссылка'):
        store(tmp_path)
    assert target.read_text()=='keep'


def test_unknown_database_schema_is_not_reset(tmp_path):
    s=store(tmp_path)
    with sqlite3.connect(s.path) as db:
        db.execute('PRAGMA user_version=99')
    with pytest.raises(ValueError,match='версия'):
        store(tmp_path)
    with sqlite3.connect(s.path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0]==99


def test_event_and_request_pagination(tmp_path):
    s=store(tmp_path)
    for year in range(2000,2004):
        s.add_range(f'{year}-01-01',f'{year}-12-31')
    page=s.ranges(limit=2)
    assert page['next_after']==2
    assert len(s.ranges(after=2,limit=2)['items'])==2
    page=s.events(limit=2)
    assert page['next_after']==2
    assert len(s.events(after=2)['items'])==3
    assert s.events(after=5)['items']==[]
    with pytest.raises(ValueError):
        s.blocks(limit=201)


def test_date_only_api_keeps_security_and_reports_pending_executor(tmp_path):
    from global_weather.lab.app import create_app
    app=create_app(tmp_path,testing=True)
    with TestClient(app, raise_server_exceptions=True) as client:
        csrf=client.get('/api/bootstrap').json()['csrf']
        data={'start_date':'2000-01-01','end_date':'2005-12-31'}
        assert client.post('/api/learning/ranges',json=data).status_code==403
        headers={'x-lab-csrf':csrf,'idempotency-key':'clicked-1'}
        response=client.post('/api/learning/ranges',json=data,headers=headers)
        assert response.status_code==200
        assert response.json()['training_started'] is False
        assert 'unsafe-eval' not in response.headers['content-security-policy']
        assert client.post('/api/learning/ranges',json=data,headers=headers).json()['replayed']
        assert client.post('/api/learning/ranges',json=dict(data,stations=['ANY']),headers=headers).status_code==422
        assert client.post('/api/learning/ranges',json=data,headers=dict(headers,origin='https://bad.invalid')).status_code==403
        assert client.get('/api/learning/blocks?limit=3').status_code==200
        assert client.get('/api/learning/blocks?limit=2000').status_code==422
        assert client.get('/api/learning/state').json()['committed_uses']==0
        assert client.post('/api/learning/mark-trained',json={},headers=headers).status_code==404
    second=TestClient(create_app(tmp_path,testing=True))
    assert second.get('/api/learning/state').json()['requested_days']==2192


def test_real_process_interruption_rolls_back_metadata_transaction(tmp_path):
    s=store(tmp_path)
    s.add_range('2000-01-01','2000-01-31')
    script='''
import os,sys
from global_weather.continuous import CampaignStore
s=CampaignStore(sys.argv[1])
with s.transaction(write=True) as db:
    db.execute('DELETE FROM intervals')
    os._exit(37)
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,timeout=30)
    assert result.returncode==37,result.stderr
    assert store(tmp_path).state()['requested_days']==31


def test_cli_persists_across_processes_and_does_not_train(tmp_path):
    command=[sys.executable,'-m','global_weather.continuous','--workspace',str(tmp_path)]
    result=subprocess.run(command+['add-range','--start-date','2000-01-01','--end-date','2005-12-31'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['requested_days']==2192
    result=subprocess.run(command+['state'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['model_initialized'] is False
    assert not list(tmp_path.rglob('*.pt'))


def test_each_year_reserves_all_roles_without_fixed_season():
    seen = set()
    for year in range(1998, 2027):
        roles = [month_role(datetime(year,m,15,tzinfo=timezone.utc)) for m in range(1,13)]
        assert [roles.count(r) for r in ('train','validation','test')] == [8,2,2]
        seen.add(tuple(roles))
    assert len(seen)>1


def test_guard_sample_cannot_be_used_for_training(tmp_path):
    s = store(tmp_path)
    s.add_range('2000-01-01','2000-12-31')
    start = datetime(2000,1,1,tzinfo=timezone.utc)
    issue = next(start+timedelta(hours=6*i) for i in range(366*4)
                 if temporal_role(start+timedelta(hours=6*i))[1]=='guard')
    sample = s.register_sample(descriptor(issue.isoformat()))
    with pytest.raises(ValueError,match='Контрольные'):
        s.plan_use(sample['id'])


def test_source_credentials_urls_and_nonfinite_fields_are_not_accepted(tmp_path):
    s = store(tmp_path)
    s.add_range('2000-01-01','2000-12-31')
    d = descriptor(); d['inputs'][0]['object_id']='https://archive.invalid/file'
    with pytest.raises(ValueError):
        s.register_sample(d)
    d = descriptor(); d['inputs'][0]['sha256']=float('nan')
    with pytest.raises(ValueError):
        s.register_sample(d)
    assert s.state()['samples']==0
