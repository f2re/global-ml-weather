"""Workbench tests use synthetic fixtures; they do not certify weather skill."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from global_weather.lab.app import create_app
from global_weather.lab.contracts import RunSpec, safe_child, atomic_json, sha256
from global_weather.lab.metrics import paired_scores, physical_diagnostics
from global_weather.lab.queue import RunQueue, TERMINAL
from global_weather.lab.agents import ROLES, authorize
from global_weather.lab.evaluate import evaluate
from global_weather.connectors.acquire import era5_request, download, StrictRedirect
from global_weather.connectors.local import inspect_file
from global_weather.connectors.noaa import convert_isd, decode_number

@pytest.fixture
def client(tmp_path):
    app=create_app(tmp_path, testing=True)
    with TestClient(app) as c:
        c.headers['X-Lab-CSRF']=c.get('/api/bootstrap').json()['csrf']
        yield c


def wait_run(client, run_id, timeout=50):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        r=client.get('/api/runs/'+run_id).json()
        if r['status'] in TERMINAL: return r
        time.sleep(.1)
    raise AssertionError('Task did not finish.')

@pytest.mark.parametrize('kwargs',[dict(mesh_level=4),dict(hidden=512),dict(horizon_hours=1),dict(horizon_hours=75),dict(seed=-1),dict(kind='shell'),dict(command='rm -rf /'),dict(mesh_level=True),dict(input_file='../data')])
def test_invalid_run_spec(kwargs):
    with pytest.raises(ValueError): RunSpec(**kwargs)

@pytest.mark.parametrize('name',['../x','/etc/passwd','a/b','.','..','x\\y'])
def test_path_escape_rejected(tmp_path,name):
    with pytest.raises(ValueError):safe_child(tmp_path,name)


def test_symlink_rejected(tmp_path):
    (tmp_path/'link').symlink_to('/tmp')
    with pytest.raises(ValueError):safe_child(tmp_path,'link')


def test_csrf_and_origin(client):
    assert client.post('/api/runs',json={},headers={'X-Lab-CSRF':''}).status_code==403
    assert client.post('/api/runs',json={},headers={'Origin':'https://evil.example'}).status_code==403
    assert client.get('/api/bootstrap',headers={'Host':'evil.example'}).status_code==400


def test_offline_static_resources(client):
    page=client.get('/')
    assert page.status_code==200 and 'Синтетическая' in page.text
    assert "frame-ancestors 'none'" in page.headers['content-security-policy']
    assert client.get('/static/app.js').status_code==200
    assert client.get('/static/style.css').status_code==200
    assert 'https://' not in page.text


def test_no_arbitrary_command_or_role(client):
    assert client.post('/api/runs',json={'kind':'shell'}).status_code==422
    assert client.post('/api/runs?role=physics',json={'kind':'baseline'}).status_code==400


def test_baseline_complete_pipeline(client):
    r=client.post('/api/runs',json={'kind':'baseline','mesh_level':0,'horizon_hours':6}).json()
    final=wait_run(client,r['id'])
    assert final['status']=='completed',client.get(f"/api/runs/{r['id']}/log").text
    report=client.get(f"/api/runs/{r['id']}/report").json()
    assert report['status']=='synthetic' and report['skill']['rmse'] is None
    assert report['lead_hours']==[0,3,6]
    frame=client.get(f"/api/runs/{r['id']}/frame?lead=3&cell=0&variable=temperature&level=1").json()
    assert len(frame['values'])==12 and len(frame['profile'])==37
    assert client.get(f"/api/runs/{r['id']}/frame?cell=-1").status_code==400
    assert client.get(f"/api/runs/{r['id']}/frame?variable=wrong").status_code==400
    assert client.get(f"/api/runs/{r['id']}/frame?lead=72").status_code==400
    assert client.get(f"/api/runs/{r['id']}/download/request.json").status_code==404
    assert client.get(f"/api/runs/{r['id']}/download/report.json").status_code==200
    assert client.get(f"/api/runs/{r['id']}/grid").json()['offsets'][-1]==60


@pytest.mark.skipif(importlib.util.find_spec('global_weather.adaptive') is None, reason='Adaptive module unavailable in this local source snapshot; CI imports it explicitly.')
def test_adaptive_complete_pipeline(client):
    r=client.post('/api/runs',json={'kind':'adaptive','mesh_level':0,'horizon_hours':72,'optimizer_steps':1}).json()
    final=wait_run(client,r['id'])
    assert final['status']=='completed',client.get(f"/api/runs/{r['id']}/log").text
    report=client.get(f"/api/runs/{r['id']}/report").json()
    assert len(report['lead_hours'])==25 and len(report['optimization'])==1


def test_upload_inspect_and_no_overwrite(client):
    data={'observation_id':'test','source':'station','variable':'t2m','value':280.,'units':'K','latitude':50.,'longitude':30.,'observed_at':'2020-01-01T00:00:00Z','available_at':'2020-01-01T00:05:00Z','valid':True}
    assert client.post('/api/inbox/observations.jsonl',content=json.dumps(data)+'\n').status_code==200
    assert client.post('/api/inbox/observations.jsonl',content='{}').status_code==409
    assert client.post('/api/inbox/program.py',content='print(1)').status_code==400
    assert len(client.get('/api/inbox').json())==1
    r=client.post('/api/runs',json={'kind':'inspect','input_file':'observations.jsonl'}).json()
    assert wait_run(client,r['id'])['status']=='completed'
    report=client.get(f"/api/runs/{r['id']}/report").json()
    assert report['accepted_contract']==1 and report['physics_verified'] is False


def test_file_being_uploaded_is_not_deleted(client,tmp_path):
    # Use actual workspace from the client app, not fixture tmp_path guesswork.
    partial=client.app.state.queue.inbox/'data.csv.part';partial.write_text('in-progress')
    assert client.post('/api/inbox/data.csv',content='other').status_code==409
    assert partial.read_text()=='in-progress'


def test_inspect_requires_existing_file(client):
    assert client.post('/api/runs',json={'kind':'inspect','input_file':'missing.jsonl'}).status_code==400
    assert client.get('/api/runs/missing').status_code==404


def test_cancel_queued_run(tmp_path):
    q=RunQueue(tmp_path)
    r=q.create(RunSpec(kind='baseline'))
    assert q.cancel(r['id'])['status']=='cancelled'
    assert q.cancel(r['id'])['status']=='cancelled'


def test_queue_cap(tmp_path):
    q=RunQueue(tmp_path)
    for _ in range(6):q.create(RunSpec(kind='baseline'))
    with pytest.raises(ValueError):q.create(RunSpec(kind='baseline'))


def test_interrupted_restart_and_single_instance(tmp_path):
    q=RunQueue(tmp_path);r=q.create(RunSpec())
    q.start()
    try:
        assert q.get(r['id'])['status']=='interrupted'
        other=RunQueue(tmp_path)
        with pytest.raises(RuntimeError):other.start()
    finally:q.close()


def test_running_cancellation(client):
    r=client.post('/api/runs',json={'kind':'baseline','mesh_level':3,'horizon_hours':72,'optimizer_steps':5}).json()
    client.post('/api/runs/'+r['id']+'/cancel')
    assert wait_run(client,r['id'])['status']=='cancelled'


def test_timeout_record(tmp_path):
    q=RunQueue(tmp_path,timeout=.001);q.start()
    try:
        r=q.create(RunSpec(kind='baseline'))
        end=time.monotonic()+20
        while q.get(r['id'])['status'] not in TERMINAL and time.monotonic()<end:time.sleep(.1)
        assert q.get(r['id'])['status']=='timed_out'
    finally:q.close()


def test_metrics_with_area_and_baseline():
    s=paired_scores([2.,0.],[0.,0.],np.array([True,True]),[3.,1.],baseline=[4.,0.])
    assert s['rmse']==pytest.approx(np.sqrt(3)) and s['mae']==pytest.approx(1.5)
    assert s['skill_rmse']==pytest.approx(.5)


def test_empty_targets_are_not_perfect_forecast():
    s=paired_scores([np.nan],[np.nan],np.array([False]),[1.])
    assert s['rmse'] is None and s['count']==0
    with pytest.raises(ValueError):paired_scores([np.nan],[0.],np.array([True]),[1.])


def test_hydrostatic_diagnostic_and_negative_values():
    pressure=np.array([100000.,70000.,50000.]);p=np.zeros((2,3,6));p[...,0]=280;p[...,4]=287.05*280*np.log(100000/pressure)
    s=np.zeros((2,8));s[:,:2]=280;s[:,4:6]=101000;s[:,7]=.5
    d=physical_diagnostics(p,s,pressure,np.array([1.,2.]))
    assert d['hydrostatic_rmse_m2_s2']<1e-8 and d['meteorological_skill']=='not_measured'
    p[0,0,1]=-.1;s[0,6]=-1
    d=physical_diagnostics(p,s,pressure,np.array([1.,2.]))
    assert d['negative_humidity']==1 and d['negative_precipitation']==1


def test_paired_evaluation_manifest(tmp_path):
    path=tmp_path/'pairs.npz'
    np.savez(path,prediction=np.ones((2,3)),target=np.zeros((2,3)),mask=np.ones((2,3),bool),lead_hours=np.array([24,72]),area_m2=np.ones(3))
    manifest=dict(variable='t2m',units='K',target_source='SYNTHETIC',forecast_source='SYNTHETIC',data_kind='synthetic',paired_sha256=sha256(path))
    r=evaluate(path,manifest)
    assert r['overall']['rmse']==1 and r['independent_test_certified'] is False
    manifest['paired_sha256']='0'*64
    with pytest.raises(ValueError):evaluate(path,manifest)


def test_era5_request_full_profiles():
    plan=era5_request('2020-02-29')
    assert len(plan['request']['pressure_level'])==37 and len(plan['request']['time'])==24
    assert 'surface_pressure' in era5_request('2020-01-01',pressure_levels=False)['request']['variable']
    with pytest.raises(ValueError):era5_request('2020-02-30')

@pytest.mark.parametrize('url',['http://www.ncei.noaa.gov/data','https://evil.example/data','https://www.ncei.noaa.gov@evil.example/a','https://www.ncei.noaa.gov/a?key=secret'])
def test_download_allowlist_without_network(tmp_path,url):
    with pytest.raises(ValueError):download(url,tmp_path/'file')


def test_noaa_temperature_wind_and_time(tmp_path):
    path=tmp_path/'isd.csv'
    path.write_text('STATION,DATE,LATITUDE,LONGITUDE,ELEVATION,TMP,DEW,SLP,WND\n26063099999,2020-01-01T00:00:00,60,30,6,"+0100,1","+0050,1","10123,1","090,1,N,0050,1"\n')
    out=tmp_path/'obs.jsonl';report=convert_isd(path,out,acquired_at='2026-10-03T12:00:00Z')
    rows={x['variable']:x for x in map(json.loads,out.read_text().splitlines())}
    assert rows['t2m']['value']==pytest.approx(283.15)
    assert rows['mslp']['value']==101230 and rows['u10']['value']==pytest.approx(-5.)
    assert abs(rows['v10']['value'])<1e-10 and rows['t2m']['available_at'].startswith('2026')
    assert report['historical_availability_known'] is False
    assert decode_number('+9999,1','9999',10) is None
    assert decode_number('+0100,3','9999',10) is None


def test_unknown_satdump_is_quarantined(tmp_path):
    p=tmp_path/'product.cbor';p.write_bytes(b'fixture')
    assert inspect_file(p)['status']=='quarantine'


def test_role_and_protocol_files_match_registry():
    root=Path(__file__).resolve().parents[1]
    for role in ROLES:
        assert (root/role['instruction']).is_file()
        assert (root/'.claude'/'agents'/f"{role['id']}.md").is_file()
    with pytest.raises(ValueError):authorize('physics','baseline')
    assert authorize('executor','adaptive')['id']=='executor'


def test_data_agent_plan_executes_without_network(tmp_path):
    from global_weather.lab.agents import main
    main(['--role','data-steward','--action','plan-era5','--execute','--date','2020-01-01','--workspace',str(tmp_path)])
    plans=list((tmp_path/'acquisition').glob('*/request.json'))
    assert len(plans)==1
    assert json.loads(plans[0].read_text())['dataset']=='reanalysis-era5-pressure-levels'
    with pytest.raises(ValueError):
        main(['--role','data-steward','--action','download-graphcast','--execute','--workspace',str(tmp_path)])
