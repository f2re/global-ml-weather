"""Contract tests plus a genuine, provenance-bearing NOAA excerpt.

Synthetic ERA5 fields below test I/O only and never enter a real training run.
"""
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import os
import numpy as np
import pytest
from fastapi.testclient import TestClient
from global_weather.devices import select_device
from global_weather.autonomous.plan import Experiment,parse_plan
from global_weather.autonomous.execute import Stages
from global_weather.autonomous.service import Credentials,Experiments
from global_weather.pipeline.io import atomic_json
from global_weather.providers.ghcnh import read_records,qc_good,station_url,station_index
from global_weather.providers.era5 import requests_for


def plan(**kw):return Experiment(archive_assumption_accepted=True,**kw)


@pytest.mark.parametrize('device',['bogus','cuda:-1','/cpu','cuda:abc',None])
def test_bad_device(device):
    with pytest.raises(ValueError):select_device(device)


def test_auto_device_cpu_and_explicit_cuda_failure(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert str(select_device('auto'))=='cpu'
    with pytest.raises(ValueError):select_device('cuda')


def test_cuda_first_policy_and_index(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    assert str(select_device('auto'))=='cuda:0'
    assert str(select_device('cuda:1'))=='cuda:1'
    with pytest.raises(ValueError):select_device('cuda:2')


def test_plan_separation_and_requests():
    from global_weather.pipeline.dataset import utc
    from datetime import timedelta
    p=plan().checked();rows=p['samples']
    for left,right in [('train','validation'),('validation','test')]:
        assert max(utc(s['issue_time']) for s in rows if s['split']==left)+timedelta(hours=6)<min(utc(s['issue_time']) for s in rows if s['split']==right)-timedelta(hours=12)
    queries=requests_for(p)
    assert queries[-1]['request']['variable']==['geopotential','land_sea_mask']
    assert len(queries[0]['request']['pressure_level'])==37
    assert queries[0]['request']['time']==['12:00','15:00','18:00']
    assert queries[2]['request']['time']==['13:00','14:00','15:00','16:00','17:00','18:00']
    assert 'sea_surface_temperature' not in str(queries)
    assert p['estimated_era5_bytes']<2*1024**3


@pytest.mark.parametrize('options',[{'horizon_hours':7},{'step_hours':2},{'station_count':0},{'source_grid_degrees':3},
    {'train_fraction':.9,'validation_fraction':.2},{'max_download_gib':.001},{'latency_minutes':-1},
    {'end':'2020-01-03T00:00:00Z'},{'stations':['BAD']},{'network':'yes'},{'mesh_level':9}])
def test_invalid_plans(options):
    with pytest.raises((ValueError,TypeError)):plan(**options).checked()


def test_no_implicit_archive_assumption():
    with pytest.raises(ValueError):Experiment().checked()
    with pytest.raises(ValueError):parse_plan({'command':'rm -rf'})


@pytest.mark.parametrize('code,source,ok',[('1','223',True),('4','223',False),('5','223',False),('5','313',True),
    ('r','313',False),('1;r','313',False),('','223',True),('z','223',False),('A','313',False),('1','unknown',False)])
def test_source_aware_qc(code,source,ok):assert qc_good(code,source)==ok


def test_actual_ghcnh_excerpt():
    path=Path(__file__).parent/'data/ghcnh_2020_excerpt.psv';receipt=json.loads(path.with_suffix('.source.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['excerpt_sha256']
    rows,report=read_records(path,acquired_at=receipt['acquired_at'],latency_minutes=60)
    assert rows and report['historical_availability_known'] is False
    assert all(r['provider']=='NOAA_GHCNh' and r['actual_available_at'] is None for r in rows)
    values={r['variable']:r for r in rows if r['observed_at']=='2020-01-01T00:00:00+00:00'}
    assert values['t2m']['value']==pytest.approx(271.45)
    assert values['mslp']['value']==101360
    assert values['u10']['value']>0 and values['v10']['value']<0
    assert values['t2m']['available_at']=='2020-01-01T01:00:00+00:00'


def test_station_parser_and_path(tmp_path):
    p=tmp_path/'stations';p.write_text('ACM00078861  17.1167  -61.7833   10.0    COOLIDGE FIELD   ANTIGUA (AUX.         78861 AG\n')
    assert station_index(p)[0]['latitude']==17.1167
    assert station_url('USW00014933',2020).endswith('GHCNh_USW00014933_2020.psv')
    with pytest.raises(ValueError):station_url('../../etc/p',2020)


def test_stages_resume_detects_corruption(tmp_path):
    calls=[];s=Stages(tmp_path,'identity')
    def create(d):calls.append(1);(d/'value').write_text('checked');return {'done':True}
    s.run('example',create);Stages(tmp_path,'identity').run('example',create);assert len(calls)==1
    (tmp_path/'example/value').write_text('changed')
    with pytest.raises(ValueError):Stages(tmp_path,'identity').run('example',create)
    with pytest.raises(ValueError):Stages(tmp_path,'another')


def test_strict_coverage_refuses_empty_late_target(tmp_path):
    from global_weather.pipeline.fixture import create_fixture
    from global_weather.pipeline.dataset import PreparedDataset
    from global_weather.pipeline.coverage import check_coverage
    ds=PreparedDataset(create_fixture(tmp_path/'synthetic',horizon_hours=6))
    d=ds.targets(ds.samples[0]);check_coverage(d)
    d['profile_mask'][-1]=False;d['surface_mask'][-1]=False
    with pytest.raises(ValueError,match='Неполные'):check_coverage(d)


def test_under_ground_excluded_but_missing_ps_not_permission(tmp_path):
    from global_weather.pipeline.fixture import create_fixture
    from global_weather.pipeline.dataset import PreparedDataset
    from global_weather.pipeline.coverage import check_coverage
    ds=PreparedDataset(create_fixture(tmp_path/'synthetic'))
    d=ds.targets(ds.samples[0]);d['surface'][:,:,4]=95000.;d['profile_mask'][:,:,:2,:]=False
    check_coverage(d)
    d['surface_mask'][:,:,4]=False
    with pytest.raises(ValueError):check_coverage(d)


def test_secrets_permissions_and_redacted_status(tmp_path):
    c=Credentials(tmp_path/'credentials.json');c.save({'cds_key':'example-secret'})
    assert c.path.stat().st_mode&0o777==0o600
    assert c.status()=={'cds_key':True}
    with pytest.raises(ValueError):c.save({'url':'https://attacker.invalid'})
    with pytest.raises(ValueError):c.save({'cds_key':'contains spaces'})
    c.path.chmod(0o644)
    with pytest.raises(ValueError):c.read()


def test_ui_csrf_credentials_and_plan(tmp_path):
    from global_weather.lab.app import create_app
    app=create_app(tmp_path,testing=True)
    # No lifespan: queued test requests do not launch acquisition.
    client=TestClient(app);headers={'X-Lab-CSRF':client.get('/api/bootstrap').json()['csrf']}
    assert client.post('/api/autonomous/credentials',json={'cds_key':'secret'}).status_code==403
    assert client.post('/api/autonomous/credentials',json={'cds_key':'secret'},headers=headers).json()=={'cds_key':True}
    state=client.get('/api/autonomous/state').json();assert state['credentials']=={'cds_key':True}
    assert client.get('/experiments').status_code==200
    body=asdict(plan());assert client.post('/api/autonomous/plan',json=body,headers=headers).status_code==200
    row=client.post('/api/autonomous/runs',json=body,headers=headers).json();assert row['status']=='queued'
    assert 'secret' not in json.dumps(row)
    assert client.post('/api/autonomous/runs/'+row['id']+'/cancel',json={},headers=headers).json()['status']=='cancelled'
    assert client.post('/api/autonomous/runs/'+row['id']+'/resume',json={},headers=headers).json()['status']=='queued'
    assert client.post('/api/satellites/search',json={'start':'2020-01-01T00:00:00Z','end':'2020-01-02T00:00:00Z','platform':'ARCM1'},headers=headers).status_code==400


def test_pinned_gptl_transport_is_unmodified():
    import global_weather.providers._gptl as m
    root=Path(m.__file__).parent;meta=json.loads((root/'UPSTREAM.json').read_text())
    for name,digest in meta['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest


def test_satellite_catalog_preserves_electro_identity_and_hides_signed_url(tmp_path,monkeypatch):
    from global_weather.providers.satellites import SatelliteCatalog
    # Transport contract, not a real Electro-L measurement.
    class Client:
        def api_json(self,*a,**k):return {'features':[{'id':'test','properties':{'platform':'TEST-ELECTRO','datetime':'2020-01-01T01:00:00Z','processing:level':'L2IR'},'assets':{'ch9':{'href':'https://s3.gptl.ru/b/test_ch9.tif?signature=private','raster:bands':[{'unit':'K','scale':1,'offset':0}]}}}]}
        def validate(self,*a,**k):pass
    c=SatelliteCatalog(tmp_path);monkeypatch.setattr(c,'client',lambda:Client())
    r=c.search(start='2020-01-01T00:00:00Z',end='2020-01-02T00:00:00Z',platform='TEST-ELECTRO',network=True)
    assert r['items'][0]['platform']=='TEST-ELECTRO'
    assert 'private' not in json.dumps(r) and 'uri' not in r['items'][0]
    assert r['items'][0]['model_ready'] is False
    with pytest.raises(ValueError):c.download('not-in-search',network=True)
    with pytest.raises(ValueError):c.search(start='2020-01-01T00:00:00Z',end='2020-01-02T00:00:00Z',platform='TEST-ELECTRO',network='yes')


def test_local_catalog_does_not_follow_symlinks(tmp_path):
    from global_weather.providers.satellites import SatelliteCatalog
    root=tmp_path/'source';root.mkdir();other=tmp_path/'outside';other.mkdir();(root/'alias').symlink_to(other,target_is_directory=True)
    (other/'product.json').write_text(json.dumps({'time':'2020-01-01T00:00:00Z','platform':'ARCM1','request':{'channel':9}}))
    c=SatelliteCatalog(tmp_path/'cache',root);assert c.browse_local()['items']==[]


def test_worker_queue_does_not_accept_a_command(tmp_path):
    service=Experiments(tmp_path)
    with pytest.raises(ValueError):service.create({'command':'echo injected'})



def test_published_stage_recovers_after_journal_interruption(tmp_path):
    s=Stages(tmp_path,'identity')
    s.run('example',lambda d: ((d/'value').write_text('checked') and {'done':True}))
    journal=json.loads((tmp_path/'stages.json').read_text());journal['stages']['example']={'status':'running'}
    atomic_json(tmp_path/'stages.json',journal)
    directory,result=Stages(tmp_path,'identity').run('example',lambda d:pytest.fail('must reuse sealed stage'))
    assert result=={'done':True}
    assert json.loads((tmp_path/'stages.json').read_text())['stages']['example']['status']=='completed'


def test_unsealed_stage_still_blocks_recovery(tmp_path):
    (tmp_path/'example').mkdir()
    with pytest.raises(ValueError,match='без паспорта'):Stages(tmp_path,'identity').run('example',lambda d:None)


def test_verified_public_cache_is_reusable_without_network(tmp_path,monkeypatch):
    from global_weather.providers.http import download
    from global_weather.pipeline.io import sha256
    url=station_url('USW00014933',2020);p=tmp_path/'archive.psv';p.write_text('not weather: cache integrity test')
    row={'url':url,'sha256':sha256(p),'bytes':p.stat().st_size,'acquired_at':'2026-01-01T00:00:00Z'}
    atomic_json(p.with_suffix('.psv.receipt.json'),row)
    assert download(url,p,network=False)==row
    p.write_text('changed')
    with pytest.raises(ValueError,match='повреждён'):download(url,p)


@pytest.mark.parametrize('url',['http://www.ncei.noaa.gov/a','https://example.invalid/a','https://www.ncei.noaa.gov/a?key=secret'])
def test_public_download_rejects_untrusted_urls(tmp_path,url):
    from global_weather.providers.http import download
    with pytest.raises(ValueError):download(url,tmp_path/'never',network=True)
    assert not (tmp_path/'never').exists()


def test_cf_collection_dispatches_distinct_time_files_and_rejects_overlap(tmp_path):
    import xarray as xr
    from global_weather.pipeline.era5 import CFCollection
    from global_weather.pipeline.dataset import utc
    from global_weather.grid import build_grid
    coordinates={'latitude':('latitude',[-90.,0.,90.],{'units':'degrees_north'}),
                 'longitude':('longitude',[0.,90.,180.,270.],{'units':'degrees_east'})}
    paths=[]
    for h in (0,3):
        ds=xr.Dataset({'t':(('time','latitude','longitude'),np.full((1,3,4),260.+h),{'units':'K'})},
                      coords=dict(coordinates,time=[np.datetime64(f'2020-01-01T{h:02d}:00')]))
        path=tmp_path/f't{h}.nc';ds.to_netcdf(path,engine='scipy');paths.append(path)
    c=CFCollection(paths);xyz=build_grid(0).xyz
    try:
        assert np.allclose(c.values(('t',),'K',xyz,when=utc('2020-01-01T03:00:00Z')),263.)
        assert np.isnan(c.values(('t',),'K',xyz,when=utc('2020-01-01T06:00:00Z'))).all()
    finally:c.close()
    c=CFCollection([paths[0],paths[0]])
    try:
        with pytest.raises(ValueError,match='Перекрывающиеся'):c.values(('t',),'K',xyz,when=utc('2020-01-01T00:00:00Z'))
    finally:c.close()


def test_completed_epoch_report_can_be_recovered_without_retraining(tmp_path):
    from global_weather.pipeline.fixture import create_fixture
    from global_weather.pipeline.runner import train,TrainConfig
    from global_weather.pipeline.io import sha256
    dataset=create_fixture(tmp_path/'synthetic',horizon_hours=3);run=tmp_path/'run'
    cfg=TrainConfig(epochs=1,horizon_hours=3,device='cpu')
    train(dataset,run,cfg,progress=lambda *a,**k:None)
    weights=run/'epochs/000001/weights.pt';before=sha256(weights)
    (run/'report.json').unlink()
    train(dataset,run,cfg,resume=True,progress=lambda *a,**k:None)
    assert (run/'report.json').is_file() and sha256(weights)==before



def test_plan_is_stable_after_json_round_trip():
    checked=plan().checked()
    assert json.loads(json.dumps(checked))==checked


def test_autonomous_chain_contract_with_injected_synthetic_providers(tmp_path,monkeypatch):
    # Deliberately fake provider responses test orchestration, NOT source truth
    # or forecast skill. These files exist only under pytest's temporary root.
    import xarray as xr
    from datetime import timedelta
    from global_weather.autonomous.execute import execute
    from global_weather.pipeline.dataset import utc,PreparedDataset
    from global_weather.pipeline.io import digest,sha256
    from global_weather.providers import ghcnh,era5
    from global_weather.vertical import PRESSURE_HPA
    spec=plan(stations=['TEST0000000'],issue_interval_hours=72,horizon_hours=3,mesh_level=0,
              train_fraction=.4,validation_fraction=.3,training={'epochs':1,'device':'cpu','hidden':8,'threads':1})
    planned=spec.checked();request=tmp_path/'request.json';atomic_json(request,asdict(spec));calls=[]
    def station(station,year,cache,**kw):
        cache.mkdir(parents=True,exist_ok=True);p=cache/'synthetic-test-provider.txt';p.write_text('SYNTHETIC PROVIDER CONTRACT TEST')
        return p,{'acquired_at':'2026-01-01T00:00:00Z','sha256':sha256(p),'bytes':p.stat().st_size,'test_fixture':True}
    def observations(path,**kw):
        rows=[]
        for sample in planned['samples']:
            issue=utc(sample['issue_time']);observed=issue-timedelta(hours=2)
            rows.append({'observation_id':'test/'+sample['id'],'source':'station','variable':'t2m',
                'value':280.,'units':'K','latitude':0.,'longitude':0.,'observed_at':observed.isoformat(),
                'available_at':(observed+timedelta(hours=1)).isoformat(),'valid':True,'revision':0})
        return rows,{'test_fixture':True}
    def fields(query,cache,**kw):
        calls.append(query['id']);d=cache/digest(query);d.mkdir(parents=True)
        req=query['request'];day='-'.join(req[k][0] for k in ('year','month','day'))
        times=[np.datetime64(day+'T'+t) for t in req['time']]
        coords={'time':times,'latitude':('latitude',[-90.,0.,90.],{'units':'degrees_north'}),
                'longitude':('longitude',[0.,90.,180.,270.],{'units':'degrees_east'})}
        dims=('time','latitude','longitude');shape=(len(times),3,4)
        if 'pressure_level' in req:
            coords['pressure_level']=('pressure_level',list(PRESSURE_HPA),{'units':'hPa'})
            dims=('time','pressure_level','latitude','longitude');shape=(len(times),37,3,4)
        names={'temperature':('t',260.,'K'),'specific_humidity':('q',.003,'kg kg-1'),
               'u_component_of_wind':('u',1.,'m s-1'),'v_component_of_wind':('v',2.,'m s-1'),
               'geopotential':('z',100.,'m2 s-2'),'vertical_velocity':('w',.1,'Pa s-1'),
               '2m_temperature':('t2m',280.,'K'),'2m_dewpoint_temperature':('d2m',275.+int(req['day'][0])*.1,'K'),
               '10m_u_component_of_wind':('u10',1.,'m s-1'),'10m_v_component_of_wind':('v10',2.,'m s-1'),
               'surface_pressure':('sp',101000.,'Pa'),'mean_sea_level_pressure':('msl',101100.,'Pa'),
               'total_cloud_cover':('tcc',.5,'1'),'total_precipitation':('tp',int(req['day'][0])*.0001,'m'),
               'land_sea_mask':('lsm',.5,'1')}
        data={names[k][0]:(dims,np.full(shape,names[k][1],np.float32),{'units':names[k][2]}) for k in req['variable']}
        file=d/'field-0.nc';xr.Dataset(data,coords=coords).to_netcdf(file,engine='scipy')
        receipt={'request':query,'acquired_at':'2026-01-01T00:00:00Z','test_fixture':True,
                 'files':[{'name':file.name,'sha256':sha256(file),'bytes':file.stat().st_size}]}
        atomic_json(d/'receipt.json',receipt);return [file],receipt
    monkeypatch.setattr(ghcnh,'fetch_station',station);monkeypatch.setattr(ghcnh,'read_records',observations)
    monkeypatch.setattr(era5,'retrieve',fields)
    root=tmp_path/'test-only-experiment';first=execute(request,root,cds_key='test-credential-never-persisted')
    assert first['status']=='completed_research' and first['meteorologically_validated'] is False
    assert PreparedDataset(root/'prepared/dataset.json').validate()['status']=='prepared_data_validated'
    count=len(calls);second=execute(request,root)
    assert second==first and len(calls)==count
    assert 'test-credential-never-persisted' not in ''.join(p.read_text() for p in root.rglob('*.json'))
