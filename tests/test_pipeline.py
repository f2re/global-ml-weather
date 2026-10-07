"""Synthetic engineering tests; no meteorological skill is asserted."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from global_weather.pipeline.io import (read_arrays, write_arrays, local_path, artifact, reference,
                                        read_json, parse_json, atomic_json)
from global_weather.pipeline.dataset import PreparedDataset, utc
from global_weather.pipeline.fit import Moments, fit_normalization
from global_weather.pipeline.fixture import create_fixture


@pytest.fixture
def fixture(tmp_path):
    return create_fixture(tmp_path/'data')


def mutate_target(path, fn, index=0):
    m=read_json(path); ref=m['samples'][index]['targets']; p=path.parent/ref['path']
    arrays=read_arrays(p); fn(arrays)
    p.unlink(); write_arrays(p, **arrays); m['samples'][index]['targets']=reference(path.parent,p)
    atomic_json(path,m)


@pytest.mark.parametrize('name',['../x','/tmp/x','a/../../x','a\\x','https://x','a//b','./a'])
def test_dataset_path_escape(tmp_path,name):
    with pytest.raises(ValueError):local_path(tmp_path,name)


def test_symlink_is_not_an_asset(tmp_path):
    (tmp_path/'file').write_text('x');(tmp_path/'link').symlink_to(tmp_path/'file')
    with pytest.raises(ValueError):local_path(tmp_path,'link')


@pytest.mark.parametrize('text',['{"a":1,"a":2}','{"x":NaN}','{"x":Infinity}'])
def test_json_ambiguity_rejected(text):
    with pytest.raises(ValueError):parse_json(text)


def test_observation_budget_is_separate_from_manifest_budget(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from global_weather.pipeline import dataset,io as pipeline_io
    assert dataset.MAX_OBSERVATIONS_JSONL==256*1024**2
    assert pipeline_io.MAX_JSON==32*1024**2
    monkeypatch.setattr(pipeline_io,'MAX_JSON',8)
    monkeypatch.setattr(dataset,'MAX_OBSERVATIONS_JSONL',32)
    path=tmp_path/'observations.jsonl'
    path.write_text('\n'*32,encoding='utf-8')
    ds=PreparedDataset.__new__(PreparedDataset);ds.root=tmp_path;ds.registry={}
    sample=SimpleNamespace(observations=reference(tmp_path,path))
    assert ds.records(sample)==[]
    with pytest.raises(ValueError,match='ограниченным'):
        read_json(path)
    path.write_text('\n'*33,encoding='utf-8')
    sample.observations=reference(tmp_path,path)
    with pytest.raises(ValueError,match='слишком велик'):
        ds.records(sample)


def test_arrays_reject_objects_and_duplicate_members(tmp_path):
    p=tmp_path/'objects.npz';np.savez(p,x=np.array([{}],dtype=object))
    with pytest.raises(ValueError):read_arrays(p)
    b=io.BytesIO();np.save(b,np.ones(2));p=tmp_path/'duplicate.npz'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('x.npy',b.getvalue());z.writestr('x.npy',b.getvalue())
    with pytest.raises(ValueError):read_arrays(p)


def test_header_cannot_request_unbounded_allocation(tmp_path):
    b=io.BytesIO();np.lib.format.write_array_header_1_0(b,{'shape':(10**12,), 'fortran_order':False,'descr':'<f8'})
    p=tmp_path/'bomb.npz'
    with zipfile.ZipFile(p,'w') as z:z.writestr('x.npy',b.getvalue())
    with pytest.raises(ValueError):read_arrays(p)


def test_fingerprint_detects_changed_bytes(fixture):
    m=read_json(fixture);p=fixture.parent/m['static']['path'];p.write_bytes(p.read_bytes()+b'extra')
    with pytest.raises(ValueError,match='Измен'):PreparedDataset(fixture)


def test_complete_fixture_admission(fixture):
    r=PreparedDataset(fixture).validate()
    assert r['data_kind']=='synthetic' and len(r['samples'])==5 and r['source_truth_verified'] is False
    assert all(s['accepted_records']>0 for s in r['samples'])


def test_incomplete_profiles_do_not_create_values(fixture):
    def mask(d):
        d['profile_mask'][1,:,:,1]=False;d['profiles'][1,:,:,1]=np.nan
    mutate_target(fixture,mask)
    ds=PreparedDataset(fixture);d=ds.targets(ds.samples[0])
    assert not d['profile_mask'][1,:,:,1].any()


def test_missing_future_slot_is_not_zero_rain(fixture):
    def mask(d):
        d['profile_mask'][1]=False;d['surface_mask'][1]=False
        d['profiles'][1]=np.nan;d['surface'][1]=np.nan
    mutate_target(fixture,mask)
    ds=PreparedDataset(fixture);d=ds.targets(ds.samples[0]);assert not d['surface_mask'][1].any()


@pytest.mark.parametrize('field',['profile_units','pressure_hpa','issue_time'])
def test_target_axes_are_not_guessed(fixture,field):
    def change(d):
        if field=='profile_units':d[field]=np.array(['degC']+d[field].tolist()[1:])
        elif field=='pressure_hpa':d[field]=d[field][::-1]
        else:d[field]=np.array('2025-01-01T00:00:00Z')
    mutate_target(fixture,change)
    with pytest.raises(ValueError):PreparedDataset(fixture).targets(PreparedDataset(fixture).samples[0])


def test_below_ground_targets_rejected(fixture):
    mutate_target(fixture,lambda d:d['surface'].__setitem__((slice(None),slice(None),4),80000.))
    ds=PreparedDataset(fixture)
    with pytest.raises(ValueError,match='поверхности'):ds.targets(ds.samples[0])


def test_profile_needs_target_surface_pressure(fixture):
    mutate_target(fixture,lambda d:d['surface_mask'].__setitem__((slice(None),slice(None),4),False))
    ds=PreparedDataset(fixture)
    with pytest.raises(ValueError,match='давление'):ds.targets(ds.samples[0])


def test_time_splits_cannot_overlap(fixture):
    m=read_json(fixture);m['samples'][3]['issue_time']='2020-01-13T14:00:00Z';atomic_json(fixture,m)
    with pytest.raises(ValueError,match='окна'):PreparedDataset(fixture)


def test_norm_fit_never_reads_validation_or_test_targets(fixture,monkeypatch):
    original=PreparedDataset.targets
    def guarded(self,sample):
        if sample.split!='train':raise AssertionError('held-out targets read while fitting')
        return original(self,sample)
    monkeypatch.setattr(PreparedDataset,'targets',guarded)
    fit_normalization(fixture.parent/'unscaled.json',fixture.parent/'refit.json')
    a=read_json(fixture.parent/'dataset.normalization.json')['variables']
    b=read_json(fixture.parent/'refit.normalization.json')['variables']
    assert a==b


def test_validation_changes_do_not_change_training_means(fixture):
    before=read_json(fixture.parent/'dataset.normalization.json')['variables']
    raw=fixture.parent/'unscaled.json'
    mutate_target(raw,lambda d:d['profiles'].__setitem__((slice(None),slice(None),slice(None),0),999.),index=3)
    fit_normalization(raw,fixture.parent/'refit.json')
    assert before==read_json(fixture.parent/'refit.normalization.json')['variables']


def test_streaming_weighted_moments_and_degenerate_variance():
    m=Moments();m.add(np.array([1.,3.]),np.ones(2,bool),np.array([1.,3.]));m.add(np.array([5.]),np.ones(1,bool),np.array([2.]))
    mean,std=m.finish();assert mean==pytest.approx(20/6)
    expected=np.sqrt((1*(1-20/6)**2+3*(3-20/6)**2+2*(5-20/6)**2)/6)
    assert std==pytest.approx(expected)
    m=Moments();m.add(np.ones(4),np.ones(4,bool),1.)
    with pytest.raises(ValueError):m.finish()


def test_unknown_source_period_is_not_inherited_as_known(fixture):
    b=read_json(fixture.parent/'dataset.normalization.json');b['provenance']['fit_period']=None
    atomic_json(fixture.parent/'unknown.json',b)
    with pytest.raises(ValueError):fit_normalization(fixture.parent/'unscaled.json',fixture.parent/'bad.json',base_path=fixture.parent/'unknown.json')
    assert not (fixture.parent/'bad.json').exists()


def test_synthetic_norms_cannot_be_relabelled_real(fixture):
    m=read_json(fixture);m['data_kind']='real';atomic_json(fixture,m)
    with pytest.raises(ValueError,match='происхождение'):PreparedDataset(fixture)


def test_norm_accumulation_interval_is_checked(fixture):
    m=read_json(fixture);path=fixture.parent/m['normalization']['path'];norm=read_json(path)
    norm['variables']['precipitation_step']['interval_hours']=6;atomic_json(path,norm)
    m['normalization']=reference(fixture.parent,path);atomic_json(fixture,m)
    with pytest.raises(ValueError,match='interval'):PreparedDataset(fixture)


def test_cli_fixture_does_not_download(tmp_path,monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('network')))
    from global_weather.pipeline.__main__ import main
    main(['demo-dataset','--output',str(tmp_path/'demo')])
    assert (tmp_path/'demo/dataset.json').is_file()


def cf_sources(tmp_path, *, missing_hour=False):
    import xarray as xr
    from global_weather.vertical import PRESSURE_HPA
    lat=np.array([-90.,-60.,-30.,0.,30.,60.,90.]);lon=np.arange(0.,360.,45.)
    times=np.arange(np.datetime64('2020-01-01T00'),np.datetime64('2020-01-01T07'),np.timedelta64(1,'h'))
    coords={'time':times,'latitude':('latitude',lat,{'units':'degrees_north'}),
            'longitude':('longitude',lon,{'units':'degrees_east'}),
            'pressure_level':('pressure_level',list(PRESSURE_HPA),{'units':'hPa'})}
    shape=(len(times),37,len(lat),len(lon))
    values={k:(('time','pressure_level','latitude','longitude'),np.full(shape,v,np.float32),{'units':u})
            for k,v,u in [('t',260.,'K'),('q',.003,'kg kg-1'),('u',1.,'m s-1'),('v',2.,'m s-1'),('z',100.,'m2 s-2'),('w',.1,'Pa s-1')]}
    pressure=xr.Dataset(values,coords=coords);pp=tmp_path/'pressure.nc';pressure.to_netcdf(pp,engine='scipy')
    shape=(len(times),len(lat),len(lon))
    values={k:(('time','latitude','longitude'),np.full(shape,v,np.float32),{'units':u})
            for k,v,u in [('t2m',280.,'K'),('d2m',275.,'K'),('u10',1.,'m s-1'),('v10',2.,'m s-1'),('sp',101000.,'Pa'),('msl',101100.,'Pa'),('tcc',.5,'1')]}
    tp=np.broadcast_to(np.arange(len(times))[:,None,None]/1000.,shape).copy()
    if missing_hour:tp[2]=np.nan
    values['tp']=(('time','latitude','longitude'),tp,{'units':'m'})
    s=xr.Dataset(values,coords={k:v for k,v in coords.items() if k!='pressure_level'});sp=tmp_path/'surface.nc';s.to_netcdf(sp,engine='scipy')
    return pp,sp


def test_era5_precipitation_uses_all_hours_and_explicit_units(tmp_path):
    from global_weather.pipeline.era5 import prepare_targets
    pp,sp=cf_sources(tmp_path);out=tmp_path/'target.npz'
    r=prepare_targets(pp,sp,out,issue_time='2020-01-01T00:00:00Z',mesh_level=0,horizon_hours=3,
                       confirm_utc=True,precipitation_kind='hourly_increment')
    d=read_arrays(out);assert np.allclose(d['surface'][1,:,6],6.)
    assert not d['surface_mask'][0,:,6].any()
    assert d['profiles'].shape==(2,12,37,6) and 'NOT conservative' in r['operator']


def test_era5_missing_hour_invalidates_precipitation(tmp_path):
    from global_weather.pipeline.era5 import prepare_targets
    pp,sp=cf_sources(tmp_path,missing_hour=True);out=tmp_path/'target.npz'
    prepare_targets(pp,sp,out,issue_time='2020-01-01T00:00:00Z',mesh_level=0,horizon_hours=3,
                    confirm_utc=True,precipitation_kind='hourly_increment')
    d=read_arrays(out);assert not d['surface_mask'][1,:,6].any()


def test_era5_unknown_time_convention_refused(tmp_path):
    from global_weather.pipeline.era5 import prepare_targets
    with pytest.raises(ValueError,match='UTC'):prepare_targets('a','b',tmp_path/'x.npz',issue_time='2020-01-01T00:00:00Z',mesh_level=0)


def test_era5_unobserved_longitude_gap_not_filled(tmp_path):
    import xarray as xr
    from global_weather.pipeline.era5 import CFField
    from global_weather.grid import unit_xyz
    p=tmp_path/'regional.nc'
    xr.Dataset({'t':(('latitude','longitude'),np.ones((3,5))*260,{'units':'K'})},
      coords={'latitude':('latitude',[-10.,0.,10.],{'units':'degrees_north'}),
              'longitude':('longitude',[-20.,-10.,0.,10.,20.],{'units':'degrees_east'})}).to_netcdf(p,engine='scipy')
    f=CFField(p)
    try:v=f.values(('temperature','t'),'K',unit_xyz(np.array([0.,0.]),np.array([10.,180.])))
    finally:f.close()
    assert v[0]==pytest.approx(260) and np.isnan(v[1])


def test_era5_scaling_only_once(tmp_path):
    import xarray as xr
    from global_weather.pipeline.era5 import CFField
    from global_weather.grid import unit_xyz
    p=tmp_path/'scaled.nc'
    d=xr.Dataset({'t':(('latitude','longitude'),np.ones((3,4))*260,{'units':'K'})},
      coords={'latitude':('latitude',[-90.,0.,90.],{'units':'degrees_north'}),
              'longitude':('longitude',[0.,90.,180.,270.],{'units':'degrees_east'})})
    d.to_netcdf(p,engine='scipy',encoding={'t':{'dtype':'int16','scale_factor':.1,'add_offset':250.,'_FillValue':-32768}})
    f=CFField(p)
    try:v=f.values(('temperature','t'),'K',unit_xyz(np.array([0.]),np.array([45.])))
    finally:f.close()
    assert v[0]==pytest.approx(260)


def test_adaptive_pipeline_train_resume_evaluate_and_no_target_forecast(fixture,tmp_path,monkeypatch):
    import torch
    from global_weather.pipeline.runner import TrainConfig,train,evaluate,forecast
    original=PreparedDataset.targets
    def no_test(self,s):
        if s.split=='test':raise AssertionError('test used in model selection')
        return original(self,s)
    monkeypatch.setattr(PreparedDataset,'targets',no_test)
    run=tmp_path/'run';r=train(fixture,run,TrainConfig(epochs=1),progress=lambda *a,**k:None)
    assert not r['test_set_used_for_selection'] and r['epochs_completed']==1
    r=train(fixture,run,TrainConfig(epochs=2),resume=True,progress=lambda *a,**k:None)
    assert r['epochs_completed']==2
    monkeypatch.setattr(PreparedDataset,'targets',original)
    e=evaluate(fixture,run,tmp_path/'evaluation.json');assert e['scores'] and e['split']=='test'
    assert all(x['rmse']>=0 for x in e['scores'])
    monkeypatch.setattr(PreparedDataset,'targets',lambda *a:(_ for _ in ()).throw(AssertionError('future targets in forecast')))
    f=forecast(fixture,run,'sample-4',tmp_path/'forecast',horizon_hours=3)
    assert f['targets_read'] is False and f['data_kind']=='synthetic'
    assert (tmp_path/'forecast/frame_003.npz').is_file()


def test_adaptive_resume_is_identical_to_uninterrupted(fixture,tmp_path):
    import torch
    from global_weather.pipeline.runner import TrainConfig,train
    a,b=tmp_path/'a',tmp_path/'b'
    train(fixture,a,TrainConfig(epochs=2),progress=lambda *a,**k:None)
    train(fixture,b,TrainConfig(epochs=1),progress=lambda *a,**k:None)
    train(fixture,b,TrainConfig(epochs=2),resume=True,progress=lambda *a,**k:None)
    x=torch.load(a/'epochs/000002/weights.pt',weights_only=True)['state_dict']
    y=torch.load(b/'epochs/000002/weights.pt',weights_only=True)['state_dict']
    for k,v in x.items():
        if isinstance(v,torch.Tensor):
            if v.is_sparse:assert torch.equal(v.coalesce().values(),y[k].coalesce().values())
            else:assert torch.equal(v,y[k]),k
        else:assert v==y[k]


def test_adaptive_resume_rejects_changed_hyperparameters(fixture,tmp_path):
    from global_weather.pipeline.runner import TrainConfig,train
    run=tmp_path/'run';train(fixture,run,TrainConfig(epochs=1),progress=lambda *a,**k:None)
    with pytest.raises(ValueError):train(fixture,run,TrainConfig(epochs=2,learning_rate=.001),resume=True)


def test_adaptive_forecast_input_without_targets(fixture,tmp_path):
    from global_weather.pipeline.runner import TrainConfig,train,forecast
    run=tmp_path/'run';train(fixture,run,TrainConfig(epochs=1),progress=lambda *a,**k:None)
    m=read_json(fixture);m['schema']='global-weather-input-1';m['samples']=[m['samples'][4]]
    m['samples'][0]['split']='inference';del m['samples'][0]['targets']
    path=fixture.parent/'input.json';atomic_json(path,m)
    with pytest.raises(ValueError):PreparedDataset(path)
    f=forecast(path,run,'sample-4',tmp_path/'forecast',horizon_hours=3)
    assert f['targets_read'] is False


def test_pipeline_ui_requires_csrf_and_bounded_dataset(fixture,tmp_path):
    from fastapi.testclient import TestClient
    from global_weather.lab.app import create_app
    import shutil
    workspace=tmp_path/'lab';(workspace/'datasets').mkdir(parents=True)
    shutil.copytree(fixture.parent,workspace/'datasets/demo')
    with TestClient(create_app(workspace,testing=True)) as c:
        assert c.get('/training').status_code==200
        assert c.get('/api/pipeline/datasets').json()[0]['id']=='demo'
        payload={'kind':'validate_dataset','dataset_id':'demo','horizon_hours':3}
        assert c.post('/api/pipeline/runs',json=payload).status_code==403
        token=c.get('/api/bootstrap').json()['csrf'];headers={'x-lab-csrf':token}
        assert c.post('/api/pipeline/runs?role=coordinator',json={**payload,'kind':'train_dataset'},headers=headers).status_code==400
        r=c.post('/api/pipeline/runs',json=payload,headers=headers);assert r.status_code==200
        assert c.post('/api/pipeline/runs',json={**payload,'dataset_id':'../../outside'},headers=headers).status_code==422


def test_in_memory_dataset_detects_changed_static_or_manifest(fixture):
    ds=PreparedDataset(fixture);ds.assert_unchanged()
    m=read_json(fixture);m['license']='changed';atomic_json(fixture,m)
    with pytest.raises(ValueError,match='изменился'):ds.assert_unchanged()


def test_nonfinite_optimizer_state_rejected():
    import torch
    from global_weather.pipeline.runner import _finite_state
    with pytest.raises(ValueError):_finite_state({'state':{'exp_avg':torch.tensor([float('nan')])}})


def test_curriculum_reaches_requested_horizon(fixture):
    from global_weather.pipeline.runner import TrainConfig
    ds=PreparedDataset(fixture)
    with pytest.raises(ValueError):TrainConfig(epochs=1,curriculum=((1,3),(2,3))).validate(ds)


def test_pipeline_dataset_root_cannot_be_symlink(fixture,tmp_path):
    from global_weather.lab.pipeline_jobs import dataset_path
    ws=tmp_path/'workspace';ws.mkdir();(ws/'datasets').symlink_to(fixture.parent,target_is_directory=True)
    with pytest.raises(ValueError):dataset_path(ws,'demo')


def test_pipeline_roles_expose_fixed_actions():
    from global_weather.lab.agents import ROLES,authorize
    assert len(ROLES)==9
    assert authorize('executor','train_dataset')['id']=='executor'
    with pytest.raises(ValueError):authorize('physics','train_dataset')
    assert authorize('verification','evaluate_dataset')['id']=='verification'


def test_withdrawn_revision_does_not_revive_old_value(fixture):
    m=read_json(fixture);sample=m['samples'][0]
    path=fixture.parent/sample['observations']['path']
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    old=rows[0];withdrawn={**old,'revision':1,'valid':False,'value':None}
    path.write_text(''.join(json.dumps(r)+'\n' for r in [*rows,withdrawn]))
    sample['observations']=reference(fixture.parent,path);atomic_json(fixture,m)
    ds=PreparedDataset(fixture)
    admitted=ds.eligible_records(ds.samples[0])
    assert old['observation_id'] not in [r['observation_id'] for r in admitted]
    assert ds.packed(ds.samples[0]).accepted_records==len(rows)-1


def test_era5_wind_interpolates_vectors_not_scalar_components(tmp_path):
    import xarray as xr
    from global_weather.pipeline.era5 import CFField
    from global_weather.grid import unit_xyz
    lon=np.arange(0.,360.,90.);lat=np.array([-90.,0.,90.])
    u=np.broadcast_to(-np.sin(np.deg2rad(lon)),(3,4))
    p=tmp_path/'wind.nc'
    xr.Dataset({'u':(('latitude','longitude'),u,{'units':'m s-1'}),
                'v':(('latitude','longitude'),np.zeros((3,4)),{'units':'m s-1'})},
               coords={'latitude':('latitude',lat,{'units':'degrees_north'}),
                       'longitude':('longitude',lon,{'units':'degrees_east'})}).to_netcdf(p,engine='scipy')
    f=CFField(p)
    try:
        east,north=f.wind(('u',),('v',),unit_xyz(np.array([0.]),np.array([45.])))
    finally:f.close()
    assert east[0]==pytest.approx(-np.sqrt(2)/4,abs=1e-6)
    assert north[0]==pytest.approx(0.)
