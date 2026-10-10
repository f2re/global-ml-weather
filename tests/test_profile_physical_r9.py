"""R9 analytical/structural checks. No remote archive or weather skill claim."""
from datetime import timedelta
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from global_weather.grid import build_grid
from global_weather.profile_physics import (PhysicalPolicy, parse_policy,
    pressure_layer_thickness, saturation_pressure_water, screen_humidity,
    hydrostatic_projection, physical_objective, RD, EPSILON)
from global_weather.profile_hydrostatic_model import HydrostaticFlowProfileModel
from global_weather.profile_training import START, VARIABLES
from global_weather.vertical import PRESSURE_HPA, PROFILE_UNITS
from test_profile_seasonal_model import normalization, rows, ClimateFixture


def observation(variable='specific_humidity', value=.001, pressure=10000., **kwargs):
    return dict(variable=variable, value=value, pressure_pa=pressure,
                observed_at=START.isoformat(), available_at=START.isoformat(),
                latitude=60., longitude=20., profile_id='unregistered-launch',
                provider='test-fixture', observation_id=variable, **kwargs)


@pytest.mark.parametrize('field,value', [('schema','bad'),('architecture','solver'),
    ('huber_delta',0),('huber_delta',float('nan')),('humidity_qc','yes'),
    ('humidity_min_pressure_pa',-1),('maximum_water_relative_humidity',.5),
    ('memory_budget_mib',True),('variable_weights',[1,1,1,0,1])])
def test_policy_rejects_ambiguous_values(field,value):
    with pytest.raises(ValueError): parse_policy({field:value})


def test_pressure_quadrature_and_explicit_policy_round_trip():
    p=np.array(PRESSURE_HPA)*100
    dp=pressure_layer_thickness(p)
    assert (dp>0).all() and dp.sum()==pytest.approx(p[0]-p[-1])
    assert PhysicalPolicy().payload()==parse_policy(PhysicalPolicy().payload()).payload()
    assert PhysicalPolicy().humidity_min_pressure_pa==0
    for invalid in ([100,100],[100,0],[100,float('nan')],[1,10]):
        with pytest.raises(ValueError): pressure_layer_thickness(invalid)


def test_gross_humidity_screen_uses_paired_observations_not_a_pressure_ban():
    q_bad=observation(value=.001)
    q_good=observation(value=1e-6, observation_error=1e-7)
    t=observation('temperature',200.)
    wind=observation('u',8.)
    original=json.dumps([q_bad,t,wind])
    retained,report=screen_humidity([q_bad,t,wind],PhysicalPolicy())
    assert retained==[t,wind] and report['excluded_records']==1
    assert report['reasons']['humidity_paired_thermodynamic_gross_error']==1
    assert json.dumps([q_bad,t,wind])==original
    retained,report=screen_humidity([q_good,t],PhysicalPolicy())
    assert retained==[q_good,t] and not report['excluded_records']
    assert saturation_pressure_water(273.15)==pytest.approx(611.21,rel=.001)


def test_unknown_temperature_or_another_launch_never_becomes_a_false_rejection():
    q=observation(value=.001)
    t=dict(observation('temperature',190.),profile_id='different-launch')
    retained,report=screen_humidity([q,t],PhysicalPolicy())
    assert retained==[q,t]
    assert report['reasons']['humidity_unchecked_missing_or_conflicting_paired_temperature']==1
    # Different time/position is not a thermodynamic pair.
    t=dict(t,profile_id=q['profile_id'],observed_at=(START+timedelta(seconds=1)).isoformat())
    assert screen_humidity([q,t],PhysicalPolicy())[0]==[q,t]


def test_pressure_cutoff_is_explicit_sensitivity_only_and_other_variables_survive():
    q=observation(value=1e-6);t=observation('temperature',200.)
    assert screen_humidity([q,t],PhysicalPolicy())[0]==[q,t]
    filtered,report=screen_humidity([q,t],PhysicalPolicy(humidity_min_pressure_pa=25000))
    assert filtered==[t]
    assert report['reasons']['humidity_explicit_pressure_sensitivity_exclusion']==1
    assert screen_humidity([q,t],PhysicalPolicy(humidity_qc=False))[0]==[q,t]


def test_hydrostatic_projection_moist_isothermal_free_anchor_and_gradients():
    p=torch.tensor([100000.,70000.,30000.,10000.],dtype=torch.float64)
    t=torch.full((2,4),270.,dtype=torch.float64,requires_grad=True)
    q=torch.full((2,4),.004,dtype=torch.float64,requires_grad=True)
    raw=torch.tensor([[1000.,40000.,100000.,200000.]]*2,dtype=torch.float64,requires_grad=True)
    w=torch.tensor([2.,1.,3.,4.],dtype=torch.float64)
    result=hydrostatic_projection(t,q,raw,p,w)
    expected=RD*270*(1+(1/EPSILON-1)*.004)*torch.log(p[0]/p)
    assert torch.allclose(result-result[:,:1],expected,rtol=1e-12,atol=1e-9)
    assert torch.allclose(((result-raw)*w).sum(-1),torch.zeros(2,dtype=torch.float64),atol=1e-8)
    shifted=hydrostatic_projection(t,q,raw+1234.,p,w)
    assert torch.allclose(shifted,result+1234.,atol=1e-9)
    result.square().mean().backward()
    for value in (t,q,raw):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0


def test_hydrostatic_projection_rejects_missing_columns():
    p=torch.tensor([100000.,10000.]); t=torch.tensor([[270.,220.]])
    with pytest.raises(ValueError):hydrostatic_projection(t,torch.full_like(t,float('nan')),t,p)
    with pytest.raises(ValueError):hydrostatic_projection(t,torch.ones_like(t),t,p)
    with pytest.raises(ValueError):hydrostatic_projection(t,torch.zeros_like(t),t,p.flip(0))


def test_s1_zero_horizon_preserves_analysis_and_does_not_execute_dynamics(monkeypatch):
    from global_weather.profile_model_v2 import PressureProfileModel
    torch.manual_seed(4);model=PressureProfileModel(build_grid(0),normalization(),8)
    issue=START+timedelta(days=3);inputs=rows(issue-timedelta(hours=6))
    complete=model(inputs,issue)
    monkeypatch.setattr(model,'_advance_state',lambda *a:pytest.fail('S1 must not run future dynamics'))
    initial=model(inputs,issue,horizon_hours=0)
    assert len(initial)==1 and torch.equal(initial[0].profiles[...,:5],complete[0].profiles[...,:5])
    for invalid in (-1,1,73,True):
        with pytest.raises(ValueError):model([],issue,horizon_hours=invalid)


@pytest.mark.parametrize('climate',[False,True])
def test_new_model_hydrostatics_causality_masks_and_all_branch_gradients(climate):
    torch.manual_seed(29);grid=build_grid(0);norms=normalization()
    context=ClimateFixture(grid,norms) if climate else None
    model=HydrostaticFlowProfileModel(grid,norms,8,context)
    policy=PhysicalPolicy(architecture='hydrostatic_flow');model.physical_policy=policy
    issue=START+timedelta(days=3);inputs=rows(issue-timedelta(hours=6))
    before={n:v.clone() for n,v in model.named_buffers() if n in ('mean','std','norm_support')}
    frames=model(inputs,issue)
    assert len(frames)==25 and frames[-1].lead_hours==72
    for frame in frames:
        t,q,phi=frame.profiles[...,0],frame.profiles[...,1],frame.profiles[...,4]
        tv=t*(1+(1/EPSILON-1)*q)
        reference=RD*.5*(tv[:,:-1]+tv[:,1:])*torch.log(model.pressure_pa[:-1]/model.pressure_pa[1:])
        assert torch.allclose(phi[:,1:]-phi[:,:-1],reference,atol=.15,rtol=3e-5)
        assert (t>0).all() and (q>=0).all() and torch.isfinite(frame.profiles[...,:5]).all()
        assert torch.isnan(frame.profiles[...,5]).all() and not frame.surface_mask.any()
    with torch.no_grad():
        polluted=model(inputs+rows(issue+timedelta(minutes=17)),issue)
    for a,b in zip(frames,polluted):assert torch.equal(a.profiles[...,:5],b.profiles[...,:5])
    loss,counts,report=physical_objective(model,frames,rows(issue+timedelta(hours=12)),policy)
    assert counts==[1]*5 and report['normalization_changed'] is False
    loss.backward()
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None,name
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0,name
    assert (model.head.weight.grad.abs().sum(1)>0).all()
    for name,value in before.items():assert torch.equal(value,dict(model.named_buffers())[name])


def test_flow_conditioning_keeps_constant_state_and_responds_to_wind():
    torch.manual_seed(11);model=HydrostaticFlowProfileModel(build_grid(0),normalization(),8)
    state=torch.randn(12,37,8)
    frame=model._decode(state,START,0);frame.profiles=frame.profiles.detach().clone()
    frame.profiles[...,2:4]=0
    assert torch.allclose(model._flow_neighbours(state,frame),model.graph.neighbours(state),atol=1e-6)
    assert torch.allclose(model._flow_neighbours(torch.ones_like(state),frame),torch.ones_like(state))
    frame.profiles[...,2]=100.
    east=model._flow_neighbours(state,frame)
    frame.profiles[...,2]=-100.
    west=model._flow_neighbours(state,frame)
    assert not torch.allclose(east,west)


def test_robust_loss_grouping_and_gradient_without_changing_sigma(monkeypatch):
    from global_weather.analysis import observation_operator
    parameter=torch.tensor(1000.,requires_grad=True)
    class Norm:
        humidity_transform='identity'
        def at(self,*a):return 0.,1.,True
    model=SimpleNamespace(normalization=Norm(),grid=None,pressure_pa=torch.tensor(PRESSURE_HPA)*100)
    monkeypatch.setattr(observation_operator,'_prediction',lambda *a:(parameter,None))
    targets=[observation(value=0.)]
    frames=[SimpleNamespace(valid_time=START)]
    a,_,_=physical_objective(model,frames,targets,PhysicalPolicy())
    b,_,_=physical_objective(model,frames,targets*25,PhysicalPolicy())
    assert torch.equal(a,b)
    a.backward();assert parameter.grad==pytest.approx(2.5)
    # Original normalized MSE would give derivative 2000 for this residual.
    uncertain=[dict(targets[0],observation_error=3.)]
    c,_,_=physical_objective(model,frames,uncertain,PhysicalPolicy())
    assert c<a
    with pytest.raises(ValueError):physical_objective(model,frames,[dict(targets[0],observation_error=-1)],PhysicalPolicy())


def test_config_preserves_legacy_limits_and_checks_new_resource_budget():
    from global_weather.profile_training_v2 import _configuration
    with pytest.raises(ValueError):_configuration({'mesh_level':3})
    cfg=_configuration({'mesh_level':3,'hidden':64,'physical_policy':{}})
    assert cfg['mesh_level']==3 and cfg['physical_policy']['humidity_min_pressure_pa']==0
    with pytest.raises(ValueError):_configuration({'mesh_level':4,'hidden':128,'physical_policy':{'memory_budget_mib':128}})


@pytest.mark.parametrize('architecture',['legacy','hydrostatic_flow'])
def test_r9_measured_pipeline_contract_resume_and_physical_diagnostics(tmp_path,monkeypatch,architecture):
    # Synthetic records with an explicitly marked test fixture; no remote
    # measurements acquired. The genuine fixed GraphCast norm bytes are reused.
    from global_weather import profile_training_v2 as training
    from global_weather.profile_training import prepare,TRAIN_END,VAL_END
    from global_weather.profile_graphcast_normalization import create
    from global_weather.profile_verification import _exact_tensor_equal
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    data=[]
    for start,split in ((START,'train'),(TRAIN_END,'validation'),(VAL_END,'test')):
        for day in (2,3):
            when=start+timedelta(days=day)
            for i,row in enumerate(rows(when)):
                row.update(value=row['value']+(day-2)*[2,.0002,2,2,100][i],source='radiosonde',provider='NOAA_IGRA2',valid=True,
                     observation_id=f'{when}/{i}',profile_id=str(when),group_split=split,revision=0,provider_qc={'software_fixture':True},
                     archive_sha256='0'*64,format_sha256='1'*64,time_basis='reported_level_time')
                data.append(row)
    source=tmp_path/'observations.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in data))
    (tmp_path/'manifest.json').write_text(json.dumps({'schema':'igra-observation-archive-1','provider':'NOAA_IGRA2','data_kind':'real',
          'observations_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}))
    dataset=tmp_path/'dataset';prepare(source,dataset);norm_path=tmp_path/'norms.json';create(dataset,norm_path)
    config=dict(mesh_level=0,hidden=8,epochs=2,patience=3,threads=1,max_train_issues=1,max_validation_issues=1,max_test_issues=1,
                max_records_per_window=50,physical_policy={'architecture':architecture})
    training.train(dataset,norm_path,tmp_path/'whole',config)
    original_score=training.score;calls=0
    def interrupted(*args):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('synthetic interruption before publishing second epoch')
        return original_score(*args)
    monkeypatch.setattr(training,'score',interrupted)
    with pytest.raises(RuntimeError,match='interruption'):training.train(dataset,norm_path,tmp_path/'resumed',config)
    monkeypatch.setattr(training,'score',original_score);training.train(dataset,norm_path,tmp_path/'resumed',config)
    a=torch.load(tmp_path/'whole/epoch-0002/state.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/epoch-0002/state.pt',weights_only=True)
    assert all(_exact_tensor_equal(a['model'][name],b['model'][name]) for name in a['model'])
    training.evaluate(dataset,tmp_path/'whole',tmp_path/'test.json')
    report=json.loads((tmp_path/'test.json').read_text())
    assert report['metrics_by_pressure'] and report['scientific_acceptance'] is False
    assert report['physical_policy']['architecture']==architecture
    assert (tmp_path/'whole/qc-queries.json').is_file()
    state=a['model'];state['std']=state['std'].clone()+1
    model,_=training.load_frozen(dataset,tmp_path/'whole')
    with pytest.raises(ValueError,match='fixed normalization'):training._load_fixed_state(model,state)
