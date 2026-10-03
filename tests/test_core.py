from dataclasses import replace
from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
import torch
from global_weather.grid import build_grid,build_pyramid,latlon,EARTH_RADIUS_M
from global_weather.contracts import ObservationEvent,select_as_issued,RadiometrySpec
from global_weather.observations import pack_observations,DEFAULT_VARIABLES,Variable,read_jsonl
from global_weather.model import GlobalWeatherModel
from global_weather.vertical import PRESSURE_HPA,above_ground,tangent_components,hydrostatic_residual,validate_levels
from global_weather.losses import masked_huber,forecast_loss
from global_weather.cli import main

torch.set_num_threads(1)
TIME=datetime(2026,10,3,12,tzinfo=timezone.utc)
P=np.array(PRESSURE_HPA)*100

@pytest.fixture(scope='module')
def grids(): return build_pyramid(1)

def record(**kw):
    rec=dict(observation_id='a',source='station',variable='t2m',value=280.,units='K',
             latitude=60.,longitude=30.,observed_at=(TIME-timedelta(hours=1)).isoformat(),
             available_at=TIME.isoformat(),valid=True)
    rec.update(kw); return rec

def packed(grids,records=None,variables=None):
    return pack_observations([record()] if records is None else records,grids[0],P,TIME,variables)

def net(grids,obs,step=3):
    torch.manual_seed(4)
    return GlobalWeatherModel(grids,obs.vocabulary,observation_schema=obs.schema_fingerprint,hidden=16,step_hours=step)

def static(grids):
    return torch.zeros(grids[0].n_cells),torch.zeros(grids[0].n_cells)

@pytest.mark.parametrize('level',[0,1,2,3])
def test_spherical_geometry(level):
    g=build_grid(level)
    assert g.n_cells==10*4**level+2
    assert len(g.faces)==20*4**level
    assert len(g.edges[0])==6*g.n_cells-12
    assert sum(len(r)==5 for r in g.regions)==12
    assert all(len(r) in (5,6) for r in g.regions)
    assert g.areas_m2.sum()==pytest.approx(4*np.pi*EARTH_RADIUS_M**2,rel=1e-10)
    assert (g.areas_m2>0).all()
    assert g.n_cells-len(g.edges[0])//2+len(g.faces)==2

def test_edges_bidirectional(grids):
    e=set(map(tuple,grids[0].edges.T))
    assert all((b,a) in e for a,b in e)

def test_grid_reproducible(grids):
    assert build_grid(1).fingerprint==grids[0].fingerprint

def test_dateline_and_poles(grids):
    g=grids[0]
    assert g.locate(0.,180.)==g.locate(0.,-180.)
    assert g.locate(90.,0.)==g.locate(90.,179.)
    assert g.locate(-90.,0.)==g.locate(-90.,-179.)

def test_mask_crosses_dateline(grids):
    g=grids[0]; ll=latlon(g.xyz)
    expected=(ll[:,1]>=150)|(ll[:,1]<=-150)
    assert np.array_equal(g.region_mask(-90,90,150,-150),expected)

def test_build_memory_guard():
    with pytest.raises(ValueError): build_grid(8)

@pytest.mark.parametrize('levels',[(100,100),(100,200),(100,np.nan),(100,0)])
def test_bad_pressure_axis(levels):
    with pytest.raises(ValueError): validate_levels(levels)

def test_37_levels():
    assert len(P)==37 and P[0]==100000 and P[-1]==100

def test_above_ground():
    m=above_ground(torch.tensor([100000.,85000.,50000.]),torch.tensor([90000.]))
    assert m.tolist()==[[False,True,True]]

def test_late_messages_excluded(grids):
    o=packed(grids,[record(available_at=(TIME+timedelta(seconds=1)).isoformat())])
    assert o.accepted_records==0

def test_history_half_open(grids):
    rs=[record(observation_id=str(i),observed_at=(TIME-timedelta(hours=i)).isoformat()) for i in range(13)]
    assert packed(grids,rs).accepted_records==12

def test_timezone_required(grids):
    with pytest.raises(ValueError): pack_observations([],grids[0],P,TIME.replace(tzinfo=None))

def test_latest_available_revision(grids):
    rs=[record(value=270.,revision=0),record(value=290.,revision=1),
        record(value=300.,revision=2,available_at=(TIME+timedelta(hours=1)).isoformat())]
    o=packed(grids,rs)
    assert o.accepted_records==1 and o.features[0,0]==pytest.approx((290-273.15)/30)

def test_rejected_revision_does_not_revive_old_value(grids):
    o=packed(grids,[record(revision=0),record(revision=1,valid=False)])
    assert o.accepted_records==0

def test_conflicting_duplicate_fails(grids):
    with pytest.raises(ValueError): packed(grids,[record(value=270),record(value=280)])

def test_partial_report_keeps_other_variables(grids):
    o=packed(grids,[record(),record(observation_id='b',variable='td2m',value=float('nan'))])
    assert o.accepted_records==1 and torch.isfinite(o.features).all()

def test_units_not_guessed(grids):
    assert packed(grids,[record(units='degC')]).accepted_records==0

def test_pressure_links_preserve_weight(grids):
    o=packed(grids,[record(source='radiosonde',variable='temperature',pressure_pa=88000.,quality=.5)])
    assert len(o.weights)==2 and o.weights.sum()==pytest.approx(.5)
    assert set(o.levels.tolist())=={4,5}

def test_no_vertical_extrapolation(grids):
    assert packed(grids,[record(source='radiosonde',variable='temperature',pressure_pa=105000.)]).accepted_records==0

def satellite(**kw):
    r=record(source='meteor_mtvza',variable='synthetic_bt',footprint_km=20.,channel_id='synthetic_channel',
             platform='SYNTHETIC',view_zenith_deg=20.,
             radiometry=dict(instrument='SYNTHETIC_ONLY',quantity='brightness_temperature',units='K',
                             physical_channel_ids=('synthetic_channel',),calibration_id='synthetic',channel_mapping_verified=True))
    r.update(kw); return r

def satellite_vocab():
    return {**DEFAULT_VARIABLES,'synthetic_bt':Variable('K',250.,50.,'column',source='meteor_mtvza',platform='SYNTHETIC',channel_id='synthetic_channel')}

def test_physical_satellite_column_token(grids):
    o=packed(grids,[satellite()],satellite_vocab())
    assert o.accepted_records==1 and o.levels.tolist()==[-1]

def test_raw_counts_not_temperature(grids):
    r=satellite(); r['radiometry']['quantity']='raw_counts'
    assert packed(grids,[r],satellite_vocab()).accepted_records==0

def test_large_microwave_footprint_is_not_point(grids):
    o=packed(grids,[satellite(footprint_km=10000.)],satellite_vocab())
    assert o.rejected=={'footprint_operator_required':1}

def test_72_hour_rollout(grids):
    o=packed(grids); m=net(grids,o)
    with torch.inference_mode(): frames=list(m(o,*static(grids),horizon_hours=72))
    assert [f.lead_hours for f in frames]==list(range(0,73,3))
    assert frames[-1].valid_time==TIME+timedelta(hours=72)
    assert frames[-1].profiles.shape==(42,37,6)
    assert frames[-1].surface.shape==(42,8)
    assert all(torch.isfinite(f.profiles).all() for f in frames)
    assert not frames[0].surface_mask[:,6].any()
    assert (frames[-1].surface[:,1]<=frames[-1].surface[:,0]).all()
    assert (frames[-1].profiles[...,1]>=0).all()

def test_product_mask_never_changes_dynamics(grids):
    o=packed(grids); m=net(grids,o); mask=torch.arange(42)%2==0
    with torch.inference_mode():
        full=list(m(o,*static(grids),horizon_hours=6))
        restricted=list(m(o,*static(grids),horizon_hours=6,product_mask=mask))
    assert torch.equal(full[-1].profiles,restricted[-1].profiles)
    assert not restricted[-1].profile_mask[~mask].any()

def test_completely_empty_sources_are_finite(grids):
    o=packed(grids,[]); m=net(grids,o)
    with torch.inference_mode(): frame=list(m(o,*static(grids),horizon_hours=3))[-1]
    assert torch.isfinite(frame.profiles).all()

def test_input_permutation_invariant(grids):
    rs=[record(observation_id=str(i),value=270.+i) for i in range(5)]
    a,b=packed(grids,rs),packed(grids,rs[::-1]); m=net(grids,a)
    with torch.inference_mode():
        x=m.analyse(a,*static(grids)); y=m.analyse(b,*static(grids))
    assert torch.allclose(x,y,atol=2e-6)

def test_missing_source_does_not_change_scale(grids):
    a=packed(grids); b=packed(grids,[record(),record(observation_id='bad',source='radiosonde',value=np.nan)])
    m=net(grids,a)
    with torch.inference_mode():
        assert torch.equal(m.analyse(a,*static(grids)),m.analyse(b,*static(grids)))

def test_backward_across_rollout(grids):
    rs=[record(),record(observation_id='sonde',source='radiosonde',variable='temperature',pressure_pa=50000.)]
    o=packed(grids,rs); m=net(grids,o)
    frame=list(m(o,*static(grids),horizon_hours=6))[-1]
    loss=(frame.profiles[...,0]/300).square().mean()
    loss.backward()
    grads=[p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads)>10 and all(torch.isfinite(g).all() for g in grads)
    assert m.encoder.value[0].weight.grad.abs().sum()>0

def test_wrong_grid_rejected(grids):
    o=packed(grids); m=net(grids,o); o.grid_fingerprint='wrong'
    with pytest.raises(ValueError): m.analyse(o,*static(grids))

@pytest.mark.parametrize('horizon',[73,-3,5])
def test_invalid_horizon(grids,horizon):
    o=packed(grids); m=net(grids,o)
    with pytest.raises(ValueError): list(m(o,*static(grids),horizon_hours=horizon))

def test_tangent_vector_roundtrip():
    xyz=torch.tensor([[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.]])
    winds=torch.tensor([[0.,3.,4.],[-3.,0.,4.],[0.,-3.,4.]])
    u,v=tangent_components(winds,xyz)
    assert torch.allclose(u,torch.full_like(u,3.)) and torch.allclose(v,torch.full_like(v,4.))

def test_hydrostatic_isothermal_profile():
    p=torch.tensor([100000.,70000.,50000.])
    profile=torch.zeros(1,3,6); profile[...,0]=280.
    profile[...,4]=287.05*280*torch.log(100000./p)
    assert hydrostatic_residual(profile,p).abs().max()<.01

def test_masked_nan_never_becomes_zero_target():
    x=torch.tensor([2.,100.],requires_grad=True)
    loss,n=masked_huber(x,torch.tensor([1.,np.nan]),torch.tensor([True,False]))
    loss.backward()
    assert n==1 and loss.item()==pytest.approx(.5) and x.grad.tolist()==[1.,0.]

def test_empty_mask_loss_zero():
    x=torch.tensor([2.],requires_grad=True)
    loss,n=masked_huber(x,torch.tensor([np.nan]),torch.tensor([False]))
    loss.backward()
    assert loss.item()==0 and n==0 and x.grad.item()==0

def test_weighted_loss_uses_cell_area():
    loss,n=masked_huber(torch.tensor([2.,0.]),torch.zeros(2),torch.ones(2,dtype=torch.bool),torch.tensor([3.,1.]))
    assert loss.item()==pytest.approx(1.125)

def test_bad_prediction_not_silently_dropped():
    with pytest.raises(ValueError): masked_huber(torch.tensor([np.nan]),torch.zeros(1),torch.ones(1,dtype=torch.bool))

def test_jsonl_reports_line(tmp_path):
    p=tmp_path/'bad.jsonl'; p.write_text('{broken}\n')
    with pytest.raises(ValueError,match=':1:'): read_jsonl(p)

def test_demo_writes_explicitly_synthetic_artifact(tmp_path):
    main(['demo','--output',str(tmp_path),'--mesh-level','0','--horizon-hours','6'])
    import json
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['status']=='synthetic_untrained' and report['scientifically_validated'] is False
    with np.load(tmp_path/'synthetic_forecast.npz',allow_pickle=False) as data:
        assert data['lead_hours'].tolist()==[0,3,6]
        assert data['profile_mask'].dtype==bool and np.isnan(data['surface'][0,:,6]).all()

def test_no_real_input_untrained_forecast(tmp_path):
    with pytest.raises(ValueError): main(['demo','--observations','real.jsonl','--output',str(tmp_path)])


def test_normalization_change_is_not_silent(grids):
    original=packed(grids); m=net(grids,original)
    changed={**DEFAULT_VARIABLES,'t2m':Variable('K',250.,100.,'surface')}
    other=packed(grids,variables=changed)
    with pytest.raises(ValueError): m.analyse(other,*static(grids))


def test_pressure_axis_change_is_not_silent(grids):
    original=packed(grids); m=net(grids,original)
    other=pack_observations([record()],grids[0],P[:-1],TIME)
    with pytest.raises(ValueError): m.analyse(other,*static(grids))


def test_observation_coverage_is_source_specific(grids):
    o=packed(grids)
    coverage,age=o.coverage(grids[0].n_cells)
    assert coverage.shape==(42,6) and coverage.sum()==1
    assert age[coverage].item()==pytest.approx(1.)
    assert torch.isinf(age[~coverage]).all()


def test_string_false_is_not_a_valid_flag(grids):
    assert packed(grids,[record(valid='false')]).accepted_records==0


def test_sensor_binding_is_required(grids):
    o=packed(grids,[satellite(platform='ANOTHER_PLATFORM')],satellite_vocab())
    assert o.accepted_records==0


def test_column_variable_requires_physical_channel_binding():
    with pytest.raises(ValueError): Variable('K',250.,50.,'column')


def test_train_validation_windows_do_not_overlap():
    from global_weather.training import assert_time_separation
    with pytest.raises(ValueError): assert_time_separation([TIME],[TIME+timedelta(hours=84)])
    assert_time_separation([TIME],[TIME+timedelta(hours=85)])


def test_supervised_train_step(grids):
    from global_weather.training import Targets,train_step
    o=packed(grids); m=net(grids,o)
    with torch.no_grad(): teacher=list(m(o,*static(grids),horizon_hours=3))[-1]
    profiles=teacher.profiles[None].clone(); profiles[...,0]+=1.
    targets=Targets((3,),profiles,teacher.profile_mask[None,...,None].expand_as(profiles).clone(),
                    teacher.surface[None].clone(),teacher.surface_mask[None].clone(),m.grid_fingerprint,m.pressure_pa)
    before=m.profile_head.weight.detach().clone()
    result=train_step(m,torch.optim.AdamW(m.parameters(),lr=1e-4),o,*static(grids),targets)
    assert result['loss']>0 and not torch.equal(before,m.profile_head.weight)


def test_training_mask_cannot_hide_below_target_terrain(grids):
    from global_weather.training import Targets
    o=packed(grids); m=net(grids,o)
    target=Targets((3,),torch.ones(1,42,37,6),torch.ones(1,42,37,6,dtype=torch.bool),
                   torch.ones(1,42,8),torch.ones(1,42,8,dtype=torch.bool),m.grid_fingerprint,m.pressure_pa)
    with pytest.raises(ValueError,match='below'): target.validate(m)
