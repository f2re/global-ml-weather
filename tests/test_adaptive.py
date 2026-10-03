"""All numeric fixtures here are synthetic, not an ERA5 weather evaluation."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from dataclasses import replace
import numpy as np
import pytest
import torch
from global_weather.grid import build_pyramid
from global_weather.observations import pack_observations, DEFAULT_VARIABLES, Variable
from global_weather.model import GlobalWeatherModel
from global_weather.adaptive import AdaptiveWeatherModel, DirectedGraph, AdaptiveBlock
from global_weather.normalization import ZStat, NormalizationBundle, weighted_statistics
from global_weather.physics import conservative_exchange, hydrostatic_penalty
from global_weather.vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS
from global_weather.checkpoints import save_checkpoint, load_checkpoint
from global_weather.import_climatology import import_graphcast, VARIABLES, file_sha256

torch.set_num_threads(1)
T = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
P = np.array(PRESSURE_HPA)*100

@pytest.fixture(scope='module')
def grids(): return build_pyramid(1)

def record(**kwargs):
    data = dict(observation_id='one', source='station', variable='t2m', value=280., units='K',
                latitude=60., longitude=30., observed_at=(T-timedelta(hours=1)).isoformat(),
                available_at=T.isoformat(), valid=True)
    data.update(kwargs)
    return data

def obs(grids, records=None, *, time=T, normalization=None, variables=None):
    return pack_observations([record()] if records is None else records, grids[0], P, time,
                              variables, normalization=normalization)

def model(grids, observations, **kwargs):
    torch.manual_seed(8)
    return AdaptiveWeatherModel(grids, observations.vocabulary,
                                observation_schema=observations.schema_fingerprint,
                                hidden=16, allow_unscaled_synthetic=True, **kwargs)

def static(grids): return (torch.zeros(grids[0].n_cells),)*2

def bundle_payload():
    variables = {}
    means, stds = (270., .003, 0., 0., 30000., 0.), (20., .002, 10., 10., 10000., .5)
    for name, unit, mean, std in zip(PROFILE_VARIABLES, PROFILE_UNITS, means, stds):
        variables[name] = dict(units=unit, mean=[mean]*37, std=[std]*37, pressure_pa=P.tolist())
    means, stds = (280., 275., 0., 0., 100000., 101000., 2., .5), (20., 20., 10., 10., 10000., 10000., 1., .2)
    for name, unit, mean, std in zip(SURFACE_VARIABLES, SURFACE_UNITS, means, stds):
        variables[name] = dict(units=unit, mean=[mean], std=[std],
                               interval_hours=3 if name == 'precipitation_step' else None)
    return dict(schema_version=1, kind='global_level_zscore', variables=variables,
                provenance=dict(repository='SYNTHETIC-TEST-FIXTURE', revision='test-v1',
                                data_family='synthetic, not ERA5', license='test fixture',
                                artifact_sha256={'fixture': hashlib.sha256(b'synthetic fixture').hexdigest()},
                                fit_period={'end': '2020-12-31T23:59:59Z'}))

@pytest.mark.parametrize('slots', [4,8,16,38])
def test_compressed_output_still_has_all_profiles(grids, slots):
    o = obs(grids); m = model(grids, o, latent_slots=slots)
    with torch.inference_mode():
        state = m.analyse(o, *static(grids))
        frames = list(m(o, *static(grids), horizon_hours=72))
    assert state.shape == (42, slots, 16)
    assert len(frames) == 25 and frames[-1].profiles.shape == (42,37,6)
    assert all(torch.isfinite(f.profiles).all() for f in frames)

def test_channel_identity_and_value_are_not_separable(grids):
    variables = {name: Variable('K', 250., 50., 'column', source='electro_l',
                               platform='SYNTHETIC', channel_id=name) for name in ('a','b')}
    def satellite(name, value):
        return record(observation_id=name, source='electro_l', variable=name, value=value,
                       platform='SYNTHETIC', channel_id=name, footprint_km=4., view_zenith_deg=20.,
                       radiometry=dict(instrument='SYNTHETIC', quantity='brightness_temperature', units='K',
                                       physical_channel_ids=['a','b'], calibration_id='SYNTHETIC', channel_mapping_verified=True))
    a = obs(grids, [satellite('a',230),satellite('b',280)], variables=variables)
    b = obs(grids, [satellite('a',280),satellite('b',230)], variables=variables)
    assert a.accepted_records == b.accepted_records == 2
    torch.manual_seed(7)
    m = GlobalWeatherModel(grids,a.vocabulary,observation_schema=a.schema_fingerprint,hidden=16)
    with torch.no_grad(): difference=(m.analyse(a,*static(grids))-m.analyse(b,*static(grids))).abs().max()
    assert difference > 1e-4

def test_adaptive_exchange_responds_to_wind(grids):
    torch.manual_seed(4)
    block = AdaptiveBlock(16); graph = DirectedGraph(grids[0])
    x = torch.randn(42,8,16); context=torch.zeros(42,8)
    wind=torch.randn(42,3)*30
    a=block(x,graph,context,wind,3); b=block(x,graph,context,-wind,3)
    assert (a-b).abs().max()>1e-5

def test_regime_changes_with_physical_context(grids):
    o=obs(grids); m=model(grids,o)
    x=torch.randn(42,8,16); c=torch.zeros(42,8); wind=torch.zeros(42,3)
    a=m.processor(x,c,wind,3); c[:,3]=2.; b=m.processor(x,c,wind,3)
    assert (a-b).abs().max()>1e-5

def test_roi_does_not_change_global_state(grids):
    o=obs(grids);m=model(grids,o); mask=torch.arange(42)%2==0
    with torch.no_grad():
        a=list(m(o,*static(grids),horizon_hours=6))[-1]
        b=list(m(o,*static(grids),horizon_hours=6,product_mask=mask))[-1]
    assert torch.equal(a.profiles,b.profiles) and not b.profile_mask[~mask].any()

def test_repeated_evidence_is_not_assimilated_twice(grids):
    o=obs(grids);m=model(grids,o)
    a=m.analysis_state(o,*static(grids))
    b=m.analysis_state(o,*static(grids),background=a)
    assert torch.equal(a.latent,b.latent) and a.evidence==b.evidence

def test_late_new_observation_can_update_background(grids):
    o=obs(grids);m=model(grids,o)
    a=m.analysis_state(o,*static(grids))
    new=obs(grids,[record(),record(observation_id='late',value=290.,observed_at=(T-timedelta(hours=5)).isoformat())])
    b=m.analysis_state(new,*static(grids),background=a)
    assert not torch.allclose(a.latent,b.latent) and len(b.evidence)==2

def test_background_requires_exact_valid_time(grids):
    o=obs(grids);m=model(grids,o);a=m.analysis_state(o,*static(grids))
    with pytest.raises(ValueError): m.analysis_state(o,*static(grids),background=replace(a,valid_time=T-timedelta(hours=3)))

def test_background_advance_matches_trained_cadence(grids):
    o=obs(grids);m=model(grids,o);a=m.analysis_state(o,*static(grids))
    with pytest.raises(ValueError): m.advance_background(a,T+timedelta(hours=1),*static(grids))
    b=m.advance_background(a,T+timedelta(hours=3),*static(grids))
    next_obs=obs(grids,[record()],time=T+timedelta(hours=3))
    c=m.analysis_state(next_obs,*static(grids),background=b)
    assert torch.equal(c.latent,b.latent)

def test_empty_inputs_are_explicitly_synthetic(grids):
    o=obs(grids,[])
    with pytest.raises(ValueError):
        AdaptiveWeatherModel(grids,o.vocabulary,observation_schema=o.schema_fingerprint)
    m=model(grids,o)
    with torch.no_grad(): assert torch.isfinite(m.analyse(o,*static(grids))).all()

def test_gradients_across_adaptive_rollout(grids):
    o=obs(grids);m=model(grids,o)
    frame=list(m(o,*static(grids),horizon_hours=6))[-1]
    (frame.profiles[...,0]/300).square().mean().backward()
    gradients=[p.grad for p in m.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients)
    assert m.processor.down[0].regime.weight.grad.abs().sum()>0

@pytest.mark.parametrize('std',[(0.,),(-1.,),(float('nan'),)])
def test_bad_sigma_rejected(std):
    with pytest.raises(ValueError): ZStat('K',(270.,),std)

def test_level_interpolation_and_no_extrapolation():
    s=ZStat('K',(280.,250.),(20.,10.),(100000.,50000.))
    mu,sd=s.at(np.sqrt(100000.*50000.)); assert mu==pytest.approx(265.) and sd==pytest.approx(15.)
    with pytest.raises(ValueError):s.at(100.)

def test_six_hour_statistics_cannot_normalize_three_hours():
    s=ZStat('kg m-2',(1.,),(2.,),interval_hours=6)
    with pytest.raises(ValueError):s.at(interval_hours=3)

def test_bundle_roundtrip_and_fingerprint(tmp_path):
    b=NormalizationBundle(bundle_payload());b.save(tmp_path/'norm.json')
    c=NormalizationBundle.load(tmp_path/'norm.json');assert c.fingerprint==b.fingerprint
    z=b.normalise('t2m',300.,'K');assert z==pytest.approx(1.)
    assert b.unnormalise('t2m',z,'K')==pytest.approx(300.)

def test_unknown_normalization_is_not_a_default_scale():
    b=NormalizationBundle(bundle_payload())
    with pytest.raises(ValueError): b.normalise('SYNTHETIC_IR',260.,'K')
    with pytest.raises(ValueError): b.normalise('t2m',10.,'degC')

def test_source_hashes_required():
    p=bundle_payload();p['provenance']['artifact_sha256']={}
    with pytest.raises(ValueError):NormalizationBundle(p)

def test_unknown_fit_period_fails_independence_check():
    p=bundle_payload();p['provenance']['fit_period']=None;b=NormalizationBundle(p)
    with pytest.raises(ValueError):b.assert_independent_test('2026-01-01T00:00:00Z')

def test_train_test_period_guard():
    b=NormalizationBundle(bundle_payload())
    b.assert_independent_test('2021-01-01T00:00:00Z')
    with pytest.raises(ValueError): b.assert_independent_test('2020-01-01T00:00:00Z')

def test_area_and_missingness_weighted_fit():
    mean,std=weighted_statistics(np.array([1.,3.,np.nan]),np.array([True,True,False]),np.array([1.,3.,9.]))
    assert mean==pytest.approx(2.5) and std==pytest.approx(np.sqrt(.75))
    with pytest.raises(ValueError):weighted_statistics([1.,1.],np.array([True,True]),[1.,1.])

def test_normalization_used_in_observations_and_decoder(grids):
    bundle=NormalizationBundle(bundle_payload())
    o=obs(grids,normalization=bundle)
    assert o.features[0,0]==0 and o.normalization_fingerprint==bundle.fingerprint
    m=model(grids,o,normalization=bundle)
    with torch.no_grad(): f=list(m(o,*static(grids),horizon_hours=3))[-1]
    assert f.profiles.shape==(42,37,6) and torch.isfinite(f.profiles).all()
    assert (f.surface[:,1]<=f.surface[:,0]).all()
    assert (f.profiles[...,1]>=0).all()

def test_checkpoint_validates_before_changing_weights(grids,tmp_path):
    o=obs(grids);a=model(grids,o);b=model(grids,o,latent_slots=4)
    save_checkpoint(tmp_path/'weights.pt',a)
    old=b.profile_head.weight.detach().clone()
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'weights.pt',b)
    assert torch.equal(old,b.profile_head.weight)
    load_checkpoint(tmp_path/'weights.pt',a)

def test_normalization_change_rejects_checkpoint(grids,tmp_path):
    a=NormalizationBundle(bundle_payload());o=obs(grids,normalization=a);m=model(grids,o,normalization=a)
    save_checkpoint(tmp_path/'weights.pt',m)
    data=bundle_payload();data['variables']['t2m']['std']=[21.]
    b=NormalizationBundle(data);ob=obs(grids,normalization=b);mb=model(grids,ob,normalization=b)
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'weights.pt',mb)

def test_conservative_exchange_uses_actual_cell_areas(grids):
    g=grids[0];pairs=torch.tensor(g.edges[:,g.edges[0]<g.edges[1]],dtype=torch.long)
    flux=torch.randn(pairs.shape[1],3,dtype=torch.float64,requires_grad=True)
    area=torch.tensor(g.areas_m2)
    tendency=conservative_exchange(flux,pairs,area)
    assert (area[:,None]*tendency).sum(0).abs().max()<1e-12
    tendency.square().sum().backward();assert torch.isfinite(flux.grad).all()

def test_duplicate_flux_edges_rejected():
    with pytest.raises(ValueError):conservative_exchange(torch.ones(2),torch.tensor([[0,1],[1,0]]),torch.ones(2))

def test_hydrostatic_penalty_has_correct_mask():
    p=torch.tensor([100000.,70000.,50000.]);x=torch.zeros(2,3,6)
    x[...,0]=280.;x[...,4]=287.05*280.*torch.log(100000./p)
    mask=torch.ones_like(x,dtype=torch.bool)
    assert hydrostatic_penalty(x,p,mask)<1e-12
    x[:,1,4]+=10000.;assert hydrostatic_penalty(x,p,mask)>0.01
    mask[:]=False;assert hydrostatic_penalty(x,p,mask)==0


def graphcast_fixture(tmp_path):
    xr=pytest.importorskip('xarray')
    means,stds={},{}
    for i,(original,(_,units,_,interval)) in enumerate(VARIABLES.items()):
        attrs={'units':'m' if interval else units}
        if i<6:
            means[original]=xr.DataArray(np.full(37,10.+i),dims=['level'],coords={'level':list(reversed(PRESSURE_HPA))},attrs=attrs)
            stds[original]=xr.DataArray(np.full(37,1.+i),dims=['level'],coords={'level':list(PRESSURE_HPA)},attrs=attrs)
        else:
            means[original]=xr.DataArray(1.,attrs=attrs);stds[original]=xr.DataArray(2.,attrs=attrs)
    mean_path=tmp_path/'mean.nc';std_path=tmp_path/'std.nc'
    xr.Dataset(means).to_netcdf(mean_path,engine='scipy');xr.Dataset(stds).to_netcdf(std_path,engine='scipy')
    return mean_path,std_path

def test_graphcast_import_is_level_aware_and_records_hashes(tmp_path):
    mean,std=graphcast_fixture(tmp_path);b=import_graphcast(mean,std)
    assert b.get('temperature','K').pressure_pa==tuple(P)
    assert b.get('precipitation_step','kg m-2').mean==(1000.,)
    assert b.get('precipitation_step','kg m-2').interval_hours==6
    assert b._payload['provenance']['artifact_sha256']['mean_by_level.nc']==file_sha256(mean)
    assert 'surface_pressure' not in b.stats and 'td2m' not in b.stats

def test_graphcast_import_hash_mismatch_is_error(tmp_path):
    mean,std=graphcast_fixture(tmp_path)
    with pytest.raises(ValueError):import_graphcast(mean,std,expected_hashes={'bad':'bad'})

def test_missing_input_statistics_fail_before_partial_admission(grids):
    data=bundle_payload();del data['variables']['surface_pressure']
    with pytest.raises(ValueError):obs(grids,normalization=NormalizationBundle(data))

def test_checkpoint_cannot_change_frozen_statistics(grids,tmp_path):
    bundle=NormalizationBundle(bundle_payload());o=obs(grids,normalization=bundle);m=model(grids,o,normalization=bundle)
    path=tmp_path/'weights.pt';save_checkpoint(path,m)
    bad=torch.load(path,weights_only=True);bad['state_dict']['profile_std']*=2;torch.save(bad,path)
    old=m.profile_head.weight.detach().clone()
    with pytest.raises(ValueError):load_checkpoint(path,m)
    assert torch.equal(old,m.profile_head.weight)

def test_physics_loss_can_be_used_in_real_training_interface(grids):
    from global_weather.training import Targets,train_step
    b=NormalizationBundle(bundle_payload());o=obs(grids,normalization=b);m=model(grids,o,normalization=b)
    with torch.no_grad():f=list(m(o,*static(grids),horizon_hours=3))[-1]
    p=f.profiles[None].clone();p[...,0]+=1
    targets=Targets((3,),p,f.profile_mask[None,...,None].expand_as(p).clone(),
                    f.surface[None].clone(),f.surface_mask[None].clone(),m.grid_fingerprint,m.pressure_pa)
    result=train_step(m,torch.optim.AdamW(m.parameters(),lr=1e-4),o,*static(grids),targets,physics_weight=.01)
    assert result['loss']>0 and np.isfinite(result['gradient_norm'])
