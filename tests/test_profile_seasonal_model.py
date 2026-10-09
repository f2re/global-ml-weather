"""Synthetic seasonal-model structure; real pinned GraphCast scales, no skill claim."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from global_weather.grid import build_grid
from global_weather.profile_graphcast_normalization import GraphCastNormalization, _reference
from global_weather.profile_model_v2 import PressureProfileModel
from global_weather.profile_seasonal_model import SeasonalProfileModel
from global_weather.profile_training import VARIABLES, START
from global_weather.profile_training_v2 import objective
from global_weather.vertical import PROFILE_UNITS


def normalization():
    return GraphCastNormalization({**_reference(), 'source_identity': {
        key: '0' * 64 for key in ('database_sha256', 'dataset_manifest_sha256',
                                'source_sha256', 'admission_sha256')}})


class ClimateFixture:
    """Explicit synthetic physical mean and support, independent of measurements."""
    def __init__(self, grid, norms):
        self.mean=np.broadcast_to(norms.mean+1.5*norms.std,(grid.n_cells,37,5)).copy()
        self.support=np.ones(self.mean.shape,dtype=bool)
        self.support[:,norms.pressure_pa<30000,1]=False
        self.support[:,norms.pressure_pa<1000,:]=False
        self.mean[~self.support]=np.nan
        self.times=[]

    def sample(self, valid_time):
        self.times.append(valid_time)
        return self.mean.copy(),self.support.copy()


def model_fixture():
    torch.manual_seed(29)
    grid=build_grid(0);norms=normalization();climate=ClimateFixture(grid,norms)
    model=SeasonalProfileModel(grid,norms,climate,8)
    return model,climate


def rows(when):
    return [dict(variable=name,value=[270.,.001,4.,-3.,30000.][i],units=PROFILE_UNITS[i],
                 latitude=10.,longitude=20.,pressure_pa=70000.,observed_at=when.isoformat(),
                 available_at=when.isoformat(),profile_id='synthetic') for i,name in enumerate(VARIABLES)]


def test_masked_context_has_explicit_masks_and_finite_neutral_features():
    model,climate=model_fixture()
    # Invalid context never changes canonical GraphCast support or invents observations.
    climate.mean[~climate.support]=np.inf
    features=model._climate_features(START)
    support=torch.tensor(climate.support)
    assert features.shape==(12,37,14) and torch.isfinite(features).all()
    assert torch.all(features[...,:5][~support]==0)
    assert torch.allclose(features[...,:5][support],torch.full_like(features[...,:5][support],1.5))
    assert torch.equal(features[...,5:10].bool(),support)
    assert model.norm_support.all()
    climate.mean[0,0,0]=np.inf
    with pytest.raises(ValueError,match='Nonfinite supported'):
        model._climate_features(START)


def test_calendar_annual_phase_leap_year_and_longitude_local_hour():
    model,_=model_fixture()
    begin=datetime(2024,1,1,tzinfo=timezone.utc)
    halfway=begin+timedelta(days=183)
    january=model._climate_features(begin)
    july=model._climate_features(halfway)
    assert torch.allclose(january[...,10],torch.zeros_like(january[...,10]),atol=1e-7)
    assert torch.allclose(january[...,11],torch.ones_like(january[...,11]))
    assert torch.allclose(july[...,10],torch.zeros_like(july[...,10]),atol=1e-7)
    assert torch.allclose(july[...,11],-torch.ones_like(july[...,11]))
    six_hours=model._climate_features(begin+timedelta(hours=6))
    assert torch.allclose(six_hours[...,12],january[...,13],atol=1e-6)
    assert torch.allclose(six_hours[...,13],-january[...,12],atol=1e-6)
    longitude=np.arctan2(model.grid.xyz[:,1],model.grid.xyz[:,0])
    assert np.allclose(january[:,0,12].numpy(),np.sin(longitude),atol=1e-6)
    assert np.allclose(january[:,0,13].numpy(),np.cos(longitude),atol=1e-6)


def test_future_or_unavailable_observations_cannot_change_started_forecast():
    model,climate=model_fixture();model.eval()
    issue=START+timedelta(days=3)
    inputs=rows(issue-timedelta(hours=6))
    future=rows(issue+timedelta(hours=1))
    unavailable=[dict(row,available_at=(issue+timedelta(hours=1)).isoformat()) for row in rows(issue-timedelta(hours=3))]
    with torch.no_grad():
        clean=model(inputs,issue)
        climate.times.clear()
        polluted=model(inputs+future+unavailable,issue)
    assert climate.times==[issue+timedelta(hours=lead) for lead in range(0,73,3)]
    for first,second in zip(clean,polluted):
        assert torch.equal(first.profiles[...,:5],second.profiles[...,:5])
        assert torch.equal(first.profile_variable_mask,second.profile_variable_mask)
    assert len(clean)==25 and clean[-1].lead_hours==72
    assert torch.isnan(clean[-1].profiles[...,5]).all()
    assert not clean[-1].surface_mask.any()


def test_observed_target_objective_reaches_every_model_and_new_climate_branch():
    model,_=model_fixture();issue=START+timedelta(days=3)
    frames=model(rows(issue-timedelta(hours=6)),issue)
    loss,counts=objective(model,frames,rows(issue+timedelta(hours=12)))
    assert counts==[1]*5 and torch.isfinite(loss)
    loss.backward()
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None,name
        assert torch.isfinite(parameter.grad).all(),name
        assert parameter.grad.abs().sum()>0,name
    for name,parameter in model.climate_encoder.named_parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0,name
    assert (model.head.weight.grad.abs().sum(dim=1)>0).all()
    assert (model.head.bias.grad.abs()>0).all()


def test_climate_cannot_replace_graphcast_output_inverse_or_mask():
    seasonal,climate=model_fixture()
    base=PressureProfileModel(seasonal.grid,seasonal.normalization,8)
    base.head.load_state_dict(seasonal.head.state_dict())
    state=torch.randn(12,37,8)
    a=base._decode(state,START,0);b=seasonal._decode(state,START,0)
    assert torch.equal(a.profiles[...,:5],b.profiles[...,:5])
    assert torch.equal(a.profile_variable_mask,b.profile_variable_mask)
    # A zero standardized output decodes exactly to canonical means, not NOAA means.
    with torch.no_grad():seasonal.head.weight.zero_();seasonal.head.bias.zero_()
    decoded=seasonal._decode(state,START,0).profiles[...,:5]
    expected=seasonal.mean[None].expand(12,-1,-1).clone();expected[...,1].clamp_min_(0.)
    assert torch.equal(decoded,expected)
    climate.mean[climate.support]+=1000.
    again=seasonal._decode(state,START,0).profiles[...,:5]
    assert torch.equal(decoded,again)
    assert not np.array_equal(climate.support,seasonal.norm_support[None].expand(12,-1,-1).numpy())
