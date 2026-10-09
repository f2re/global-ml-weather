"""Analytic vector-basis tests on the sphere, using synthetic fields."""
from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from global_weather.analysis.observation_operator import _prediction,_horizontal
from global_weather.grid import build_grid
from global_weather.observations import utc


@pytest.mark.parametrize('latitude,longitude',[(30,179.5),(89,15),(-50,-179.5)])
def test_vector_h_matches_cartesian_projection_and_legacy_is_unchanged(latitude,longitude):
    grid=build_grid(1); when=utc('2021-01-04T00:00:00Z'); xyz=grid.xyz
    lon=np.arctan2(xyz[:,1],xyz[:,0]);lat=np.arctan2(xyz[:,2],np.hypot(xyz[:,0],xyz[:,1]))
    east=np.column_stack((-np.sin(lon),np.cos(lon),np.zeros(len(lon))))
    north=np.column_stack((-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)))
    vector=np.array([4.,-2.,3.]);u=east@vector;v=north@vector
    profiles=torch.zeros(grid.n_cells,2,6,dtype=torch.float64)
    profiles[...,2]=torch.tensor(u)[:,None];profiles[...,3]=torch.tensor(v)[:,None]
    frame=SimpleNamespace(profiles=profiles,profile_mask=torch.ones(grid.n_cells,2,dtype=torch.bool),
         profile_variable_mask=torch.ones_like(profiles,dtype=torch.bool),valid_time=when,wind_basis='local_enu_vector')
    record=dict(variable='u',units='m s-1',latitude=latitude,longitude=longitude,pressure_pa=70000,observed_at=when.isoformat())
    indices,weights=_horizontal(grid,latitude,longitude,3)
    tangent=u[:,None]*east+v[:,None]*north
    qlon=np.deg2rad(longitude);direction=np.array([-np.sin(qlon),np.cos(qlon),0.])
    expected=np.sum(weights*(tangent[indices]@direction))
    result=_prediction([frame],grid,record,[75000,70000],3)
    assert float(result[0])==pytest.approx(expected,abs=1e-12)
    del frame.wind_basis
    assert float(_prediction([frame],grid,record,[75000,70000],3)[0])==pytest.approx(np.sum(weights*u[indices]))
    frame.wind_basis='local_enu_vector';frame.profile_variable_mask[...,3]=False
    assert _prediction([frame],grid,record,[75000,70000],3) is None
