"""Real public normalization bytes; analytic terrain and integration checks."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from global_weather.import_climatology import (bundled_directory, import_graphcast,
    PINNED_HASHES, source_fit_period, file_sha256)
from global_weather.terrain import sample_grid, combine_surface, verify_dem, prepare


def bundle():
    p=bundled_directory()
    return import_graphcast(p/'mean_by_level.nc',p/'stddev_by_level.nc',expected_hashes=PINNED_HASHES)


def test_actual_upstream_bytes_and_fit_period():
    b=bundle();p=b._payload['provenance']
    assert p['fit_period']=={'start':'1979-01-02T00:00:00+00:00','end':'2015-12-31T23:59:59.999999+00:00'}
    assert p['source_attributes']['date_end']=='2015'
    assert b.get('t2m','K').mean[0]==pytest.approx(278.2418983529725)
    assert b.get('t2m','K').std[0]==pytest.approx(21.40771678509187)
    assert b.get('surface_pressure','Pa').mean[0]==pytest.approx(96604.64498733886)
    assert b.get('total_cloud_fraction','1').std[0]==pytest.approx(.3658826881372015)
    assert len(b.get('temperature','K').mean)==37
    assert bundle().fingerprint==b.fingerprint
    assert 'td2m' not in b.stats


def test_reference_period_does_not_leak_into_2015_test():
    b=bundle()
    with pytest.raises(ValueError):b.assert_independent_test('2015-12-31T12:00:00Z')
    b.assert_independent_test('2016-01-01T00:00:00Z')
    with pytest.raises(ValueError):b.get('precipitation_step','kg m-2').at(interval_hours=3)
    assert b.get('precipitation_step','kg m-2').mean[0]==pytest.approx(.5949484786685348)


def test_period_requires_matching_sources_and_known_format():
    import xarray as xr
    a=xr.Dataset(attrs={'date_start':'1979-01-02','date_end':'2015'})
    b=xr.Dataset(attrs={'date_start':'1979-01-02','date_end':'2017'})
    with pytest.raises(ValueError):source_fit_period(a,b)
    assert source_fit_period(xr.Dataset(),xr.Dataset())[0] is None


def test_tampered_reference_file_rejected(tmp_path):
    p=bundled_directory();broken=tmp_path/'mean.nc';broken.write_bytes((p/'mean_by_level.nc').read_bytes()+b'x')
    with pytest.raises(ValueError):import_graphcast(broken,p/'stddev_by_level.nc',expected_hashes=PINNED_HASHES)


def test_surface_mask_does_not_use_elevation_sign():
    dem=np.array([-400.,-5000.,1200.,-120.]);ref=np.array([-380.,0.,700.,100.]);land=np.array([1.,0.,.5,0.])
    height,mask=combine_surface(dem,ref,land)
    assert height.tolist()==[-400.,0.,700.,100.]
    assert mask.tolist()==[True,False,False,False]


def test_dem_scale_applied_once_and_dateline_periodic():
    values=np.arange(32,dtype=np.int16).reshape(4,8)
    result=sample_grid(values,[-90.,90.,0.,0.],[-180.,180.,180.,-180.],scale=.5,offset=0,fill=-32768,block_shape=(2,2))
    assert result.tolist()==[0.,12.,8.,8.]
    with pytest.raises(ValueError):sample_grid(values,[91.],[0.],scale=.5,offset=0,fill=-32768,block_shape=(2,2))
    values[0,0]=-32768
    with pytest.raises(ValueError):sample_grid(values,[-90.],[-180.],scale=.5,offset=0,fill=-32768,block_shape=(2,2))


def test_dem_file_bound_is_checked_before_hdf5(tmp_path):
    f=tmp_path/'bad';f.write_bytes(b'bad')
    m={'schema':'global-dem-source-1','registration':'pixel','bytes':900000001,'sha256':file_sha256(f)}
    with pytest.raises(ValueError):verify_dem(f,m)


def test_dem_preparation_reaches_all_three_models(tmp_path,monkeypatch):
    from global_weather.pipeline.io import write_arrays,read_arrays
    from global_weather.grid import build_pyramid
    from global_weather.multimodal.fixture import fixture
    from global_weather.multimodal.model import MultimodalWeatherModel
    from global_weather.model import GlobalWeatherModel
    from global_weather.adaptive import AdaptiveWeatherModel
    grids=build_pyramid(0);n=grids[0].n_cells
    initial=tmp_path/'static.npz'
    write_arrays(initial,elevation_m=np.zeros(n,dtype=np.float32),land_fraction=np.ones(n,dtype=np.float32),
                 surface_units=np.array(['m','1']),grid_fingerprint=np.array(grids[0].fingerprint))
    manifest=tmp_path/'dem.json';manifest.write_text(json.dumps({'sha256':'a'*64,'bytes':896166188}))
    monkeypatch.setattr('global_weather.terrain.dem_samples',lambda *a:np.linspace(-100.,3000.,n))
    out=tmp_path/'terrain.npz';prepare('not-read',manifest,initial,out,mesh_level=0,confirm_landmask=True)
    fields=read_arrays(out);e=torch.tensor(fields['elevation_m']);land=torch.ones(n)
    sensors,observations=fixture(grids)
    torch.set_num_threads(1);torch.manual_seed(77)
    common=dict(observation_schema=observations.schema_fingerprint,hidden=16)
    models=[GlobalWeatherModel(grids,observations.vocabulary,**common),
            AdaptiveWeatherModel(grids,observations.vocabulary,allow_unscaled_synthetic=True,**common),
            MultimodalWeatherModel(grids,observations.vocabulary,sensors=sensors,sensor_signature=observations.sensor_signature,
                base_channels=8,allow_unscaled_synthetic=True,**common)]
    for model in models:
        with torch.no_grad():
            before=model.analyse(observations,torch.zeros(n),land)
            after=model.analyse(observations,e,land)
        assert not torch.allclose(before,after,atol=1e-7)
    assert fields['dem_applied_mask'].all()


def test_actual_norms_can_be_completed_by_training_only(tmp_path):
    from global_weather.pipeline.fixture import create_fixture
    from global_weather.pipeline.fit import fit_normalization
    from global_weather.pipeline.io import read_json,atomic_json
    from global_weather.pipeline.dataset import PreparedDataset
    from global_weather.pipeline.runner import config_from_json,make_model
    ds_path=create_fixture(tmp_path/'data',horizon_hours=3)
    m=read_json(ds_path);m['normalization']=None
    source=ds_path.with_name('unscaled-ref.json');atomic_json(source,m)
    b=bundle();base=tmp_path/'base.json';b.save(base)
    target=ds_path.with_name('with-reference.json');fit_normalization(source,target,base_path=base)
    ds=PreparedDataset(target);model=make_model(ds,config_from_json({'hidden':16,'horizon_hours':3}))
    assert ds.norm.get('t2m','K').mean==b.get('t2m','K').mean
    assert ds.norm.get('td2m','K').std[0]>0
    assert ds.norm.get('precipitation_step','kg m-2').interval_hours==3
    i=model.vocabulary.index('t2m');obs=ds.packed(ds.samples[0]);assert (obs.variables==i).any()
    assert torch.isfinite(model.profile_mean).all()
    with torch.no_grad():
        frames=list(model(obs,torch.tensor(ds.elevation),torch.tensor(ds.land),horizon_hours=3))
    assert torch.isfinite(frames[-1].profiles).all()
