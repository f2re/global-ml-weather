"""Synthetic climate fixtures test conversions and operators, not weather accuracy."""
from datetime import datetime, timezone
import json

import numpy as np
import pytest
import xarray as xr

from global_weather import seasonal_climatology as climate
from global_weather.grid import build_grid, unit_xyz
from global_weather.observation_training import digest


def sources(root):
    root.mkdir()
    latitude=np.array([90.,0.,-90.]); longitude=np.array([0.,90.,180.,270.])
    time=np.array([(datetime(2001,month,1)-datetime(2001,1,1)).days for month in range(1,13)])
    for name,filename,unit in zip(climate.NAMES,climate.FILES,climate.UNITS):
        levels=np.array(climate.NATIVE_LEVELS[:8] if name=='shum' else climate.NATIVE_LEVELS)
        shape=(12,len(levels),3,4)
        value={'air':0.,'shum':5.,'uwnd':1.,'vwnd':2.,'hgt':1000.}[name]
        data=xr.Dataset({name:(('time','level','lat','lon'),np.full(shape,value)),
                         'valid_yr_count':(('time','level','lat','lon'),np.full(shape,30,dtype=np.int16))},
                        coords={'time':time,'level':levels,'lat':latitude,'lon':longitude})
        data[name].attrs={'units':unit,'statistic':'Long Term Mean','parent_stat':'Mean'}
        data.time.attrs={'units':'days since 2001-01-01','climo_period':climate.PERIOD}
        data.level.attrs={'units':'millibar','positive':'down'}
        data.lat.attrs={'units':'degrees_north'};data.lon.attrs={'units':'degrees_east'}
        data.to_netcdf(root/filename)
    return root


def mutate(path,function):
    with xr.open_dataset(path,decode_times=False) as data:modified=data.load()
    function(modified)
    modified.to_netcdf(path,mode='w')


def test_prepare_physical_conversions_coverage_immutable_and_source_drift(tmp_path):
    source=sources(tmp_path/'sources');output=tmp_path/'context'
    manifest=climate.prepare(source,output,mesh_level=0)
    assert manifest['period']=={'start':'1991-01-01','end':'2020-12-31'}
    assert climate.prepare(source,output,mesh_level=0)==manifest
    context=climate.SeasonalClimatology(output,build_grid(0))
    supported=context.support
    assert np.allclose(context.mean[...,0][supported[...,0]],273.15)
    assert np.allclose(context.mean[...,1][supported[...,1]],.005)
    assert np.allclose(context.mean[...,4][supported[...,4]],9806.65)
    assert not supported[:,:,climate.PRESSURE_PA<30000,1].any()
    assert not supported[:,:,climate.PRESSURE_PA<1000][:,:,:,[0,2,3,4]].any()
    assert np.isnan(context.mean[~supported]).all()
    assert not context.mean.flags.writeable and not context.support.flags.writeable
    modified=context.payload;modified['minimum_valid_years']=1
    assert context.payload['minimum_valid_years']==25
    with (source/climate.FILES[0]).open('ab') as file:file.write(b'changed')
    with pytest.raises(ValueError,match='original source changed'):context.verify_sources()


@pytest.mark.parametrize('case',['std','units','period','counts','months'])
def test_reject_nonmean_wrongunits_wrongperiod_counts_and_months(tmp_path,case):
    source=sources(tmp_path/'sources')
    def change(dataset):
        if case=='std':dataset.air.attrs['statistic']='Standard Deviation'
        elif case=='units':dataset.air.attrs['units']='K'
        elif case=='period':dataset.time.attrs['climo_period']='1981/01/01 - 2010/12/31'
        elif case=='counts':dataset.valid_yr_count.values[0,0,0,0]=31
        else:dataset.coords['time']=('time',dataset.time.values[::-1].copy(),dict(dataset.time.attrs))
    mutate(source/climate.FILES[0],change)
    with pytest.raises(ValueError):climate.prepare(source,tmp_path/'context',mesh_level=0)
    assert not (tmp_path/'context').exists()


def test_year_count_support_is_strict_and_no_vertical_gap_bridging(tmp_path):
    source=sources(tmp_path/'sources')
    def change(dataset):dataset.valid_yr_count.values[:,3,:,:]=24
    mutate(source/climate.FILES[0],change)
    climate.prepare(source,tmp_path/'context',mesh_level=0)
    context=climate.SeasonalClimatology(tmp_path/'context')
    # 700 hPa missing: exact 700 and brackets 650/750 remain unsupported.
    for pressure in (70000,65000,75000):
        level=int(np.flatnonzero(climate.PRESSURE_PA==pressure)[0])
        assert not context.support[:,:,level,0].any()
    with pytest.raises(ValueError,match='25,30'):
        climate.prepare(source,tmp_path/'loose',mesh_level=0,minimum_years=24)


def test_periodic_bilinear_scalar_constant_linear_and_corner_mask():
    lat=np.array([-90.,0.,90.]);lon=np.array([0.,90.,180.,270.])
    values=np.broadcast_to(lat[:,None],(3,4)).copy();support=np.ones((3,4),bool)
    result,valid=climate._horizontal(values,support,lat,lon,np.array([30.,0.,90.]),np.array([20.,359.,-180.]))
    assert valid.all() and np.allclose(result,[30.,0.,90.])
    constant=np.full((3,4),7.)
    assert np.allclose(climate._horizontal(constant,support,lat,lon,np.array([0.,80.]),np.array([359.,-179.]))[0],7.)
    support[1,0]=False
    assert not climate._horizontal(constant,support,lat,lon,np.array([0.]),np.array([359.]))[1][0]


def test_paired_wind_cartesian_projection_at_periodic_seam(tmp_path,monkeypatch):
    source=sources(tmp_path/'sources')
    for name,index in (('uwnd',2),('vwnd',3)):
        def field(dataset,name=name):
            lat=np.deg2rad(dataset.lat.values)
            values=np.cos(lat)[:,None] if name=='uwnd' else np.zeros((3,1))
            dataset[name].values[:]=values[None,None]
        mutate(source/climate.FILES[index],field)
    class Grid:
        level=0;n_cells=2;fingerprint='synthetic-seam-grid'
        xyz=unit_xyz(np.array([0.,90.]),np.array([315.,0.]))
    monkeypatch.setattr(climate,'build_grid',lambda level:Grid())
    climate.prepare(source,tmp_path/'context',mesh_level=0)
    context=climate.SeasonalClimatology(tmp_path/'context',Grid())
    level=int(np.flatnonzero(climate.PRESSURE_PA==70000)[0])
    assert context.mean[0,0,level,2]==pytest.approx(2**-.5)
    assert context.mean[0,0,level,3]==pytest.approx(0.,abs=1e-12)
    assert abs(context.mean[0,1,level,2])<1e-12
    # One missing v corner invalidates both paired model wind components.
    mutate(source/climate.FILES[3],lambda dataset:dataset.valid_yr_count.values.__setitem__((slice(None),slice(None),1,0),24))
    climate.prepare(source,tmp_path/'paired-missing',mesh_level=0)
    missing=climate.SeasonalClimatology(tmp_path/'paired-missing',Grid())
    assert not missing.support[:,0,:,2:4].any()


def test_calendar_month_centres_leap_year_and_december_january_continuity(tmp_path):
    source=sources(tmp_path/'sources')
    mutate(source/climate.FILES[0],lambda dataset:dataset.air.values.__setitem__(slice(None),np.arange(1,13)[:,None,None,None]))
    climate.prepare(source,tmp_path/'context',mesh_level=0)
    context=climate.SeasonalClimatology(tmp_path/'context')
    for year in (2023,2024):
        centre=context._centre(year,2)
        assert centre.day==(15 if year==2024 else 15)
        assert centre.hour==(12 if year==2024 else 0)
        field,support=context.sample(centre)
        assert np.allclose(field[...,0][support[...,0]],275.15)
    left=context._centre(2023,12);right=context._centre(2024,1)
    midpoint=left+(right-left)/2
    field,support=context.sample(midpoint)
    assert np.allclose(field[...,0][support[...,0]],273.15+6.5)
    a=context.sample('2023-12-31T23:59:59+00:00')[0][...,0]
    b=context.sample('2024-01-01T00:00:01+00:00')[0][...,0]
    assert np.nanmax(np.abs(a-b))<1e-4


@pytest.mark.parametrize('what',['artifact','manifest','grid','semantic'])
def test_prepared_artifact_tampering_and_geometry_refused(tmp_path,what):
    source=sources(tmp_path/'sources');output=tmp_path/'context'
    climate.prepare(source,output,mesh_level=0)
    context=climate.SeasonalClimatology(output)
    if what=='artifact':
        with (output/'seasonal-climatology.npz').open('ab') as file:file.write(b'tamper')
        with pytest.raises(ValueError,match='prepared artifact changed'):context.verify_sources()
    elif what=='manifest':
        with (output/'manifest.json').open('a') as file:file.write('\n')
        with pytest.raises(ValueError,match='manifest changed'):context.verify_sources()
    elif what=='grid':
        with pytest.raises(ValueError,match='grid differs'):climate.SeasonalClimatology(output,build_grid(1))
    else:
        metadata=json.loads((output/'manifest.json').read_text())
        metadata['source_metadata']['air']['statistic']='Standard Deviation'
        (output/'manifest.json').write_text(json.dumps(metadata))
        with pytest.raises(ValueError,match='source metadata differs'):climate.SeasonalClimatology(output)


def test_log_pressure_interpolation_and_original_symlink_refusal(tmp_path):
    source=sources(tmp_path/'sources')
    def field(dataset):dataset.air.values[:]=np.log(dataset.level.values*100.)[None,:,None,None]
    mutate(source/climate.FILES[0],field)
    climate.prepare(source,tmp_path/'context',mesh_level=0)
    context=climate.SeasonalClimatology(tmp_path/'context')
    level=int(np.flatnonzero(climate.PRESSURE_PA==65000.)[0])
    assert np.allclose(context.mean[:,:,level,0],273.15+np.log(65000.))
    link=tmp_path/'source-link';link.symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError,match='symlinks'):
        climate.prepare(link,tmp_path/'linked-output',mesh_level=0)
