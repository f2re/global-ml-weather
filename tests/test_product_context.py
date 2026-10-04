"""Analytic optical inversion and complete product-context tests; no real retrieval claim."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from global_weather.products.core import Field, QC, utc
from global_weather.products.optical import (CloudOpticsLUT, CLOUD_CHECKS,
    retrieve_cloud_optics, cloud_lwp_from_reflectances)

TIME='2020-01-01T00:00:00+00:00'
READY='2020-01-01T00:05:00+00:00'
HASH=hashlib.sha256(b'analytic test fixture; not satellite measurements').hexdigest()


def optical_fixture(tau=3.,radius_um=8.,shape=(1,)):
    t=np.array([1.,5.,10.]);r=np.array([5.,10.,20.])*1e-6
    tt,rr=np.meshgrid(t,r*1e6,indexing='ij')
    refl=np.stack((.1+.02*tt+.005*rr,.05+.003*tt+.018*rr),-1)
    meta=dict(source='meteor_msu_mr',platform='SYNTHETIC',instrument='ANALYTIC',
        calibration_family='analytic-1',surface_condition_id='test-background',
        atmosphere_condition_id='test-atmosphere',radiative_transfer_model='analytic-affine-test-NOT-RT',
        reference='synthetic regression fixture',license='synthetic test',data_kind='synthetic',
        phase='liquid',vertical_model='homogeneous',quantity='toa_reflectance',
        channel_ids=['vis','swir'],wavelength_um=[.6,1.7],spectral_response_sha256=[HASH,HASH],
        solar_zenith_deg=30.,view_zenith_deg=20.,relative_azimuth_deg=60.,geometry_tolerance_deg=.1)
    lut=CloudOpticsLUT(t,r,refl,meta,HASH)
    def f(value,quantity,units='1',**metadata):
        return Field(np.full(shape,value),np.ones(shape,bool),quantity,units,'analytic-grid',TIME,READY,HASH,
                     {**meta,**metadata},np.full(shape,.001) if quantity=='toa_reflectance' else None)
    a=f(.1+.02*tau+.005*radius_um,'toa_reflectance',channel_id='vis',wavelength_um=.6,spectral_response_sha256=HASH)
    b=f(.05+.003*tau+.018*radius_um,'toa_reflectance',channel_id='swir',wavelength_um=1.7,spectral_response_sha256=HASH)
    fields=dict(nonabsorbing=a,absorbing=b,solar_zenith=f(30.,'solar_zenith_angle','degree'),
                view_zenith=f(20.,'view_zenith_angle','degree'),relative_azimuth=f(60.,'relative_azimuth_angle','degree'),
                eligible=f(1.,'eligibility_mask',checks=sorted(CLOUD_CHECKS)))
    return lut,fields


def solve(fields,lut,**params):
    return retrieve_cloud_optics(**fields,lut=lut,available_at=READY,data_kind='synthetic',**params)


def test_optical_exact_affine_inverse_and_covariance():
    lut,fields=optical_fixture();x=solve(fields,lut)
    assert x.valid.all() and x.optical_depth[0]==pytest.approx(3.)
    assert x.radius_m[0]==pytest.approx(8e-6)
    inverse=np.linalg.inv(np.array([[.02,.005e6],[.003,.018e6]]))
    expected=inverse@np.diag([.001**2,.001**2])@inverse.T
    np.testing.assert_allclose(x.covariance[0],expected,rtol=1e-10,atol=1e-20)
    assert expected[0,1] != 0


def test_optical_lwp_uses_off_diagonal_covariance():
    lut,fields=optical_fixture();x=solve(fields,lut)
    p=cloud_lwp_from_reflectances(**fields,lut=lut,available_at=READY,data_kind='synthetic')
    assert p.name=='cloud_liquid_water_path' and p.method=='reflectance-lut-liquid-lwp-v1'
    assert p.values[0]==pytest.approx(.016)
    g=np.array([8e-6,3.])*(2000/3)
    assert p.uncertainty[0]**2==pytest.approx(g@x.covariance[0]@g,rel=1e-10)
    independent=np.sum(g*g*np.diag(x.covariance[0]))
    assert not np.isclose(p.uncertainty[0]**2,independent,rtol=.01,atol=0)
    assert p.metadata['attributes']['parameter_covariance_used'] is True
    assert p.metadata['meteorologically_validated'] is False


@pytest.mark.parametrize('key,value', [('channel_id','wrong'),('instrument','other'),
    ('calibration_family','other'),('surface_condition_id','other'),('atmosphere_condition_id','other'),
    ('spectral_response_sha256','0'*64),('wavelength_um',.8)])
def test_optical_rejects_wrong_sensor_or_conditions(key,value):
    lut,fields=optical_fixture();a=fields['nonabsorbing']
    fields['nonabsorbing']=replace(a,metadata={**a.metadata,key:value})
    with pytest.raises(ValueError):solve(fields,lut)


def test_optical_rejects_fake_smap_or_37_micron_table():
    lut,_=optical_fixture();meta={**lut.metadata,'wavelength_um':[.6,3.7]}
    with pytest.raises(ValueError):replace(lut,metadata=meta)


def test_optical_needs_all_eligibility_checks():
    lut,fields=optical_fixture();a=fields['eligible']
    fields['eligible']=replace(a,metadata={**a.metadata,'checks':['liquid_phase']})
    with pytest.raises(ValueError):solve(fields,lut)


def test_optical_needs_radiometric_uncertainty():
    lut,fields=optical_fixture();fields['absorbing']=replace(fields['absorbing'],uncertainty=None)
    with pytest.raises(ValueError):solve(fields,lut)


def test_optical_rejects_unavailable_dependency():
    lut,fields=optical_fixture();a=fields['absorbing']
    fields['absorbing']=replace(a,available_at='2020-01-01T00:06:00Z')
    with pytest.raises(ValueError,match='раньше'):solve(fields,lut)


def test_optical_synthetic_lut_is_not_real_retrieval():
    lut,fields=optical_fixture()
    with pytest.raises(ValueError):retrieve_cloud_optics(**fields,lut=lut,available_at=READY,data_kind='real')


def test_optical_angle_outside_slice_is_masked():
    lut,fields=optical_fixture();fields['solar_zenith']=replace(fields['solar_zenith'],values=np.array([45.]))
    x=solve(fields,lut);assert not x.valid.any() and x.qc[0]&QC.CONDITIONS
    assert np.isnan(x.optical_depth[0])


def test_optical_negative_angle_does_not_enter_zero_degree_slice():
    lut,fields=optical_fixture();lut=replace(lut,metadata={**lut.metadata,'view_zenith_deg':0.})
    fields['view_zenith']=replace(fields['view_zenith'],values=np.array([-.01]))
    assert not solve(fields,lut).valid.any()


def test_optical_boundary_is_not_extrapolated():
    lut,fields=optical_fixture(tau=1.);x=solve(fields,lut)
    assert not x.valid.any() and x.qc[0]&QC.BOUNDARY


def test_optical_no_solution_is_not_a_clipped_output():
    lut,fields=optical_fixture(tau=30.);x=solve(fields,lut)
    assert not x.valid.any() and x.qc[0]&QC.NO_SOLUTION


def test_optical_large_noise_masks_low_sensitivity():
    lut,fields=optical_fixture()
    for key in ('nonabsorbing','absorbing'):
        fields[key]=replace(fields[key],uncertainty=np.ones(1))
    x=solve(fields,lut);assert not x.valid.any() and x.qc[0]&QC.LOW_SENSITIVITY


@pytest.mark.parametrize('shape',[(1,),()])
def test_optical_degenerate_table_supports_scalar_and_array(shape):
    lut,fields=optical_fixture(shape=shape);lut=replace(lut,reflectance=np.full(lut.reflectance.shape,.3))
    x=solve(fields,lut);assert not x.valid.any() and (x.qc&QC.LOW_SENSITIVITY).all()


def test_optical_missing_reflectance_keeps_mask():
    lut,fields=optical_fixture(shape=(2,))
    a=fields['absorbing'];fields['absorbing']=replace(a,values=np.array([a.values[0],np.nan]),valid=np.array([True,False]))
    x=solve(fields,lut);assert x.valid.tolist()==[True,False] and np.isnan(x.radius_m[1])


def test_optical_multiple_parameter_roots_rejected():
    lut,fields=optical_fixture()
    ff=lut.reflectance.copy();ff[:,:,0]=np.array([.1,.4,.1])[:,None]
    ff[:,:,1]=np.array([.1,.2,.4])[None,:]
    lut=replace(lut,reflectance=ff)
    fields['nonabsorbing']=replace(fields['nonabsorbing'],values=np.array([.25]))
    fields['absorbing']=replace(fields['absorbing'],values=np.array([.25]))
    x=solve(fields,lut);assert not x.valid.any() and x.qc[0]&QC.AMBIGUOUS


def write_optical_job(root):
    from global_weather.products.io import save_field,sha256,canonical
    root.mkdir(parents=True,exist_ok=True);lut,fields=optical_fixture()
    refs={}
    for name,f in fields.items():
        path=root/(name+'.npz');save_field(path,f);refs[name]={'path':path.name,'sha256':sha256(path)}
    lp=root/'lut.json';lp.write_text(canonical(dict(optical_depth=lut.optical_depth.tolist(),
        radius_m=lut.radius_m.tolist(),reflectance=lut.reflectance.tolist(),metadata=lut.metadata)))
    job=dict(schema='satellite-product-job-1',product='cloud_liquid_water_path',
        method='reflectance-lut-liquid-lwp-v1',inputs=refs,available_at=READY,
        availability_reference='analytic-test-only',data_kind='synthetic',lut={'path':lp.name,'sha256':sha256(lp)})
    path=root/'job.json';path.write_text(canonical(job))
    gp=root/'geometry.npz';np.savez(gp,latitude=np.array([60.]),longitude=np.array([30.]),
         view_zenith_deg=np.array([20.]),footprint_km=np.array([4.]),grid_id=np.array('analytic-grid'))
    return path,gp


def test_cli_optical_calculate_export_preserves_method_and_admission(tmp_path):
    from global_weather.products.__main__ import calculate
    from global_weather.products.ingest import export_product
    from global_weather.products.io import load_product
    from global_weather.observations import Variable,pack_observations
    from global_weather.grid import build_grid
    from global_weather.vertical import PRESSURE_HPA
    job,geo=write_optical_job(tmp_path/'input');target=tmp_path/'lwp.npz'
    report=calculate(job,target);assert report['valid_pixels']==1
    export=export_product(target,geo,tmp_path/'features.jsonl')
    assert next(iter(export['registry'].values()))['method']=='reflectance-lut-liquid-lwp-v1'
    registry={k:Variable(**v) for k,v in export['registry'].items()}
    records=[json.loads(line) for line in (tmp_path/'features.jsonl').read_text().splitlines()]
    packed=pack_observations(records,build_grid(0),np.array(PRESSURE_HPA)*100,utc(READY),registry)
    assert packed.accepted_records==1 and not packed.rejected
    assert load_product(target).values[0]==pytest.approx(.016)


def context_plan(root, *, optional_missing=False, required_missing=False,issue=READY):
    from global_weather.products.io import sha256,canonical
    job,geometry=write_optical_job(root)
    item=dict(id='lwp',job=dict(path=job.name,sha256=sha256(job)),
              geometry=dict(path=geometry.name,sha256=sha256(geometry)),required=True)
    items=[item]
    if optional_missing or required_missing:
        items.append(dict(id='soil',job=dict(path='missing.json',sha256=HASH),geometry=item['geometry'],required=required_missing))
    plan=dict(schema='satellite-context-plan-1',issue_time=issue,data_kind='synthetic',items=items,max_records=100)
    pp=root/'plan.json';pp.write_text(canonical(plan));return pp


def test_batch_complete_context(tmp_path):
    from global_weather.products.batch import build_context
    plan=context_plan(tmp_path/'input');out=tmp_path/'context';r=build_context(plan,out)
    assert r['status']=='prepared' and r['records']==1 and not r['model_admission_granted']
    assert len((out/'derived.jsonl').read_text().splitlines())==1
    assert (out/'registry.json').is_file() and (out/'dependencies.json').is_file()


@pytest.mark.parametrize('required',[False,True])
def test_batch_reports_optional_and_required_failure(tmp_path,required):
    from global_weather.products.batch import build_context
    plan=context_plan(tmp_path/'input',optional_missing=not required,required_missing=required)
    r=build_context(plan,tmp_path/'context')
    assert r['status']==('blocked' if required else 'partial') and r['records']==1
    assert r['items'][1]['status']=='blocked'


def test_batch_requires_nonzero_exit_for_missing_required(tmp_path):
    from global_weather.products.__main__ import main
    plan=context_plan(tmp_path/'input',required_missing=True)
    with pytest.raises(SystemExit) as e:main(['build-context','--plan',str(plan),'--output',str(tmp_path/'out')])
    assert e.value.code==2


@pytest.mark.parametrize('issue',['2019-12-31T23:59:00Z','2020-01-01T04:00:00Z'])
def test_batch_rejects_future_or_stale_cloud_product(tmp_path,issue):
    from global_weather.products.batch import build_context
    r=build_context(context_plan(tmp_path/'input',issue=issue),tmp_path/'out')
    assert r['status']=='blocked' and r['records']==0


def test_batch_source_hash_change_blocks_product(tmp_path):
    from global_weather.products.batch import build_context
    plan=context_plan(tmp_path/'input');(plan.parent/'job.json').write_text('{}')
    r=build_context(plan,tmp_path/'out');assert r['status']=='blocked' and r['records']==0


def test_batch_no_overwrite(tmp_path):
    from global_weather.products.batch import build_context
    plan=context_plan(tmp_path/'input');out=tmp_path/'out';build_context(plan,out)
    with pytest.raises(FileExistsError):build_context(plan,out)


def test_batch_rejects_duplicate_plan_ids(tmp_path):
    from global_weather.products.batch import build_context
    plan=context_plan(tmp_path/'input');data=json.loads(plan.read_text());data['items']*=2;plan.write_text(json.dumps(data))
    with pytest.raises(ValueError):build_context(plan,tmp_path/'out')


def test_cli_does_not_accept_optical_inputs_as_old_lwp_method(tmp_path):
    from global_weather.products.__main__ import calculate
    job,_=write_optical_job(tmp_path/'input');data=json.loads(job.read_text());del data['method'];job.write_text(json.dumps(data))
    with pytest.raises(ValueError):calculate(job,tmp_path/'out.npz')


def test_optical_model_end_to_end_gradients_and_72h(tmp_path):
    from global_weather.products.__main__ import calculate
    from global_weather.products.ingest import export_product
    from global_weather.observations import Variable,pack_observations
    from global_weather.grid import build_pyramid
    from global_weather.vertical import PRESSURE_HPA
    from global_weather.adaptive import AdaptiveWeatherModel
    import torch
    torch.set_num_threads(1);torch.manual_seed(23)
    job,geo=write_optical_job(tmp_path/'input');product=tmp_path/'lwp.npz';calculate(job,product)
    rows=tmp_path/'rows.jsonl';export=export_product(product,geo,rows)
    reg={k:Variable(**v) for k,v in export['registry'].items()};grid=build_pyramid(0)
    records=[json.loads(line) for line in rows.read_text().splitlines()]
    obs=pack_observations(records,grid[0],np.array(PRESSURE_HPA)*100,utc(READY),reg)
    model=AdaptiveWeatherModel(grid,obs.vocabulary,observation_schema=obs.schema_fingerprint,
             hidden=16,latent_slots=4,allow_unscaled_synthetic=True)
    static=torch.zeros(grid[0].n_cells)
    a=model.analyse(obs,static,static)
    missing=pack_observations([],grid[0],np.array(PRESSURE_HPA)*100,utc(READY),reg)
    b=model.analyse(missing,static,static)
    assert not torch.allclose(a,b)
    frame=list(model(obs,static,static,horizon_hours=3))[-1]
    (frame.profiles[...,0]/300).square().mean().backward()
    assert torch.isfinite(model.encoder.value[0].weight.grad).all()
    assert model.encoder.value[0].weight.grad.abs().sum()>0
    with torch.inference_mode():
        frames=list(model(obs,static,static,horizon_hours=72))
    assert frames[-1].lead_hours==72 and all(torch.isfinite(f.profiles).all() for f in frames)


def test_batch_combines_ndvi_soil_height_and_optical_water(tmp_path):
    from global_weather.products.io import save_field,sha256,canonical
    from global_weather.products.batch import build_context
    from global_weather.observations import Variable,pack_observations
    from global_weather.grid import build_grid
    from global_weather.vertical import PRESSURE_HPA
    from global_weather.products.soil import tau_omega_forward,SOIL_CHECKS
    root=tmp_path/'inputs';job,geo=write_optical_job(root/'optical');items=[]
    def entry(name,j,g):
        return dict(id=name,required=True,job=dict(path=str(j.relative_to(root)),sha256=sha256(j)),
                    geometry=dict(path=str(g.relative_to(root)),sha256=sha256(g)))
    items.append(entry('water',job,geo))
    metadata=dict(source='meteor_msu_mr',platform='SYNTHETIC',calibration_family='test',data_kind='synthetic')
    def field(value,q,u='1',**kwargs):
        v=np.asarray(value,float);v=v.reshape(1) if v.ndim==0 else v
        return Field(v,np.ones(v.shape,bool),q,u,'analytic-grid',TIME,READY,HASH,{**metadata,**kwargs})
    def build_job(name,product,fields,lut=None):
        d=root/name;d.mkdir();refs={}
        for k,f in fields.items():
            fp=d/(k+'.npz');save_field(fp,f);refs[k]=dict(path=fp.name,sha256=sha256(fp))
        job=dict(schema='satellite-product-job-1',product=product,inputs=refs,available_at=READY,
                 availability_reference='analytic-fixture-only',data_kind='synthetic')
        if lut is not None:
            lp=d/'lut.json';lp.write_text(canonical(lut));job['lut']=dict(path=lp.name,sha256=sha256(lp))
        j=d/'job.json';j.write_text(canonical(job));items.append(entry(name,j,geo))
    build_job('vegetation','ndvi',dict(a=field(.6,'surface_reflectance',spectral_role='nir'),
        b=field(.2,'surface_reflectance',spectral_role='red'),eligible=field(1.,'eligibility_mask'),
        solar_zenith=field(30.,'solar_zenith_angle','degree')))
    build_job('height','cloud_top_height',dict(cloud_temperature=field(265.,'cloud_top_temperature','K',atmospheric_correction_verified=True),
        temperature_profile=field([[270.,260.,250.]],'air_temperature','K'),
        height_profile=field([[1000.,2000.,3000.]],'height_above_mean_sea_level','m'),
        opaque_single_layer=field(1.,'eligibility_mask')))
    sm=dict(source='meteor_mtvza',platform='SYNTHETIC',instrument='TEST',condition_id='test-soil',
        dielectric_model='analytic-test-only',soil_texture='test',roughness='test',license='test',
        reference='analytic-fixture-only',data_kind='synthetic',units='m3 m-3',frequency_ghz=10.6,
        incidence_deg=40.,depth_top_m=0,depth_bottom_m=.05,channel_ids=['h','v'])
    soil_lut=dict(moisture=[0.,.2,.4],emissivity_hv=[[.9,.95],[.8,.85],[.7,.75]],metadata=sm)
    observed=tau_omega_forward(np.array([.8,.85]),290.,288.,.2,.05,40.)
    tb={}
    for index,pol in enumerate(('H','V')):
        tb['tb_'+pol.lower()]=replace(field(observed[index],'surface_brightness_temperature','K',
            **{k:v for k,v in sm.items() if k not in ('units',)},polarization=pol,channel_id=pol.lower(),
            atmosphere_corrected=True,footprint_id='test-footprint'),uncertainty=np.array([.1]))
    build_job('soil','soil_moisture_surface',dict(**tb,soil_temperature=field(290.,'effective_soil_temperature','K'),
        vegetation_temperature=field(288.,'vegetation_temperature','K'),tau_nadir=field(.2,'vegetation_optical_depth_nadir'),
        omega=field(.05,'vegetation_scattering_albedo'),incidence=field(40.,'view_zenith_angle','degree'),
        eligible=field(1.,'eligibility_mask',checks=sorted(SOIL_CHECKS))),soil_lut)
    plan=root/'plan.json';plan.write_text(canonical(dict(schema='satellite-context-plan-1',issue_time=READY,
        data_kind='synthetic',items=items,max_records=100)))
    output=tmp_path/'context';report=build_context(plan,output)
    assert report['status']=='prepared',report
    assert report['records']==4
    rows=[json.loads(x) for x in (output/'derived.jsonl').read_text().splitlines()]
    values={r['derivation']['product']:r['value'] for r in rows}
    assert values['ndvi']==pytest.approx(.5) and values['cloud_top_height']==pytest.approx(1500.)
    assert values['soil_moisture_surface']==pytest.approx(.2) and values['cloud_liquid_water_path']==pytest.approx(.016)
    registry={k:Variable(**v) for k,v in json.loads((output/'registry.json').read_text()).items()}
    packed=pack_observations(rows,build_grid(0),np.array(PRESSURE_HPA)*100,utc(READY),registry)
    assert packed.accepted_records==4 and not packed.rejected
