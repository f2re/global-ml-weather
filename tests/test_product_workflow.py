"""Analytic fixtures, not observations. Test the actual retrieval/export path."""
from dataclasses import replace
from pathlib import Path
import json
import hashlib
import numpy as np
import pytest
import torch
from global_weather.products.core import Field, QC, canonical
from global_weather.products.io import save_field, load_field, load_product, sha256
from global_weather.products.algorithms import liquid_water_path, planck_radiance
from global_weather.products.soil import tau_omega_forward
from global_weather.products.workflow import preflight, run_batch, load_snapshot
from global_weather.products.__main__ import main

OBS='2026-10-04T12:00:00Z'
READY='2026-10-04T12:05:00Z'
ISSUE='2026-10-04T12:10:00Z'


def json_file(path,value):
    path.write_text(canonical(value)+'\n',encoding='utf-8')
    return {'path':path.name,'sha256':sha256(path)}


def fixture(root):
    root.mkdir(parents=True)
    files={}
    def field(name,quantity,units,values,metadata=None,uncertainty=None):
        values=np.broadcast_to(np.asarray(values,dtype=float),(2,2)).copy() if np.asarray(values).ndim<2 else np.asarray(values,dtype=float)
        valid=np.ones(values.shape,dtype=bool)
        m={**dict(source='meteor_msu_mr',platform='SYNTHETIC',data_kind='synthetic',calibration_family='SYNTHETIC'),**(metadata or {})}
        f=Field(values,valid,quantity,units,'test-grid',OBS,'2026-10-04T12:01:00Z',
                hashlib.sha256(('SYNTHETIC-'+name).encode()).hexdigest(),m,
                None if uncertainty is None else np.broadcast_to(uncertainty,values.shape).copy())
        path=root/(name+'.npz');save_field(path,f);files[name]=path
        return {'path':path.name,'sha256':sha256(path)}
    eligible=field('eligible','eligibility_mask','1',[[1,1],[1,0]])
    sun=field('sun','solar_zenith_angle','degree',30)
    nir=field('nir','surface_reflectance','1',[[.6,.5],[.8,.7]],{'spectral_role':'nir'})
    red=field('red','surface_reflectance','1',[[.2,.3],[.4,.2]],{'spectral_role':'red'})
    swir=field('swir','surface_reflectance','1',.3,{'spectral_role':'swir_1p6'})
    green=field('green','surface_reflectance','1',.5,{'spectral_role':'green'})
    toa_nir=field('toa_nir','toa_reflectance','1',.6,{'spectral_role':'nir'})
    toa_red=field('toa_red','toa_reflectance','1',.2,{'spectral_role':'red'})
    bt=field('bt','brightness_temperature','K',250,{'spectral_role':'thermal_window'})
    ct=field('ct','cloud_top_temperature','K',260,{'atmospheric_correction_verified':True},1.)
    t=field('profile_t','air_temperature','K',np.broadcast_to([280,260,240],(2,2,3)),uncertainty=1.)
    z=field('profile_z','height_above_mean_sea_level','m',np.broadcast_to([0,1000,3000],(2,2,3)))
    tau=field('tau','cloud_optical_depth','1',10,{'retrieval_verified':True},1.)
    re=field('radius','cloud_droplet_effective_radius','m',10e-6,{'retrieval_verified':True},1e-6)
    cov=field('cov','covariance_optical_depth_effective_radius','m',.5e-6)
    temp=285.;eps=.98;trans=.85;lup=.5;ldown=1.;wave=10.8
    rad=field('radiance','spectral_radiance','W m-2 sr-1 um-1',trans*(eps*planck_radiance(temp,wave)+(1-eps)*ldown)+lup,
              {'spectral_model':'monochromatic','wavelength_um':wave})
    emiss=field('emiss','surface_emissivity','1',eps)
    tr=field('trans','atmospheric_transmittance','1',trans)
    up=field('up','upwelling_spectral_radiance','W m-2 sr-1 um-1',lup,{'wavelength_um':wave})
    down=field('down','downwelling_spectral_radiance','W m-2 sr-1 um-1',ldown,{'wavelength_um':wave})
    moisture=np.array([.05,.2,.4,.55]);e=np.stack([.95-.5*moisture,.99-.4*moisture],-1)
    lut_meta=dict(source='meteor_mtvza',platform='SYNTHETIC',instrument='SYNTHETIC',condition_id='SYNTHETIC',
          dielectric_model='SYNTHETIC_LINEAR_NOT_CALIBRATION',soil_texture='SYNTHETIC',roughness='SYNTHETIC',
          license='SYNTHETIC',reference='Analytic software fixture; not measurements',data_kind='synthetic',
          units='m3 m-3',frequency_ghz=10.65,incidence_deg=40,depth_top_m=0,depth_bottom_m=.05,channel_ids=['H','V'])
    lut=json_file(root/'soil_lut.json',dict(moisture=moisture.tolist(),emissivity_hv=e.tolist(),metadata=lut_meta))
    target=np.array([[.25,.3],[.35,.4]])
    expected_e=np.stack([.95-.5*target,.99-.4*target],-1)
    brightness=tau_omega_forward(expected_e,290.,288.,.2,.05,40.)
    # Use explicit metadata without duplicate keyword arguments in helper.
    def soil_field(name,quantity,units,value,extra=None,uncertainty=None):
        ref=field(name,quantity,units,value,extra,uncertainty)
        p=files[name];f=load_field(p);p.unlink()
        save_field(p,replace(f,metadata={**f.metadata,'source':'meteor_mtvza'}))
        return {'path':p.name,'sha256':sha256(p)}
    sm_meta={k:lut_meta[k] for k in ('platform','instrument','condition_id','frequency_ghz')}
    sm_meta.update(atmosphere_corrected=True,footprint_id='SYNTHETIC')
    h=soil_field('h','surface_brightness_temperature','K',brightness[...,0],{**sm_meta,'channel_id':'H','polarization':'H'},.5)
    v=soil_field('v','surface_brightness_temperature','K',brightness[...,1],{**sm_meta,'channel_id':'V','polarization':'V'},.5)
    st=soil_field('st','effective_soil_temperature','K',290.)
    vt=soil_field('vt','vegetation_temperature','K',288.)
    tv=soil_field('tv','vegetation_optical_depth_nadir','1',.2)
    om=soil_field('om','vegetation_scattering_albedo','1',.05)
    inc=soil_field('inc','view_zenith_angle','degree',40.)
    ok=soil_field('soil_ok','eligibility_mask','1',[[1,1],[1,0]],{'checks':['unfrozen','snow_free','no_precipitation','open_water_excluded','rfi_screened']})
    recipes={
        'ndvi':dict(a=nir,b=red,eligible=eligible,solar_zenith=sun),
        'ndvi_toa':dict(a=toa_nir,b=toa_red,eligible=eligible,solar_zenith=sun),
        'ndmi':dict(a=nir,b=swir,eligible=eligible,solar_zenith=sun),
        'ndsi':dict(a=green,b=swir,eligible=eligible,solar_zenith=sun),
        'cloud_top_brightness_temperature':dict(bt=bt,cloudy=eligible),
        'cloud_top_height':dict(cloud_temperature=ct,temperature_profile=t,height_profile=z,opaque_single_layer=eligible),
        'cloud_liquid_water_path':dict(optical_depth=tau,effective_radius=re,liquid_single_layer=eligible,covariance=cov),
        'land_surface_temperature':dict(radiance=rad,emissivity=emiss,transmittance=tr,upwelling=up,downwelling=down,clear_land=eligible),
        'soil_moisture_surface':dict(tb_h=h,tb_v=v,soil_temperature=st,vegetation_temperature=vt,tau_nadir=tv,omega=om,incidence=inc,eligible=ok)}
    np.savez_compressed(root/'geometry.npz',latitude=np.full((2,2),55.),longitude=np.array([[30,31],[32,33.]]),
                        view_zenith_deg=np.full((2,2),40.),footprint_km=np.full((2,2),20.),grid_id='test-grid')
    geometry={'path':'geometry.npz','sha256':sha256(root/'geometry.npz')}
    items=[]
    for product,inputs in recipes.items():
        job=dict(schema='satellite-product-job-1',product=product,inputs=inputs,available_at=READY,
                 availability_reference='SYNTHETIC unit test availability',data_kind='synthetic')
        if product=='soil_moisture_surface':job['lut']=lut
        ref=json_file(root/(product+'.json'),job)
        items.append(dict(id=product,product=product,job=ref,geometry=geometry,required=True))
    plan=root/'plan.json';json_file(plan,dict(schema='satellite-product-batch-1',issue_time=ISSUE,data_kind='synthetic',items=items))
    return plan,files


@pytest.fixture
def sample(tmp_path):return fixture(tmp_path/'inputs')


def change_plan(plan,fn):
    data=json.loads(plan.read_text());fn(data);json_file(plan,data)


def change_job(plan,product,fn):
    data=json.loads(plan.read_text());item=next(i for i in data['items'] if i['product']==product)
    p=plan.parent/item['job']['path'];job=json.loads(p.read_text());fn(job)
    item['job']=json_file(p,job);json_file(plan,data)


def test_all_nine_products_calculate_and_export(sample,tmp_path):
    plan,_=sample
    assert preflight(plan)['status']=='inputs_ready'
    report=run_batch(plan,tmp_path/'result')
    assert report['status']=='prepared',report
    assert report['records']==27 and len(report['items'])==9
    records,registry,meta=load_snapshot(tmp_path/'result/snapshot.json')
    assert len(records)==27 and len(registry)==9 and meta['data_kind']=='synthetic'
    assert report['shared_dependencies'] and not report['physical_boundary_model']
    soil=load_product(tmp_path/'result/soil_moisture_surface/product.npz')
    assert np.allclose(soil.values[soil.valid],[.25,.3,.35],atol=1e-12)
    assert soil.metadata['attributes']['depth_bottom_m']==.05
    ndvi=load_product(tmp_path/'result/ndvi/product.npz')
    assert ndvi.values[0,0]==pytest.approx(.5)
    height=load_product(tmp_path/'result/cloud_top_height/product.npz')
    assert (height.values[height.valid]==1000.).all()
    lst=load_product(tmp_path/'result/land_surface_temperature/product.npz')
    assert np.allclose(lst.values[lst.valid],285.)
    lwp=load_product(tmp_path/'result/cloud_liquid_water_path/product.npz')
    assert lwp.values[0,0]==pytest.approx(2/3*1000*10*10e-6)
    assert lwp.method=='homogeneous-liquid-lwp-cov-v1'
    assert any(v.method==lwp.method for v in registry.values())
    assert all(i['valid_pixels']==3 and i['total_pixels']==4 for i in report['items'])
    assert all(r['quality']==1 and r['derivation']['meteorologically_validated'] is False for r in records)


def test_missing_optional_is_reported_not_zero_filled(sample,tmp_path):
    plan,_=sample
    def edit(d):
        d['items'][0]['required']=False;d['items'][0]['job']['path']='absent.json'
    change_plan(plan,edit)
    r=run_batch(plan,tmp_path/'out')
    assert r['status']=='prepared_partial' and r['records']==24 and r['missing_optional']==['ndvi']
    rows,_,_=load_snapshot(tmp_path/'out/snapshot.json');assert all(x['derivation']['product']!='ndvi' for x in rows)


def test_missing_required_blocks_snapshot(sample,tmp_path):
    plan,_=sample;change_plan(plan,lambda d:d['items'][0]['job'].update(path='absent.json'))
    assert preflight(plan)['status']=='blocked'
    r=run_batch(plan,tmp_path/'out')
    assert r['status']=='blocked' and not (tmp_path/'out/snapshot.json').exists()
    assert not (tmp_path/'out/observations.jsonl').exists()


@pytest.mark.parametrize('key,value',[('required','false'),('history_hours',9999),('min_valid_fraction',1.1),('id','../escape'),('product','made_up')])
def test_plan_configuration_rejected(sample,tmp_path,key,value):
    plan,_=sample;change_plan(plan,lambda d:d['items'][0].update({key:value}))
    with pytest.raises(ValueError):run_batch(plan,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_future_processing_not_used(sample,tmp_path):
    plan,_=sample;change_job(plan,'ndvi',lambda j:j.update(available_at='2026-10-04T13:00:00Z'))
    r=run_batch(plan,tmp_path/'out');assert r['status']=='blocked'
    assert r['items'][0]['reason']=='not_available'


def test_field_hash_mismatch(sample,tmp_path):
    plan,files=sample;files['nir'].write_bytes(b'corrupt')
    r=run_batch(plan,tmp_path/'out');assert r['status']=='blocked'
    assert r['items'][0]['reason']=='invalid_input'


def test_cloud_age_not_extended_to_ndvi_policy(sample,tmp_path):
    plan,_=sample;change_plan(plan,lambda d:d.update(issue_time='2026-10-04T16:00:00Z'))
    r=run_batch(plan,tmp_path/'out');assert r['status']=='blocked'
    assert next(i for i in r['items'] if i['id']=='cloud_top_height')['reason']=='stale_product'
    assert r['items'][0]['status']=='exported'


def test_masked_coverage_never_declares_full(sample,tmp_path):
    plan,_=sample;change_plan(plan,lambda d:d['items'][0].update(min_valid_fraction=1.))
    r=run_batch(plan,tmp_path/'out');assert r['status']=='blocked'
    assert r['items'][0]['reason']=='insufficient_valid_pixels'


def test_output_cannot_be_reused(sample,tmp_path):
    plan,_=sample;run_batch(plan,tmp_path/'out')
    with pytest.raises(FileExistsError):run_batch(plan,tmp_path/'out')


def test_symlink_output_rejected(sample,tmp_path):
    plan,_=sample;(tmp_path/'link').symlink_to(tmp_path/'target',target_is_directory=True)
    with pytest.raises(ValueError):run_batch(plan,tmp_path/'link')


def test_aggregate_record_limit(sample,tmp_path):
    plan,_=sample;change_plan(plan,lambda d:d.update(max_records=3))
    r=run_batch(plan,tmp_path/'out');assert r['status']=='blocked' and not (tmp_path/'out/snapshot.json').exists()


def test_registry_conflict_not_overwritten(sample,tmp_path):
    plan,_=sample
    change_plan(plan,lambda d:[i.update(variable='same') for i in d['items']])
    r=run_batch(plan,tmp_path/'out');assert r['status']=='blocked' and r['reason']=='registry_conflict'


def test_exact_repeated_job_deduplicated(sample,tmp_path):
    plan,_=sample
    def edit(d):
        original=d['items'][0];d['items']=[original,{**original,'id':'same_ndvi_again'}]
    change_plan(plan,edit)
    r=run_batch(plan,tmp_path/'out');assert r['status']=='prepared',r
    assert r['records']==3 and r['deduplicated']==3


def test_cli_blocked_nonzero_and_report(sample,tmp_path,capsys):
    plan,_=sample;change_plan(plan,lambda d:d['items'][0]['job'].update(path='absent.json'))
    with pytest.raises(SystemExit) as ex:main(['batch','--plan',str(plan),'--output',str(tmp_path/'out')])
    assert ex.value.code==2 and json.loads(capsys.readouterr().out)['status']=='blocked'


def test_snapshot_hash_is_checked(sample,tmp_path):
    plan,_=sample;run_batch(plan,tmp_path/'out');(tmp_path/'out/registry.json').write_text('{}')
    with pytest.raises(ValueError):load_snapshot(tmp_path/'out/snapshot.json')


def test_lwp_correlated_errors(sample):
    _,files=sample;tau,re,mask=(load_field(files[k]) for k in ('tau','radius','eligible'))
    cov=load_field(files['cov'])
    positive=liquid_water_path(tau,re,mask,available_at=READY,data_kind='synthetic',covariance=cov)
    negative=liquid_water_path(tau,re,mask,available_at=READY,data_kind='synthetic',covariance=replace(cov,values=-cov.values))
    independent=liquid_water_path(tau,re,mask,available_at=READY,data_kind='synthetic')
    assert positive.uncertainty[0,0]>independent.uncertainty[0,0]>negative.uncertainty[0,0]
    assert positive.values[0,0]==negative.values[0,0]
    assert positive.uncertainty[0,0]==pytest.approx(2/3*1000*np.sqrt(3e-10))


def test_lwp_impossible_covariance_masked(sample):
    _,files=sample;tau,re,mask=(load_field(files[k]) for k in ('tau','radius','eligible'))
    cov=load_field(files['cov']);cov=replace(cov,values=np.full((2,2),1.01e-6))
    p=liquid_water_path(tau,re,mask,available_at=READY,data_kind='synthetic',covariance=cov)
    assert not p.valid.any() and (p.qc[:1]&int(QC.DOMAIN)).all()


def test_lwp_unknown_std_does_not_make_known_sigma(sample):
    _,files=sample;tau,re,mask=(load_field(files[k]) for k in ('tau','radius','eligible'))
    cov=load_field(files['cov'])
    with pytest.raises(ValueError):liquid_water_path(replace(tau,uncertainty=None),re,mask,available_at=READY,data_kind='synthetic',covariance=cov)


def test_tb_not_cloud_temperature(sample,tmp_path):
    plan,files=sample
    change_job(plan,'cloud_top_height',lambda j:j['inputs'].update(cloud_temperature={'path':files['bt'].name,'sha256':sha256(files['bt'])}))
    r=run_batch(plan,tmp_path/'out');assert r['status']=='blocked'
    assert next(i for i in r['items'] if i['id']=='cloud_top_height')['status']=='blocked'


def test_synthetic_source_never_promoted_to_real(sample,tmp_path):
    plan,_=sample;change_plan(plan,lambda d:d.update(data_kind='real'))
    r=run_batch(plan,tmp_path/'out');assert r['status']=='blocked'
    assert all(i['reason']=='recipe_mismatch' for i in r['items'])


def test_missing_lut_is_not_a_default_calibration(sample,tmp_path):
    plan,_=sample;change_job(plan,'soil_moisture_surface',lambda j:j.pop('lut'))
    r=run_batch(plan,tmp_path/'out');assert r['status']=='blocked'
    assert r['items'][-1]['reason']=='missing_calibration'


def test_snapshot_reaches_72h_model_and_gradients(sample,tmp_path):
    from global_weather.grid import build_pyramid
    from global_weather.observations import pack_observations
    from global_weather.model import GlobalWeatherModel
    from global_weather.vertical import PRESSURE_HPA
    plan,_=sample;run_batch(plan,tmp_path/'out')
    rows,registry,meta=load_snapshot(tmp_path/'out/snapshot.json')
    grids=build_pyramid(1);pressure=np.array(PRESSURE_HPA)*100
    actual=pack_observations(rows,grids[0],pressure,meta['issue_time'],registry)
    absent=pack_observations([],grids[0],pressure,meta['issue_time'],registry)
    assert actual.accepted_records==27 and not actual.rejected
    torch.set_num_threads(1);torch.manual_seed(8)
    model=GlobalWeatherModel(grids,actual.vocabulary,observation_schema=actual.schema_fingerprint,hidden=16)
    zeros=torch.zeros(grids[0].n_cells)
    with torch.no_grad():
        a=list(model(actual,zeros,zeros,horizon_hours=72))[-1]
        b=list(model(absent,zeros,zeros,horizon_hours=72))[-1]
    assert a.profiles.shape==(42,37,6) and torch.isfinite(a.profiles).all()
    assert not torch.allclose(a.profiles,b.profiles)
    loss=list(model(actual,zeros,zeros,horizon_hours=3))[-1].profiles[...,0].mean()
    loss.backward()
    grad=model.encoder.variable.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and (grad.abs().sum(1)>0).all()


def replace_input_field(plan,files,key,transform):
    p=files[key];f=load_field(p);f=transform(f);p.unlink();save_field(p,f)
    data=json.loads(plan.read_text())
    for item in data['items']:
        jp=plan.parent/item['job']['path'];j=json.loads(jp.read_text());changed=False
        for ref in j['inputs'].values():
            if ref['path']==p.name:ref['sha256']=sha256(p);changed=True
        if changed:item['job']=json_file(jp,j)
    json_file(plan,data)


def test_composite_future_interval_refused(sample):
    _,files=sample;f=load_field(files['nir'])
    with pytest.raises(ValueError):replace(f,metadata={**f.metadata,'temporal_support':{'start':OBS,'end':'2026-10-04T13:00:00Z'}})


def test_composite_without_support_refused(sample):
    _,files=sample;f=load_field(files['nir'])
    with pytest.raises(ValueError):replace(f,metadata={**f.metadata,'composite':True})


def test_composite_history_checked_for_training_boundary(sample,tmp_path):
    plan,files=sample
    replace_input_field(plan,files,'nir',lambda f:replace(f,metadata={**f.metadata,'composite':True,
       'temporal_support':{'start':'2026-09-20T12:00:00Z','end':OBS}}))
    r=run_batch(plan,tmp_path/'out')
    assert r['status']=='blocked' and r['items'][0]['status']=='blocked'
    p=load_product(tmp_path/'out/ndvi/product.npz')
    assert any('temporal_support' in d for d in p.metadata['dependencies'])


def test_dependency_not_ready_is_refused(sample,tmp_path):
    plan,files=sample
    replace_input_field(plan,files,'nir',lambda f:replace(f,available_at='2026-10-04T12:06:00Z'))
    r=run_batch(plan,tmp_path/'out')
    assert r['items'][0]['reason']=='dependency_not_available'


def test_raw_microwave_counts_refused(sample,tmp_path):
    plan,files=sample
    replace_input_field(plan,files,'h',lambda f:replace(f,quantity='raw_counts',units='1'))
    r=run_batch(plan,tmp_path/'out')
    assert r['status']=='blocked' and r['items'][-1]['status']=='blocked'


def test_all_cloud_masked_is_missing_not_zero(sample,tmp_path):
    plan,files=sample
    replace_input_field(plan,files,'eligible',lambda f:replace(f,values=np.zeros((2,2))))
    r=run_batch(plan,tmp_path/'out')
    assert r['status']=='blocked' and r['items'][0]['reason']=='insufficient_valid_pixels'
    p=load_product(tmp_path/'out/ndvi/product.npz')
    assert not p.valid.any() and np.isnan(p.values).all()


def test_inversion_ambiguity_not_resolved_arbitrarily(sample,tmp_path):
    plan,files=sample
    replace_input_field(plan,files,'profile_t',lambda f:replace(f,values=np.broadcast_to([280.,240.,280.],(2,2,3)).copy()))
    r=run_batch(plan,tmp_path/'out')
    p=load_product(tmp_path/'out/cloud_top_height/product.npz')
    assert not p.valid.any() and p.qc[0,0]&int(QC.AMBIGUOUS)
    assert r['status']=='blocked'


def test_geometry_payload_failure_is_reported(sample,tmp_path):
    plan,_=sample;np.savez_compressed(plan.parent/'geometry.npz',latitude=[0.])
    change_plan(plan,lambda d:[i['geometry'].update(sha256=sha256(plan.parent/'geometry.npz')) for i in d['items']])
    r=run_batch(plan,tmp_path/'out')
    assert r['status']=='blocked' and all(i['status']=='blocked' for i in r['items'])


def test_invalid_npz_even_with_matching_hash_is_reported(sample,tmp_path):
    plan,_=sample;(plan.parent/'geometry.npz').write_bytes(b'not an npz')
    change_plan(plan,lambda d:[i['geometry'].update(sha256=sha256(plan.parent/'geometry.npz')) for i in d['items']])
    r=run_batch(plan,tmp_path/'out')
    assert r['status']=='blocked'


def test_lwp_missing_covariance_masks_pixel(sample):
    _,files=sample;tau,re,mask=(load_field(files[k]) for k in ('tau','radius','eligible'))
    cov=load_field(files['cov']);m=cov.valid.copy();m[0,0]=False;cov=replace(cov,valid=m)
    p=liquid_water_path(tau,re,mask,available_at=READY,data_kind='synthetic',covariance=cov)
    assert not p.valid[0,0] and p.qc[0,0]&int(QC.MISSING)


def test_lwp_covariance_units_must_not_be_correlation(sample):
    _,files=sample;tau,re,mask=(load_field(files[k]) for k in ('tau','radius','eligible'))
    cov=replace(load_field(files['cov']),units='1')
    with pytest.raises(ValueError):liquid_water_path(tau,re,mask,available_at=READY,data_kind='synthetic',covariance=cov)


def test_export_byte_limit_does_not_leave_partial_file(sample,tmp_path):
    from global_weather.products.ingest import export_product
    plan,_=sample;run_batch(plan,tmp_path/'out')
    target=tmp_path/'limited.jsonl'
    with pytest.raises(ValueError):
        export_product(tmp_path/'out/ndvi/product.npz',plan.parent/'geometry.npz',target,max_output_bytes=1)
    assert not target.exists()


def test_pack_helper_requires_norms_and_preserves_station_input(sample,tmp_path):
    from global_weather.products.workflow import pack_snapshot
    from global_weather.grid import build_grid
    from global_weather.vertical import PRESSURE_HPA
    from global_weather.observations import DEFAULT_VARIABLES
    plan,_=sample;run_batch(plan,tmp_path/'out');grid=build_grid(1);press=np.array(PRESSURE_HPA)*100
    with pytest.raises(ValueError):pack_snapshot(tmp_path/'out/snapshot.json',grid,press)
    rec=dict(observation_id='station',source='station',variable='t2m',value=285.,units='K',latitude=55.,longitude=30.,observed_at=OBS,available_at=READY)
    o,registry=pack_snapshot(tmp_path/'out/snapshot.json',grid,press,
          allow_synthetic_unscaled=True,records=[rec],variables={'t2m':DEFAULT_VARIABLES['t2m']})
    assert o.accepted_records==28 and len(registry)==10


def test_unknown_catalog_method_is_not_silently_replaced():
    from global_weather.products.catalog import variable_spec
    with pytest.raises(ValueError):variable_spec('ndvi','meteor_msu_mr','SYNTHETIC',method='wrong')


@pytest.mark.parametrize('product',['soil_moisture_surface','ndvi','cloud_liquid_water_path','cloud_top_height'])
def test_each_requested_product_has_a_72h_path(sample,tmp_path,product):
    from global_weather.grid import build_pyramid
    from global_weather.observations import pack_observations
    from global_weather.model import GlobalWeatherModel
    from global_weather.vertical import PRESSURE_HPA
    plan,_=sample;run_batch(plan,tmp_path/'out');rows,registry,meta=load_snapshot(tmp_path/'out/snapshot.json')
    grids=build_pyramid(0);p=np.array(PRESSURE_HPA)*100
    full=pack_observations(rows,grids[0],p,meta['issue_time'],registry)
    reduced=pack_observations([r for r in rows if r['derivation']['product']!=product],grids[0],p,meta['issue_time'],registry)
    assert full.accepted_records-reduced.accepted_records==3
    torch.set_num_threads(1);torch.manual_seed(7)
    m=GlobalWeatherModel(grids,full.vocabulary,observation_schema=full.schema_fingerprint,hidden=16)
    zero=torch.zeros(12)
    with torch.no_grad():
        a=list(m(full,zero,zero,horizon_hours=72))[-1]
        b=list(m(reduced,zero,zero,horizon_hours=72))[-1]
    assert torch.isfinite(a.profiles).all() and (a.profiles-b.profiles).abs().max().item()>0
