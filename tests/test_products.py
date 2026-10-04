"""Analytic tests of conditional retrievals, not real satellite validation."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
import torch
from global_weather.products.core import Field, QC, validate_metadata
from global_weather.products.catalog import variable_spec
from global_weather.products.algorithms import (spectral_index, cloud_height, cloud_brightness_temperature,
    liquid_water_path, planck_radiance, surface_temperature)
from global_weather.products.soil import SoilEmissivityLUT, tau_omega_forward, soil_moisture, SOIL_CHECKS
from global_weather.products.io import save_field, save_product, load_product, load_field, arrays, sha256, resolve
from global_weather.products.ingest import export_product, check_record, maximum_history
from global_weather.products.__main__ import calculate, main
from global_weather.observations import Variable, pack_observations
from global_weather.grid import build_pyramid
from global_weather.vertical import PRESSURE_HPA

T = datetime(2020, 1, 10, 12, tzinfo=timezone.utc)
TS=T.isoformat(); READY=(T+timedelta(minutes=10)).isoformat()
META=dict(source='meteor_msu_mr',platform='SYNTHETIC',calibration_family='analytic-v1',data_kind='synthetic')
KW=dict(available_at=READY,data_kind='synthetic')


def field(values, quantity, units='1', *, metadata=None, **kw):
    x=np.asarray(values,dtype=float)
    return Field(x,kw.pop('valid',np.ones(x.shape,bool)),quantity,units,'analytic-grid',
                 kw.pop('observed_at',TS),kw.pop('available_at',TS),
                 hashlib.sha256((quantity+str(x.tolist())).encode()).hexdigest(),
                 {**META,**(metadata or {})},**kw)


def spectral(value=.6, **kw):
    a=field([value],'surface_reflectance',metadata={'spectral_role':'nir'})
    b=field([.2],'surface_reflectance',metadata={'spectral_role':'red'})
    c=field([1.],'eligibility_mask');s=field([30.],'solar_zenith_angle','degree')
    return spectral_index(a,b,c,s,product='ndvi',**{**KW,**kw})


@pytest.mark.parametrize('product,a_role,b_role,q',[('ndvi','nir','red','surface_reflectance'),
 ('ndvi_toa','nir','red','toa_reflectance'),('ndmi','nir','swir_1p6','surface_reflectance'),
 ('ndsi','green','swir_1p6','surface_reflectance')])
def test_index_formula_and_separate_identity(product,a_role,b_role,q):
    a=field([.6],q,metadata={'spectral_role':a_role},uncertainty=np.array([.01]))
    b=field([.2],q,metadata={'spectral_role':b_role},uncertainty=np.array([.01]))
    p=spectral_index(a,b,field([1.],'eligibility_mask'),field([30.],'solar_zenith_angle','degree'),product=product,**KW)
    assert p.values[0]==pytest.approx(.5) and p.valid.all() and p.uncertainty[0]>0
    assert p.name==product and p.metadata['meteorologically_validated'] is False


@pytest.mark.parametrize('a,b,sun,flag',[(0.,0.,30.,1.),(.6,.2,95.,1.),(.6,.2,30.,0.),(-.1,.2,30.,1.),(1.2,.2,30.,1.)])
def test_unusable_indices_are_masked_not_zero(a,b,sun,flag):
    x=field([a],'surface_reflectance',metadata={'spectral_role':'nir'})
    y=field([b],'surface_reflectance',metadata={'spectral_role':'red'})
    p=spectral_index(x,y,field([flag],'eligibility_mask'),field([sun],'solar_zenith_angle','degree'),product='ndvi',**KW)
    assert not p.valid.any() and np.isnan(p.values).all() and p.qc[0]!=0


def test_index_rejects_missing_swir_and_raw_counts():
    for q,role in [('raw_counts','nir'),('surface_reflectance','thermal_window')]:
        with pytest.raises(ValueError):
            spectral_index(field([.6],q,metadata={'spectral_role':role}),field([.2],'surface_reflectance',metadata={'spectral_role':'red'}),
                field([1.],'eligibility_mask'),field([30.],'solar_zenith_angle','degree'),product='ndmi',**KW)


def test_real_label_cannot_promote_synthetic():
    with pytest.raises(ValueError,match='синтетический'):spectral(data_kind='real')


def test_inputs_cannot_be_available_after_product():
    with pytest.raises(ValueError): spectral(available_at=(T-timedelta(seconds=1)).isoformat())


def test_different_grid_or_time_rejected():
    a=field([.6],'surface_reflectance',metadata={'spectral_role':'nir'})
    b=field([.2],'surface_reflectance',metadata={'spectral_role':'red'})
    for other in [replace(b,grid_id='other'),replace(b,observed_at=(T-timedelta(hours=1)).isoformat())]:
        with pytest.raises(ValueError):spectral_index(a,other,field([1.],'eligibility_mask'),field([30.],'solar_zenith_angle','degree'),product='ndvi',**KW)


def test_stale_cloud_mask_is_not_current():
    old=field([1.],'eligibility_mask',observed_at=(T-timedelta(hours=1)).isoformat())
    with pytest.raises(ValueError):cloud_brightness_temperature(field([260.],'brightness_temperature','K',metadata={'spectral_role':'thermal_window'}),old,**KW)


def test_cloud_brightness_is_not_thermodynamic_temperature():
    bt=field([250.],'brightness_temperature','K',metadata={'spectral_role':'thermal_window'})
    p=cloud_brightness_temperature(bt,field([1.],'eligibility_mask'),**KW)
    assert p.name=='cloud_top_brightness_temperature' and p.values[0]==250.
    with pytest.raises(ValueError):cloud_height(bt,field([[280.,250.]],'air_temperature','K'),
        field([[0.,5000.]],'height_above_mean_sea_level','m'),field([1.],'eligibility_mask'),**KW)


def height_args(target=270., profile=None):
    return (field([target],'cloud_top_temperature','K',metadata={'atmospheric_correction_verified':True},uncertainty=np.array([1.])),
            field([profile or [280.,260.,240.]],'air_temperature','K',uncertainty=np.ones((1,3))),
            field([[0.,4000.,8000.]],'height_above_mean_sea_level','m'),field([1.],'eligibility_mask'))


def test_cloud_height_uses_actual_profile():
    p=cloud_height(*height_args(),**KW)
    assert p.values[0]==pytest.approx(2000.) and p.uncertainty[0]>0
    assert p.metadata['attributes']['profile_assisted']


@pytest.mark.parametrize('target,profile,flag',[(270.,[280.,260.,280.],QC.AMBIGUOUS),
    (220.,[280.,260.,240.],QC.NO_SOLUTION),(270.,[270.,270.,260.],QC.AMBIGUOUS)])
def test_cloud_height_masks_ambiguity_and_no_extrapolation(target,profile,flag):
    p=cloud_height(*height_args(target,profile),**KW)
    assert not p.valid.any() and p.qc[0] & int(flag)


def test_exact_profile_level_has_one_root():
    p=cloud_height(*height_args(260.),**KW)
    assert p.valid.all() and p.values[0]==4000.


def test_unverified_cloud_temperature_rejected():
    args=list(height_args());args[0]=replace(args[0],metadata=META)
    with pytest.raises(ValueError):cloud_height(*args,**KW)


def test_cloud_height_requires_real_vertical_geometry():
    args=list(height_args());args[2]=replace(args[2],values=np.array([[0.,4000.,3000.]]))
    with pytest.raises(ValueError):cloud_height(*args,**KW)


def lwp_args():
    return (field([10.],'cloud_optical_depth',metadata={'retrieval_verified':True}),
            field([10e-6],'cloud_droplet_effective_radius','m',metadata={'retrieval_verified':True}),
            field([1.],'eligibility_mask'))


def test_liquid_water_mass_formula():
    p=liquid_water_path(*lwp_args(),**KW)
    assert p.values[0]==pytest.approx(2/3*.1) and p.metadata['units']=='kg m-2'


def test_cloud_water_not_inferred_without_radius_or_liquid_phase():
    args=list(lwp_args());args[2]=field([0.],'eligibility_mask')
    assert not liquid_water_path(*args,**KW).valid.any()
    args=list(lwp_args());args[1]=replace(args[1],units='um')
    with pytest.raises(ValueError):liquid_water_path(*args,**KW)
    args=list(lwp_args());args[0]=replace(args[0],metadata=META)
    with pytest.raises(ValueError):liquid_water_path(*args,**KW)


def test_surface_temperature_radiative_inversion():
    unit='W m-2 sr-1 um-1';wl=11.;eps=.96;tau=.8;up=1.;down=2.;temperature=300.
    L=tau*(eps*planck_radiance(temperature,wl)+(1-eps)*down)+up
    args=(field([L],'spectral_radiance',unit,metadata={'spectral_model':'monochromatic','wavelength_um':wl}),
          field([eps],'surface_emissivity'),field([tau],'atmospheric_transmittance'),
          field([up],'upwelling_spectral_radiance',unit,metadata={'wavelength_um':wl}),
          field([down],'downwelling_spectral_radiance',unit,metadata={'wavelength_um':wl}),field([1.],'eligibility_mask'))
    p=surface_temperature(*args,**KW)
    assert p.values[0]==pytest.approx(temperature)
    bad=list(args);bad[0]=replace(bad[0],metadata={**META,'spectral_model':'broadband','wavelength_um':wl})
    with pytest.raises(ValueError):surface_temperature(*bad,**KW)


def soil_case(theta=.2):
    m=dict(source='meteor_mtvza',platform='SYNTHETIC',instrument='SYNTHETIC_MW',condition_id='test',
        dielectric_model='analytic_fixture_not_real_soil',soil_texture='synthetic',roughness='synthetic',
        license='test fixture',reference='analytic test only',data_kind='synthetic',units='m3 m-3',
        frequency_ghz=10.7,incidence_deg=40.,depth_top_m=0.,depth_bottom_m=.03,channel_ids=['test-h','test-v'])
    moisture=np.linspace(0.,.5,101);e=np.stack([.95-.6*moisture,.98-.4*moisture],-1)
    lut=SoilEmissivityLUT(moisture,e,m,hashlib.sha256(b'analytic lut').hexdigest())
    tb=tau_omega_forward(np.array([.95-.6*theta,.98-.4*theta]),290.,288.,.1,.05,40.)
    common={**m,'atmosphere_corrected':True,'footprint_id':'test-footprint'}
    h=field([tb[0]],'surface_brightness_temperature','K',metadata={**common,'polarization':'H','channel_id':'test-h'},uncertainty=np.array([.5]))
    v=field([tb[1]],'surface_brightness_temperature','K',metadata={**common,'polarization':'V','channel_id':'test-v'},uncertainty=np.array([.5]))
    args=[h,v,field([290.],'effective_soil_temperature','K'),field([288.],'vegetation_temperature','K'),
          field([.1],'vegetation_optical_depth_nadir'),field([.05],'vegetation_scattering_albedo'),
          field([40.],'view_zenith_angle','degree'),field([1.],'eligibility_mask',metadata={'checks':sorted(SOIL_CHECKS)}),lut]
    return args


@pytest.mark.parametrize('theta',[.07,.2,.437])
def test_soil_retrieval_inverts_forward_equation(theta):
    p=soil_moisture(*soil_case(theta),**KW)
    assert p.valid.all() and p.values[0]==pytest.approx(theta,abs=1e-10)
    assert p.uncertainty[0]>0 and p.metadata['attributes']['depth_bottom_m']==.03


def test_soil_boundaries_are_not_false_exact_values():
    for theta in (0.,.5):
        p=soil_moisture(*soil_case(theta),**KW)
        assert not p.valid.any() and p.qc[0] & int(QC.BOUNDARY)


@pytest.mark.parametrize('change',['frozen','frequency','angle','raw','noise','kind','footprint'])
def test_soil_requires_calibration_conditions_and_geometry(change):
    args=soil_case()
    if change=='frozen':args[7]=field([1.],'eligibility_mask',metadata={'checks':['snow_free']})
    if change=='frequency':args[0]=replace(args[0],metadata={**args[0].metadata,'frequency_ghz':36.})
    if change=='angle':args[6]=field([55.],'view_zenith_angle','degree')
    if change=='raw':args[0]=replace(args[0],quantity='raw_counts')
    if change=='noise':args[0]=replace(args[0],uncertainty=None)
    if change=='footprint':args[1]=replace(args[1],metadata={**args[1].metadata,'footprint_id':'different'})
    if change=='angle':assert not soil_moisture(*args,**KW).valid.any()
    else:
        with pytest.raises(ValueError):soil_moisture(*args,**{**KW,'data_kind':'real' if change=='kind' else 'synthetic'})


def test_soil_high_canopy_has_low_sensitivity():
    args=soil_case();args[4]=field([100.],'vegetation_optical_depth_nadir')
    p=soil_moisture(*args,**KW)
    assert not p.valid.any() and p.qc[0]&int(QC.LOW_SENSITIVITY)


def test_soil_lut_not_arbitrary_constant():
    args=soil_case();lut=args[-1]
    with pytest.raises(ValueError):replace(lut,emissivity_hv=np.ones((101,2))*.9)
    with pytest.raises(ValueError):replace(lut,metadata={**lut.metadata,'depth_bottom_m':1.})


def test_product_file_roundtrip_and_no_overwrite(tmp_path):
    p=spectral();save_product(tmp_path/'p.npz',p);read=load_product(tmp_path/'p.npz')
    assert read.values==pytest.approx(p.values)
    with pytest.raises(FileExistsError):save_product(tmp_path/'p.npz',p)


def test_field_hash_is_actual_bytes(tmp_path):
    f=field([1.],'eligibility_mask');save_field(tmp_path/'f.npz',f)
    assert load_field(tmp_path/'f.npz').source_sha256==sha256(tmp_path/'f.npz')


def test_unsafe_arrays_and_links_fail(tmp_path):
    np.savez(tmp_path/'object.npz',values=np.array([{}],dtype=object))
    with pytest.raises(ValueError):arrays(tmp_path/'object.npz')
    (tmp_path/'link').symlink_to(tmp_path/'object.npz')
    with pytest.raises(ValueError):arrays(tmp_path/'link')
    with pytest.raises(ValueError):resolve(tmp_path,{'path':'../other','sha256':'a'*64})


def export_case(tmp_path, value=.6, history=None):
    tmp_path.mkdir(exist_ok=True)
    p=spectral(value);path=tmp_path/'p.npz';save_product(path,p)
    geo=tmp_path/'g.npz';np.savez(geo,latitude=np.array([60.]),longitude=np.array([30.]),
                                view_zenith_deg=np.array([20.]),footprint_km=np.array([4.]),grid_id=np.array(p.metadata['grid_id']))
    out=tmp_path/'obs.jsonl';r=export_product(path,geo,out,history_hours=history)
    records=[json.loads(x) for x in out.read_text().splitlines()]
    return records,r['registry']


def test_export_keeps_observed_age_not_processing_time(tmp_path):
    recs,reg=export_case(tmp_path)
    g=build_pyramid(0)[0];v={k:Variable(**x) for k,x in reg.items()}
    packed=pack_observations(recs,g,np.array(PRESSURE_HPA)*100,T+timedelta(hours=48),v)
    assert packed.accepted_records==1 and packed.features[0,1]==4. and packed.slots[0]==0
    assert packed.levels[0]==37 and recs[0]['observed_at']==TS
    assert maximum_history(reg)==168


def test_raw_history_cannot_be_extended():
    with pytest.raises(ValueError):Variable('K',270.,20.,'surface',history_hours=168)


def test_product_history_cannot_be_extended_beyond_method():
    v=variable_spec('cloud_top_height','electro_l','SYNTHETIC',history_hours=24)
    with pytest.raises(ValueError):Variable(**v)


def test_future_stale_bad_qc_and_wrong_product_rejected(tmp_path):
    records,reg=export_case(tmp_path);v={k:Variable(**x) for k,x in reg.items()};g=build_pyramid(0)[0]
    assert pack_observations(records,g,np.array(PRESSURE_HPA)*100,T,v).accepted_records==0
    assert pack_observations(records,g,np.array(PRESSURE_HPA)*100,T+timedelta(hours=169),v).accepted_records==0
    for change in ({'qc':1},{'product_sha256':'bad'},{'geometry_sha256':'bad'}):
        assert pack_observations([{**records[0],**change}],g,np.array(PRESSURE_HPA)*100,T+timedelta(hours=1),v).accepted_records==0
    bad=dict(records[0]);bad['derivation']=dict(bad['derivation'],product='soil_moisture_surface')
    assert pack_observations([bad],g,np.array(PRESSURE_HPA)*100,T+timedelta(hours=1),v).accepted_records==0


def test_product_reaches_adaptive_model_and_72h(tmp_path):
    from global_weather.adaptive import AdaptiveWeatherModel
    a,r=export_case(tmp_path/'a',.6);b,_=export_case(tmp_path/'b',.3)
    grids=build_pyramid(0);v={k:Variable(**x) for k,x in r.items()};issue=T+timedelta(hours=48)
    aa=pack_observations(a,grids[0],np.array(PRESSURE_HPA)*100,issue,v)
    bb=pack_observations(b,grids[0],np.array(PRESSURE_HPA)*100,issue,v)
    torch.manual_seed(31);torch.set_num_threads(1)
    m=AdaptiveWeatherModel(grids,aa.vocabulary,observation_schema=aa.schema_fingerprint,hidden=16,allow_unscaled_synthetic=True)
    z=torch.zeros(12)
    with torch.no_grad():
        sa=m.analysis_state(aa,z,z);sb=m.analysis_state(bb,z,z)
        repeated=m.analysis_state(aa,z,z,background=sa)
        next_state=m.advance_background(sa,issue+timedelta(hours=3),z,z)
        frames=list(m(aa,z,z,horizon_hours=72))
    assert not torch.allclose(sa.latent,sb.latent,atol=1e-7)
    assert torch.equal(sa.latent,repeated.latent) and len(sa.evidence)==len(next_state.evidence)==1
    assert len(frames)==25 and frames[-1].profiles.shape==(12,37,6)


def test_cli_hash_pinned_calculation(tmp_path):
    inputs={'a':field([.6],'surface_reflectance',metadata={'spectral_role':'nir'}),
            'b':field([.2],'surface_reflectance',metadata={'spectral_role':'red'}),
            'eligible':field([1.],'eligibility_mask'),'solar_zenith':field([30.],'solar_zenith_angle','degree')}
    refs={}
    for name,f in inputs.items():
        path=tmp_path/(name+'.npz');save_field(path,f);refs[name]={'path':path.name,'sha256':sha256(path)}
    job={'schema':'satellite-product-job-1','product':'ndvi','data_kind':'synthetic','available_at':READY,
         'availability_reference':'analytic fixture time','inputs':refs}
    path=tmp_path/'job.json';path.write_text(json.dumps(job))
    out=calculate(path,tmp_path/'product.npz')
    assert out['valid_pixels']==1
    (tmp_path/'a.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):calculate(path,tmp_path/'other.npz')


def test_catalog_has_no_claim_of_operational_readiness(capsys):
    main(['catalog']);data=json.loads(capsys.readouterr().out)
    assert data['soil_moisture_surface']['units']=='m3 m-3' and 'ndvi' in data


@pytest.mark.parametrize('kind',['soil','cloud_height','lwp'])
def test_retrieved_products_are_actual_model_features(tmp_path,kind):
    p={'soil':lambda:soil_moisture(*soil_case(),**KW),
       'cloud_height':lambda:cloud_height(*height_args(),**KW),
       'lwp':lambda:liquid_water_path(*lwp_args(),**KW)}[kind]()
    pp=tmp_path/'p.npz';gp=tmp_path/'g.npz';save_product(pp,p)
    np.savez(gp,latitude=np.array([60.]),longitude=np.array([30.]),view_zenith_deg=np.array([20.]),
             footprint_km=np.array([40.]),grid_id=np.array(p.metadata['grid_id']))
    report=export_product(pp,gp,tmp_path/'r.jsonl')
    records=[json.loads(t) for t in (tmp_path/'r.jsonl').read_text().splitlines()]
    v={k:Variable(**r) for k,r in report['registry'].items()};grid=build_pyramid(0)[0]
    packed=pack_observations(records,grid,np.array(PRESSURE_HPA)*100,T+timedelta(hours=1),v)
    assert packed.accepted_records==1 and packed.levels[0]==(37 if kind=='soil' else -1)
    if kind=='soil':
        v={k:replace(x,product_depth_m=.05) for k,x in v.items()}
        assert pack_observations(records,grid,np.array(PRESSURE_HPA)*100,T+timedelta(hours=1),v).accepted_records==0
