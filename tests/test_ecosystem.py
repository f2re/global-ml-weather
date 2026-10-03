"""Schema-derived SYNTHETIC fixtures; not real satellite acquisition validation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
import pytest
from global_weather.connectors.native_cbor import loads
from global_weather.connectors.ecosystem import (inspect_satdump, arktika_product, gptl_asset, sha256, read_json, child)
from global_weather.connectors.raster_bridge import import_raster, export_observations, read_capsule
from global_weather.compatibility import check_release_evidence, check_checkout, main


def write(path, value):
    path.write_text(json.dumps(value),encoding='utf-8');return path


def cbor(value):
    def h(t,n):
        if n<24:return bytes([(t<<5)|n])
        if n<256:return bytes([(t<<5)|24,n])
        if n<65536:return bytes([(t<<5)|25])+struct.pack('>H',n)
        return bytes([(t<<5)|26])+struct.pack('>I',n)
    if value is None:return b'\xf6'
    if type(value)==bool:return b'\xf5' if value else b'\xf4'
    if type(value)==int:return h(0,value) if value>=0 else h(1,-1-value)
    if type(value)==float:return b'\xfb'+struct.pack('>d',value)
    if isinstance(value,str):
        b=value.encode();return h(3,len(b))+b
    if isinstance(value,list):return h(4,len(value))+b''.join(map(cbor,value))
    if isinstance(value,dict):return h(5,len(value))+b''.join(cbor(k)+cbor(v) for k,v in value.items())
    raise TypeError()


def native(tmp,layout='hrpt30'):
    d=tmp/'MTVZA';d.mkdir();n=30 if layout=='hrpt30' else 46
    images=[]
    for i in range(n):
        # Transport presence fixture, deliberately not an image validation claim.
        (d/f'MTVZA-{i+1}.png').write_bytes(b'SYNTHETIC_TRANSPORT_ONLY')
        images.append(dict(file=f'MTVZA-{i+1}.png',name=str(i+1),ifov_x=-1,ifov_y=-1))
    product=dict(instrument='mtvza',type='image',channel_layout=layout,has_timestamps=False,
                 images=images,bit_depth=16,needs_correlation=False)
    (d/'product.cbor').write_bytes(cbor(product))
    write(d/'processing-status.json',dict(instrument='mtvza',status='done',generated=5))
    dataset=write(tmp/'dataset.json',dict(satellite='METEOR-M2-4',timestamp=-1,products=['MTVZA']))
    return dataset,d,product


@pytest.mark.parametrize('layout,n',[('hrpt30',30),('dump46',46)])
def test_native_layouts_are_distinct_and_never_kelvin(tmp_path,layout,n):
    path,_,_=native(tmp_path,layout);r=inspect_satdump(path,required_instruments=['mtvza'])
    assert r['observed_at'] is None and not r['model_ready']
    assert r['products'][0]['channel_count']==n and r['products'][0]['transport_complete']
    assert 'microwave_calibration_required' in r['products'][0]['reasons']


def test_missing_required_instrument_is_blocked(tmp_path):
    p,_,_=native(tmp_path);r=inspect_satdump(p,required_instruments=['msu_mr'])
    assert r['status']=='blocked' and r['missing_required_instruments']==['msu_mr']


def test_missing_product_does_not_hide_good_instrument(tmp_path):
    p,_,_=native(tmp_path);v=read_json(p);v['products'].append('MISSING');write(p,v)
    r=inspect_satdump(p);assert len(r['products'])==1 and r['failures']


def test_missing_channel_blocks_transport(tmp_path):
    p,d,_=native(tmp_path);(d/'MTVZA-1.png').unlink()
    assert not inspect_satdump(p)['products'][0]['transport_complete']


def test_cbor_roundtrip_and_bounds():
    v=dict(a=[1,-1,2.5,False,None],b='Арктика');assert loads(cbor(v))==v
    for b in (b'\xbf\xff',b'\xa1\x61a\xfb'+struct.pack('>d',float('nan')),cbor(v)+b'x',b'\xc1\x00'):
        with pytest.raises(ValueError):loads(b)
    with pytest.raises(ValueError):loads(cbor(v),max_items=2)
    with pytest.raises(ValueError):loads(cbor(v),max_bytes=2)
    with pytest.raises(ValueError):loads(b'\xa2\x61a\x00\x61a\x01')


def test_duplicate_json_rejected(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{"time":1,"time":2}')
    with pytest.raises(ValueError):read_json(p)


@pytest.mark.parametrize('name',['../secret','/etc/passwd','https://host/file','\\server\\file'])
def test_no_escaping_paths(tmp_path,name):
    with pytest.raises(ValueError):child(tmp_path,name)


def test_symlink_inputs_rejected(tmp_path):
    p=tmp_path/'real';p.write_bytes(b'x');(tmp_path/'link').symlink_to(p)
    with pytest.raises(ValueError):sha256(tmp_path/'link')


def raster_fixture(tmp):
    rio=pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    profile=dict(driver='GTiff',width=2,height=2,count=1,crs='EPSG:4326',transform=from_origin(20,60,1,1))
    with rio.open(tmp/'values.tif','w',dtype='float32',nodata=-9999,**profile) as ds:
        ds.write(np.array([[250,260],[270,-9999]],dtype='float32'),1);ds.set_band_unit(1,'K')
        ds.update_tags(time='2020-01-01T00:00:00Z')
    with rio.open(tmp/'quality.tif','w',dtype='uint8',**profile) as ds:ds.write(np.array([[0,4],[0,1]],dtype='uint8'),1)
    return write(tmp/'product.json',dict(id='synthetic',scene_id='synthetic_scene',platform='ARCM1',time='2020-01-01T00:00:00Z',
        product='channel',time_assumed=False,calibration_status='declared',request={'channel':9},
        legend=dict(units='K',calibration=[dict(channel=9,units='K',status='declared',reference='SYNTHETIC_ONLY',scale=2.,offset=10.)])))


def import_fixture(tmp,geometry=True):
    p=raster_fixture(tmp);geo=None
    if geometry:geo=write(tmp/'geometry.json',dict(reference='SYNTHETIC_GEOMETRY_NOT_REAL',view_zenith_deg=30.,footprint_km=4.))
    out=tmp/'capsule';spec=arktika_product(p)
    m=import_raster(spec,out,available_at='2020-01-01T01:00:00Z',availability_reference='SYNTHETIC_EVENT',geometry=geo)
    return out,m


def test_values_already_calibrated_are_not_scaled_twice(tmp_path):
    out,m=import_fixture(tmp_path);_,a=read_capsule(out)
    assert sorted(a['values'][a['valid']].tolist())==[250.,270.]  # not 510/550
    assert m['valid_pixels']==2 and not m['model_ready'] and m['observation_export_ready']
    assert a['latitude'][0,0]==pytest.approx(59.5) and a['longitude'][0,0]==pytest.approx(20.5)


def test_export_matches_existing_observation_contract(tmp_path):
    out,m=import_fixture(tmp_path);target=tmp_path/'observations.jsonl'
    r=export_observations(out,target);rows=[json.loads(v) for v in target.read_text().splitlines()]
    assert len(rows)==2 and rows[0]['radiometry']['quantity']=='brightness_temperature'
    from global_weather.grid import build_grid
    from global_weather.observations import pack_observations,Variable
    from global_weather.vertical import PRESSURE_HPA
    v=Variable('K',250.,30.,'column',source='arktika_m',platform='ARCM1',channel_id='9')
    obs=pack_observations(rows,build_grid(0),np.array(PRESSURE_HPA)*100,
        datetime(2020,1,1,1,tzinfo=timezone.utc),{r['variable']:v})
    assert obs.accepted_records==2


def test_missing_geometry_stages_but_blocks_model_export(tmp_path):
    out,m=import_fixture(tmp_path,False)
    assert not m['observation_export_ready']
    with pytest.raises(ValueError):export_observations(out,tmp_path/'out.jsonl')


@pytest.mark.parametrize('field,value',[('time_assumed',True),('calibration_status','assumed'),('product','difference'),('product','rgb')])
def test_native_assumptions_and_derived_products_blocked(tmp_path,field,value):
    p=raster_fixture(tmp_path);v=read_json(p);v[field]=value;write(p,v)
    with pytest.raises(ValueError):arktika_product(p)


def test_cannot_backdate_availability(tmp_path):
    p=raster_fixture(tmp_path)
    with pytest.raises(ValueError):import_raster(arktika_product(p),tmp_path/'c',available_at='2019-12-31T23:00:00Z',availability_reference='x')


def test_capsule_tampering_detected(tmp_path):
    out,_=import_fixture(tmp_path);(out/'pixels.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):read_capsule(out)


def test_no_overwrite_or_silent_downsampling(tmp_path):
    out,_=import_fixture(tmp_path)
    with pytest.raises(ValueError):export_observations(out,tmp_path/'o',max_records=1)
    export_observations(out,tmp_path/'o')
    with pytest.raises(ValueError):export_observations(out,tmp_path/'o')


def test_gptl_asset_uses_download_receipt_and_scale(tmp_path):
    raster_fixture(tmp_path);r=tmp_path/'values.tif'
    write(Path(str(r)+'.download.json'),dict(asset_id='asset',size=r.stat().st_size,sha256=sha256(r),time='2020-01-01T01:00:00Z'))
    asset=dict(id='asset',item_id='item',filename='values.tif',platform='ELECTRO-L-2',category='channel',level='L2IR',
               time='2020-01-01T00:00:00Z',time_assumed=False,channel=9,uri='https://example.org/source.tif?secret=hidden',
               raster_bands=[dict(unit='K',scale=2.,offset=10.)])
    p=write(tmp_path/'asset.json',asset);spec=gptl_asset(p,r,source='electro_l')
    assert spec['source_uri']=='https://example.org/source.tif'
    out=tmp_path/'capsule'
    import_raster(spec,out,available_at='2020-01-01T02:00:00Z',availability_reference='SYNTHETIC_RECEIPT')
    _,a=read_capsule(out);assert a['values'][0,0]==510.
    with pytest.raises(ValueError):gptl_asset(p,r,source='arktika_m')
    j=read_json(Path(str(r)+'.download.json'));j['sha256']='0'*64;write(Path(str(r)+'.download.json'),j)
    with pytest.raises(ValueError):gptl_asset(p,r,source='electro_l')


def test_unknown_gptl_level_not_guessed(tmp_path):
    r=tmp_path/'file.tif';r.write_bytes(b'x')
    a=dict(item_id='i',filename='file.tif',raster_bands=[],platform='EL2',level='L3BT9')
    with pytest.raises(ValueError):gptl_asset(write(tmp_path/'a.json',a),r,source='electro_l')


def test_release_gate_requires_hashed_matching_reports(tmp_path):
    m=dict(schema='release-evidence-v1',commit='a'*40,executor='synthetic_executor',reviewer='synthetic_reviewer',forecast_skill_claimed=False,checks={})
    manifest=write(tmp_path/'release.json',m);assert check_release_evidence(manifest)['status']=='blocked'
    for k in ('pytest','baseline_72h','adaptive_72h','browser','ecosystem_contracts','physics_review'):
        p=write(tmp_path/(k+'.json'),dict(check=k,commit='a'*40,status='passed',fixture='SYNTHETIC'))
        m['checks'][k]=dict(report=p.name,sha256=sha256(p))
    write(manifest,m);assert check_release_evidence(manifest)['status']=='engineering_evidence_complete'
    m['reviewer']=m['executor'];write(manifest,m);assert check_release_evidence(manifest)['status']=='blocked'


def test_checkout_drift_is_not_compatible(tmp_path):
    subprocess.run(['git','init','-b','main',str(tmp_path)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','--allow-empty','-m','synthetic'],check=True,capture_output=True)
    assert check_checkout(tmp_path,'arktika-worker')['status']=='review_required'


def test_cli_has_explicit_nonzero_gate(tmp_path,capsys):
    p=write(tmp_path/'report.json',{})
    assert main(['release-gate',str(p)])==2
    assert json.loads(capsys.readouterr().out)['forecast_quality_certified'] is False


def test_raw_stac_uses_same_identity_as_worker(tmp_path):
    import hashlib
    raster_fixture(tmp_path);r=tmp_path/'values.tif';uri='s3://bucket/EL2/ch9.tif'
    identity=hashlib.sha256(('native-item\n'+uri).encode()).hexdigest()[:32]
    write(Path(str(r)+'.download.json'),dict(asset_id=identity,size=r.stat().st_size,sha256=sha256(r),time='2020-01-01T01:00:00Z'))
    stac=write(tmp_path/'item.json',dict(type='Feature',id='native-item',properties={'platform':'EL2','datetime':'2020-01-01T00:00:00Z','processing:level':'L2IR'},
              assets={'ch9':{'href':uri,'raster:bands':[dict(unit='K',scale=1.,offset=0.)]}}))
    spec=gptl_asset(stac,r,source='electro_l',asset_key='ch9',channel=9)
    assert spec['channel_id']=='9' and spec['source']=='electro_l'
    with pytest.raises(ValueError):gptl_asset(stac,r,source='electro_l')


def test_geometry_shape_is_not_broadcast_silently(tmp_path):
    p=raster_fixture(tmp_path);g=write(tmp_path/'geo.json',dict(reference='SYNTHETIC',view_zenith_deg=[0,20],footprint_km=4.))
    with pytest.raises(ValueError):import_raster(arktika_product(p),tmp_path/'out',available_at='2020-01-01T01:00:00Z',availability_reference='TEST',geometry=g)


def test_quality_raster_alignment_required(tmp_path):
    p=raster_fixture(tmp_path);rio=pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    with rio.open(tmp_path/'quality.tif','r+') as d:d.transform=from_origin(0,30,1,1)
    with pytest.raises(ValueError):import_raster(arktika_product(p),tmp_path/'out',available_at='2020-01-01T01:00:00Z',availability_reference='TEST')


def test_future_message_still_rejected_by_core(tmp_path):
    out,_=import_fixture(tmp_path);f=tmp_path/'r.jsonl';r=export_observations(out,f)
    from global_weather.grid import build_grid
    from global_weather.observations import pack_observations,Variable
    from global_weather.vertical import PRESSURE_HPA
    rows=[json.loads(s) for s in f.read_text().splitlines()]
    v=Variable('K',250.,30.,'column',source='arktika_m',platform='ARCM1',channel_id='9')
    o=pack_observations(rows,build_grid(0),np.array(PRESSURE_HPA)*100,datetime(2020,1,1,0,tzinfo=timezone.utc),{r['variable']:v})
    assert o.accepted_records==0


def test_empty_native_dataset_is_blocked(tmp_path):
    p=write(tmp_path/'dataset.json',dict(satellite='SYNTHETIC',timestamp=-1,products=[]))
    assert inspect_satdump(p)['status']=='blocked'


def test_failed_processing_not_hidden_by_existing_images(tmp_path):
    p,d,_=native(tmp_path)
    write(d/'processing-status.json',dict(instrument='mtvza',status='failed'))
    r=inspect_satdump(p)
    assert r['status']=='blocked' and not r['products'][0]['transport_complete']


def test_duplicate_native_channel_ids_block_transport(tmp_path):
    p,d,m=native(tmp_path);m['images'][1]['name']=m['images'][0]['name']
    (d/'product.cbor').write_bytes(cbor(m))
    assert inspect_satdump(p)['status']=='blocked'


def test_capsule_bad_coordinates_even_with_matching_hash_rejected(tmp_path):
    out,m=import_fixture(tmp_path);_,a=read_capsule(out)
    a['latitude'][a['valid']]=91
    np.savez_compressed(out/'pixels.npz',**a)
    m['arrays_sha256']=sha256(out/'pixels.npz');write(out/'manifest.json',m)
    with pytest.raises(ValueError,match='geographical'):read_capsule(out)


def test_malformed_release_evidence_remains_blocked(tmp_path):
    p=write(tmp_path/'evidence.json',dict(commit=17,checks=[]))
    assert check_release_evidence(p)['status']=='blocked'


def test_asset_nodata_is_masked_before_scale_and_offset(tmp_path):
    rio=pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    r=tmp_path/'channel.tif'
    with rio.open(r,'w',driver='GTiff',width=2,height=1,count=1,dtype='uint16',crs='EPSG:4326',transform=from_origin(20,60,1,1)) as ds:
        ds.write(np.array([[0,100]],dtype='uint16'),1)
    write(Path(str(r)+'.download.json'),dict(asset_id='a',size=r.stat().st_size,sha256=sha256(r),time='2020-01-01T01:00:00Z'))
    a=write(tmp_path/'asset.json',dict(id='a',item_id='i',filename=r.name,platform='EL2',time='2020-01-01T00:00:00Z',time_assumed=False,
        level='L2IR',category='channel',channel=9,uri='s3://bucket/channel.tif',raster_bands=[dict(unit='K',scale=.1,offset=250.,nodata=0)]))
    spec=gptl_asset(a,r,source='electro_l')
    out=tmp_path/'out';import_raster(spec,out,available_at='2020-01-01T02:00:00Z',availability_reference='SYNTHETIC')
    _,arrays=read_capsule(out)
    assert arrays['valid'].tolist()==[[False,True]] and arrays['values'][0,1]==260.
