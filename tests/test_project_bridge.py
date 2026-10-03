"""Synthetic files built to the audited producer schemas; not captured satellite data."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import numpy as np
import pytest
from global_weather.connectors.project_bridge import (decode_cbor, fingerprint, load_json, local_file,
    local_directory, publish_json, scan_arktika, scan_satdump, ARKTIKA_REV, SATDUMP_REV)
from global_weather.connectors.physical_bridge import export_geotiff
from global_weather.connectors.preflight import blob_sha, check_source, check_agent_contracts

T=datetime(2026,10,3,12,tzinfo=timezone.utc)

def put(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj),encoding='utf-8');return path


def cbor(x):
    """Independent tiny test encoder for nlohmann's definite JSON subset."""
    def head(major,n):
        if n<24:return bytes([(major<<5)|n])
        size=1 if n<256 else 2 if n<65536 else 4 if n<2**32 else 8
        return bytes([(major<<5)|{1:24,2:25,4:26,8:27}[size]])+n.to_bytes(size,'big')
    if x is None:return b'\xf6'
    if type(x) is bool:return b'\xf5' if x else b'\xf4'
    if type(x) is int:return head(0,x) if x>=0 else head(1,-1-x)
    if isinstance(x,float):return b'\xfb'+struct.pack('>d',x)
    if isinstance(x,str):b=x.encode();return head(3,len(b))+b
    if isinstance(x,list):return head(4,len(x))+b''.join(cbor(v) for v in x)
    if isinstance(x,dict):return head(5,len(x))+b''.join(cbor(k)+cbor(v) for k,v in x.items())
    raise TypeError(type(x))

@pytest.mark.parametrize('obj',[{'a':1},{'b':[False,True,None,23,24,256,-2,1.25,'Арктика']},{'nested':{'x':[]}}])
def test_native_cbor_subset(obj):assert decode_cbor(cbor(obj))==obj

@pytest.mark.parametrize('raw',[b'\xa1aa',b'\xa0x',b'\x9f\xff',b'\xc0\xa0',b'\xa2aa\x01aa\x02',b'\xa1\x00\x01'])
def test_invalid_cbor_closed(raw):
    with pytest.raises(ValueError):decode_cbor(raw)

def test_cbor_depth_and_count():
    with pytest.raises(ValueError):decode_cbor(cbor({'a':[[[1]]]}),max_depth=2)
    with pytest.raises(ValueError):decode_cbor(cbor({'a':[1,2,3]}),max_nodes=3)

def arktika_fixture(tmp):
    root=tmp/'producer';root.mkdir()
    raster=root/'channel.tif';raster.write_bytes(b'SYNTHETIC-NOT-A-TIFF')
    a=dict(id='asset1',item_id='item1',platform='ARCM1',time='2026-10-03T11:00:00Z',
           time_assumed=False,channel=9,category='channel',level='L2IR',epsg=4326,
           raster_bands=[{'unit':'K','scale':1,'offset':0}],uri='https://example.invalid/signed?SECRET=never-copy')
    receipt=dict(asset_id='asset1',size=raster.stat().st_size,sha256=fingerprint(raster),time='2026-10-03T11:10:00Z')
    put(str(raster)+'.download.json',receipt)
    db=tmp/'catalog.sqlite'
    with sqlite3.connect(db) as c:
        c.executescript('CREATE TABLE assets(id TEXT, item_id TEXT,platform TEXT,stamp TEXT,data TEXT);'
          'CREATE TABLE jobs(id TEXT,asset_id TEXT,state TEXT,done INTEGER,total INTEGER,path TEXT,error TEXT,sha256 TEXT,created TEXT);'
          'CREATE TABLE settings(key TEXT,value TEXT);')
        c.execute('INSERT INTO assets VALUES(?,?,?,?,?)',('asset1','item1','ARCM1',a['time'],json.dumps(a)))
        c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)',('asset1','asset1','done',1,1,str(raster),'',receipt['sha256'],''))
        c.execute('INSERT INTO settings VALUES(?,?)',('token','SECRET_NEVER_READ'))
    return db,root,a,raster

def test_arktika_native_catalog_is_readonly_and_no_secrets(tmp_path):
    db,root,a,r=arktika_fixture(tmp_path);before=fingerprint(db)
    out=scan_arktika(db,root)
    assert fingerprint(db)==before and out['records'][0]['sha256']==fingerprint(r)
    assert out['records'][0]['available_at']=='2026-10-03T11:10:00Z'
    assert out['records'][0]['blockers']==['physical_export_review_required']
    assert not out['training_ready'] and out['actual_producer_revision'] is None
    assert 'SECRET' not in json.dumps(out)

@pytest.mark.parametrize('change',['bytes','receipt','partial','unsafe','assumed','electro'])
def test_arktika_incompatibilities_not_hidden(tmp_path,change):
    db,root,a,r=arktika_fixture(tmp_path)
    with sqlite3.connect(db) as c:
        if change=='bytes':r.write_bytes(b'TAMPERED')
        elif change=='receipt':Path(str(r)+'.download.json').unlink()
        elif change=='partial':c.execute("UPDATE jobs SET state='running'")
        elif change=='unsafe':c.execute("UPDATE jobs SET path='../secret'")
        else:
            if change=='assumed':a['time_assumed']=True
            if change=='electro':a['platform']='ELEKTRO-L-3'
            c.execute('UPDATE assets SET data=?',(json.dumps(a),))
    blockers=scan_arktika(db,root)['records'][0]['blockers']
    required={'bytes':'missing_unsafe_or_unverified_download','receipt':'missing_unsafe_or_unverified_download',
              'partial':'download_not_complete','unsafe':'missing_unsafe_or_unverified_download',
              'assumed':'observation_timezone_assumed','electro':'unsupported_platform_in_arktika_catalog'}
    assert required[change] in blockers


def satdump_fixture(tmp, inst='msu_mr', count=6, platform='METEOR-M2-3'):
    root=tmp/'satdump';directory=root/'PRODUCT';directory.mkdir(parents=True)
    product=dict(type='image',instrument=inst,has_timestamps=True,timestamps_type=0,
                 timestamps=[T.timestamp()-3600,T.timestamp()-3590],bit_depth=16,
                 images=[dict(file=f'channel-{i}.png',name=str(i)) for i in range(1,count+1)],
                 projection_cfg={'type':'SYNTHETIC'},needs_correlation=False)
    if inst=='mtvza':product.update(channel_layout='hrpt30',decode_quality={'status':'partial'})
    for image in product['images']:(directory/image['file']).write_bytes(b'SYNTHETIC-PIXELS')
    (directory/'product.cbor').write_bytes(cbor(product))
    put(root/'dataset.json',dict(satellite=platform,timestamp=T.timestamp()-3600,products=['PRODUCT']))
    put(root/'decode-status.json',dict(instruments=[dict(instrument=inst,status='ok')]))
    return root,product

@pytest.mark.parametrize('inst,count',[('msu_mr',6),('mtvza',30),('msu_gs',10)])
def test_satdump_native_metadata_recognized(tmp_path,inst,count):
    root,prod=satdump_fixture(tmp_path,inst,count)
    out=scan_satdump(root);r=out['records'][0]
    assert len(r['channels'])==count and r['instrument']==inst
    assert r['channels'][0]['time']['count']==2 and r['projection_present']
    assert not r['calibration_present'] and r['available_at'] is None and not out['training_ready']
    if inst=='mtvza':assert 'verify_hrpt30_physical_mapping' in r['blockers'] and r['decode_status']=='partial'

@pytest.mark.parametrize('change',['missing','path','timestamps','matrix','correlation','no_data','bad_quality'])
def test_satdump_edge_cases(tmp_path,change):
    root,p=satdump_fixture(tmp_path)
    if change=='missing':(root/'PRODUCT/channel-1.png').unlink()
    if change=='path':p['images'][0]['file']='../../secret'
    if change=='timestamps':p['timestamps']=[-1,float('nan'),T.timestamp()]
    if change=='matrix':
        p['save_as_matrix']=True
        for img in p['images'][1:]:(root/'PRODUCT'/img['file']).unlink()
    if change=='correlation':p['needs_correlation']=True
    if change=='no_data':p['decode_quality']={'status':'no_data'}
    if change=='bad_quality':p['decode_quality']='bad'
    (root/'PRODUCT/product.cbor').write_bytes(cbor(p))
    r=scan_satdump(root)['records'][0]
    if change in ('missing','path'):assert 'missing_or_unsafe_channel_file' in r['blockers']
    if change=='timestamps':assert r['channels'][0]['time']['invalid']==2
    if change=='matrix':assert 'matrix_unpack_required' in r['blockers'] and 'missing_or_unsafe_channel_file' not in r['blockers']
    if change=='correlation':assert 'channel_correlation_required' in r['blockers']
    if change=='no_data':assert 'no_complete_scans' in r['blockers']
    if change=='bad_quality':assert 'invalid_native_metadata' in r['blockers']

def test_report_no_overwrite(tmp_path):
    p=tmp_path/'r.json';publish_json(p,{'ok':1})
    with pytest.raises(FileExistsError):publish_json(p,{'ok':2})
    assert load_json(p)=={'ok':1}

def test_symlinks_and_parent_paths_rejected(tmp_path):
    p=tmp_path/'one';p.write_text('one');(tmp_path/'link').symlink_to(p)
    with pytest.raises(ValueError):local_file(tmp_path/'link')
    with pytest.raises(ValueError):local_file(tmp_path/'..'/'one')
    (tmp_path/'folder').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):local_directory(tmp_path/'folder')


def physical_fixture(tmp, source='arktika_m'):
    rio=pytest.importorskip('rasterio')
    if source=='arktika_m':
        db,root,a,original=arktika_fixture(tmp)
        native=put(root/'native.json',a);inst='MSU-GS/A';platform='ARCM1';channel='9'
    else:
        root,product=satdump_fixture(tmp,'msu_gs',10,platform='ELEKTRO-L-3')
        native=root/'PRODUCT/product.cbor';inst='msu_gs';platform='ELEKTRO-L-3';channel='9'
    raster=root/'tile.tif';transform=rio.transform.from_origin(20.,60.,.1,.1)
    with rio.open(raster,'w',driver='GTiff',width=2,height=2,count=1,dtype='uint16',crs='EPSG:4326',transform=transform,nodata=0) as f:
        f.write(np.array([[250,260],[0,270]],dtype=np.uint16),1)
    geom=root/'geometry.npz'
    def geometry(**updates):
        g=dict(observed_at_unix=np.full((2,2),T.timestamp()-3600),view_zenith_deg=np.full((2,2),25.),
               footprint_km=np.full((2,2),4.),valid=np.array([[True,False],[True,True]]),
               grid_transform=np.array(tuple(transform)[:6]),grid_crs=np.array('EPSG:4326'))
        g.update(updates);np.savez(geom,**g);return fingerprint(geom)
    review=dict(schema='global-weather.physical-raster/1',validation_status='reviewed',data_kind='synthetic',
                reviewer='TEST ONLY',calibration_reference='SYNTHETIC declared scale',geometry_reference='SYNTHETIC geometry',
                time_reference='SYNTHETIC time log',quality_reference='SYNTHETIC mask',license='synthetic fixture',
                producer=dict(repository='f2re/arktika-worker' if source=='arktika_m' else 'f2re/SatDump',revision=ARKTIKA_REV if source=='arktika_m' else SATDUMP_REV),
                source=source,platform=platform,channel_id=channel,instrument=inst,variable='IR9',
                native_metadata=str(native.relative_to(root)),native_metadata_sha256=fingerprint(native),
                raster='tile.tif',raster_sha256=fingerprint(raster),geometry='geometry.npz',geometry_sha256=geometry(),
                calibration_status='declared',quantity='brightness_temperature',units='K',scale=1.,offset=0.,
                channel_mapping_verified=True,physical_valid_range=[120.,400.],
                available_at='2026-10-03T11:20:00Z',download_completed_at='2026-10-03T11:10:00Z',
                observation_id_prefix='SYNTHETIC:scene:grid',pixel_origin=[10,20],revision=0)
    if source=='arktika_m':review.update(asset_id='asset1',original_raster='channel.tif',download_receipt='channel.tif.download.json',download_receipt_sha256=fingerprint(str(original)+'.download.json'))
    else:review.update(dataset_metadata='dataset.json',dataset_metadata_sha256=fingerprint(root/'dataset.json'),platform_family=source)
    rp=put(tmp/'review.json',review)
    return root,rp,review,geometry

@pytest.mark.parametrize('source',['arktika_m','electro_l'])
def test_real_contract_to_model_roundtrip_on_synthetic_raster(tmp_path,source):
    from global_weather.grid import build_grid
    from global_weather.observations import pack_observations, Variable
    from global_weather.vertical import PRESSURE_HPA
    root,rp,r,g=physical_fixture(tmp_path,source);out=tmp_path/'out.jsonl'
    summary=export_geotiff(rp,root,out,issue_time=T.isoformat())
    rows=[json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows)==2 and [x['value'] for x in rows]==[250.,270.]
    assert rows[0]['observation_id'].endswith(':9:10:20')
    assert summary['data_kind']=='synthetic' and summary['training_ready'] is False
    registry={'IR9':Variable('K',250.,50.,'column',source=source,platform=r['platform'],channel_id='9')}
    packed=pack_observations(rows,build_grid(1),np.array(PRESSURE_HPA)*100,T,registry)
    assert packed.accepted_records==2

@pytest.mark.parametrize('field,value',[('calibration_status','assumed'),('channel_mapping_verified',False),
  ('quantity','raw_counts'),('available_at','2026-10-03T12:01:00Z'),('platform','ARCM2'),
  ('channel_id','8'),('raster_sha256','0'*64),('native_metadata_sha256','0'*64),('scale',-1),
  ('revision',-1),('instrument','wrong'),('time_reference',''),('source','meteor_mtvza')])
def test_physical_contract_refuses_unsafe_admission(tmp_path,field,value):
    root,rp,r,g=physical_fixture(tmp_path);r[field]=value;put(rp,r)
    with pytest.raises(ValueError):export_geotiff(rp,root,tmp_path/'out.jsonl',issue_time=T.isoformat())
    assert not (tmp_path/'out.jsonl').exists()

def test_no_future_pixels_or_guessed_geometry(tmp_path):
    root,rp,r,g=physical_fixture(tmp_path)
    r['geometry_sha256']=g(observed_at_unix=np.full((2,2),T.timestamp()+1));put(rp,r)
    with pytest.raises(ValueError):export_geotiff(rp,root,tmp_path/'out',issue_time=T.isoformat())
    r['geometry_sha256']=g(grid_crs=np.array('EPSG:3857'));put(rp,r)
    with pytest.raises(ValueError):export_geotiff(rp,root,tmp_path/'out',issue_time=T.isoformat())

def test_no_export_into_producer_or_overwrite(tmp_path):
    root,rp,r,g=physical_fixture(tmp_path)
    with pytest.raises(ValueError):export_geotiff(rp,root,root/'out.jsonl',issue_time=T.isoformat())
    out=tmp_path/'out.jsonl';export_geotiff(rp,root,out,issue_time=T.isoformat())
    with pytest.raises(FileExistsError):export_geotiff(rp,root,out,issue_time=T.isoformat())

def test_reflectance_night_mask(tmp_path):
    root,rp,r,g=physical_fixture(tmp_path)
    a=load_json(root/'native.json');a['channel']=1;put(root/'native.json',a)
    r.update(channel_id='1',quantity='reflectance',units='1',scale=.001,physical_valid_range=[0.,1.],native_metadata_sha256=fingerprint(root/'native.json'))
    r['geometry_sha256']=g(solar_zenith_deg=np.array([[45.,50.],[50.,100.]]));put(rp,r)
    out=tmp_path/'out';report=export_geotiff(rp,root,out,issue_time=T.isoformat())
    assert report['records']==1 and json.loads(out.read_text())['value']==pytest.approx(.25)

def test_preflight_detects_changed_source_contract(tmp_path):
    root=tmp_path/'repo';root.mkdir();p=root/'x.py';p.write_text('x=1\n')
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    git('init','-q');git('add','x.py')
    git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
    spec=dict(repository='SYNTHETIC',revision=git('rev-parse','HEAD'),files={'x.py':blob_sha(p)})
    assert check_source(root,spec)['passed']
    p.write_text('x=2\n');assert not check_source(root,spec)['passed']

def test_agent_checker_fails_empty_project(tmp_path):
    assert not check_agent_contracts(tmp_path)['passed']


def test_disguised_vrt_does_not_become_geotiff(tmp_path):
    root,rp,r,g=physical_fixture(tmp_path)
    (root/'tile.tif').write_text('<VRTDataset rasterXSize="2" rasterYSize="2"></VRTDataset>')
    r['raster_sha256']=fingerprint(root/'tile.tif');put(rp,r)
    with pytest.raises(Exception):export_geotiff(rp,root,tmp_path/'out',issue_time=T.isoformat())
    assert not (tmp_path/'out').exists()


def test_auxiliary_masks_are_not_silently_ignored(tmp_path):
    root,rp,r,g=physical_fixture(tmp_path)
    (root/'tile.tif.msk').write_bytes(b'UNREVIEWED')
    with pytest.raises(ValueError,match='GeoTIFF'):export_geotiff(rp,root,tmp_path/'out',issue_time=T.isoformat())


def test_reports_reject_parent_traversal(tmp_path):
    with pytest.raises(ValueError):publish_json(tmp_path/'..'/'report.json',{})
