"""Реальный код чтения капсул; входы синтетические, не проходы спутника."""
from dataclasses import replace
from datetime import timedelta
import json
import numpy as np
import pytest
from global_weather.grid import build_pyramid
from global_weather.multimodal.fixture import fixture
from global_weather.multimodal.io import save_sensors,reference,write_json,sha256,load_sequence
from global_weather.multimodal.prepare import from_capsules


def create_case(root,source_index=0,missing=False,legacy=False):
    sensors,obs=fixture(build_pyramid(0));sensor=sensors[source_index]
    raw=replace(sensor,channels=tuple(replace(c,mean=None,std=None) for c in sensor.channels),
                normalization_sha256=None,fit_end=None)
    save_sensors(root/'sensors.json',[raw]);frames=[]
    rows,cols=np.indices((8,8),dtype=float);latitude=64.-.1*(rows+.5);longitude=30.+.1*(cols+.5)
    for t in range(2):
        entries={};when=(obs.issue_time-timedelta(hours=2-t)).isoformat()
        for j,ch in enumerate(raw.channels):
            if missing and j==1:
                entries[ch.id]=None;continue
            d=root/f'frame-{t}-channel-{j}';d.mkdir()
            p=d/'pixels.npz';np.savez(p,values=np.full((8,8),250.+t+j),valid=np.ones((8,8),bool),
                latitude=latitude,longitude=longitude,view_zenith_deg=np.full((8,8),20.),footprint_km=np.full((8,8),4.))
            m=dict(schema='physical-raster-v1',quantity='brightness_temperature',units='K',source=raw.source,
                   platform=raw.platform,channel_id=ch.id,calibration_reference=ch.calibration_id,
                   geometry_reference='analytic coordinates, not real orbital geometry',crs='EPSG:4326',
                   transform=[.1,0,30.,0,-.1,64.],shape=[8,8],observed_at=when,available_at=when,
                   arrays_sha256=sha256(p),valid_pixels=64)
            if not legacy:m.update(data_kind='synthetic',time_support='frame')
            write_json(d/'manifest.json',m)
            entries[ch.id]={'manifest':reference(root,d/'manifest.json'),'pixels':reference(root,p)}
        frames.append({'channels':entries})
    plan={'schema':'multimodal-capsule-plan-1','sensor':raw.id,'sensor_registry':reference(root,root/'sensors.json'),
          'frames':frames,'data_kind':'synthetic'}
    write_json(root/'plan.json',plan)
    return raw,plan


@pytest.mark.parametrize('source_index',[0,1,2])
def test_physical_capsules_prepare_without_double_scaling(tmp_path,source_index):
    s,p=create_case(tmp_path,source_index)
    report=from_capsules(tmp_path/'plan.json',tmp_path/'seq.npz')
    seq=load_sequence(tmp_path/'seq.npz',s)
    assert report['frames']==2 and seq.values[0,0,0,0]==250.
    assert seq.area_m2.min()>0 and seq.channel_ids==tuple(c.id for c in s.channels)


def test_missing_channel_is_kept_as_mask(tmp_path):
    s,p=create_case(tmp_path,missing=True)
    from_capsules(tmp_path/'plan.json',tmp_path/'seq.npz');seq=load_sequence(tmp_path/'seq.npz',s)
    assert not seq.valid[:,1].any() and seq.valid[:,0].all()


def test_legacy_capsule_requires_separate_immutable_declarations(tmp_path):
    s,p=create_case(tmp_path,legacy=True)
    with pytest.raises(ValueError,match='происхождения'):from_capsules(tmp_path/'plan.json',tmp_path/'seq.npz')
    write_json(tmp_path/'declaration.json',{'data_kind':'synthetic','timing':'analytically simultaneous frame'})
    ref=reference(tmp_path,tmp_path/'declaration.json')
    p['declarations']={'data_kind_reference':ref,'time_support':'frame','time_support_reference':ref}
    write_json(tmp_path/'legacy-plan.json',p)
    from_capsules(tmp_path/'legacy-plan.json',tmp_path/'seq.npz')
    assert load_sequence(tmp_path/'seq.npz',s).data_kind=='synthetic'


def test_changed_native_input_stops_preparation(tmp_path):
    s,p=create_case(tmp_path)
    file=tmp_path/p['frames'][0]['channels']['0']['pixels']['path'];file.write_bytes(b'changed')
    with pytest.raises(ValueError):from_capsules(tmp_path/'plan.json',tmp_path/'seq.npz')


def test_channel_calibration_cannot_be_guessed(tmp_path):
    s,p=create_case(tmp_path)
    registry=json.loads((tmp_path/'sensors.json').read_text());registry['sensors'][0]['channels'][0]['calibration_id']='OTHER'
    write_json(tmp_path/'other-sensors.json',registry);p['sensor_registry']=reference(tmp_path,tmp_path/'other-sensors.json')
    write_json(tmp_path/'other-plan.json',p)
    with pytest.raises(ValueError,match='калибровка'):from_capsules(tmp_path/'other-plan.json',tmp_path/'seq.npz')
