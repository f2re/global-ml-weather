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


def test_arktika_elektro_msugs_adapter_and_film_block():
    import torch
    from global_weather.multimodal.ru_sat_msugs import ArktikaElektroMSUGSAdapter, ViewConditionedFiLMBlock
    adapter = ArktikaElektroMSUGSAdapter()
    b, n_sats, n_nodes = 2, 3, 10
    raw_channels = torch.full((b, n_sats, n_nodes, 10), 260.0)
    valid_mask = torch.ones((b, n_sats, n_nodes), dtype=torch.bool)
    node_xyz = torch.randn(n_nodes, 3)
    node_xyz = node_xyz / torch.linalg.norm(node_xyz, dim=-1, keepdim=True) * 6371.0
    node_normal = node_xyz / 6371.0
    sat_xyz = torch.full((b, n_sats, 3), 42164.0)
    sun_dir = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gsics_slope = torch.ones(n_sats, 7)
    gsics_offset = torch.zeros(n_sats, 7)

    fused = adapter(raw_channels, valid_mask, node_xyz, node_normal, sat_xyz, sun_dir, gsics_slope, gsics_offset)
    assert fused.shape == (b, n_nodes, 20)
    assert torch.isfinite(fused).all()

    film = ViewConditionedFiLMBlock(hidden_dim=32, geom_dim=4)
    h = torch.randn(b, n_nodes, 32)
    geom_feat = fused[..., 13:17]
    h_mod = film(h, geom_feat)
    assert h_mod.shape == (b, n_nodes, 32)
    assert torch.isfinite(h_mod).all()


def test_collocate_msumr_ikfs2_clouds():
    from global_weather.products.algorithms import collocate_msumr_ikfs2
    h, w = 40, 40
    lat_grid, lon_grid = np.mgrid[60.0:64.0:40j, 30.0:34.0:40j]
    cloud_mask = np.zeros((h, w), dtype=bool)
    cloud_mask[10:30, 10:30] = True
    cth_pressure = np.full((h, w), 450.0, dtype=np.float32)

    ikfs2_lats = np.array([62.0, 60.1])
    ikfs2_lons = np.array([32.0, 30.1])
    cld_frac, cth = collocate_msumr_ikfs2(cloud_mask, cth_pressure, lat_grid, lon_grid, ikfs2_lats, ikfs2_lons, footprint_km=35.0)
    assert cld_frac[0] > 0.8
    assert cth[0] == pytest.approx(450.0, abs=10.0)
    assert cld_frac[1] < 0.2
