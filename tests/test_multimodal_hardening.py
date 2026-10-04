"""Проверки отказов по нормам, памяти и передаче геометрии."""
from dataclasses import replace
import pytest
import torch
from global_weather.grid import build_pyramid
from global_weather.multimodal.fixture import fixture
from global_weather.multimodal.fusion import project_sequence
from global_weather.multimodal.io import save_sensors,load_sensors


def test_masked_footprint_nan_does_not_hide_large_valid_footprint():
    grids=build_pyramid(0);sensors,obs=fixture(grids);s=sensors[0];q=obs.sequences[0]
    valid=q.valid.clone();valid[0]=False
    footprints=q.footprint_km.clone();footprints[0]=float('nan');footprints[1]=1e8
    q=replace(q,valid=valid,footprint_km=footprints)
    with pytest.raises(ValueError,match='Пятно больше'):
        project_sequence(torch.zeros(16,8,8),torch.ones(8,8,dtype=torch.bool),torch.ones(8,8),q,s,grids[0])


def test_real_fitted_norms_require_source_files(tmp_path):
    s,_=fixture(build_pyramid(0));real=replace(s[0],data_kind='real')
    save_sensors(tmp_path/'registry.json',[real])
    with pytest.raises(ValueError,match='исходный файл'):load_sensors(tmp_path/'registry.json')


def test_small_image_cannot_reach_invalid_groupnorm_size():
    sensors,obs=fixture(build_pyramid(0),height=1,width=1)
    with pytest.raises(ValueError,match='8×8'):obs.sequences[0].validate(sensors[0])


def test_permanently_missing_pixels_need_no_invented_coordinates():
    sensors,obs=fixture(build_pyramid(0));q=obs.sequences[0];v=q.valid.clone();v[:,:,0,0]=False
    changes={'valid':v}
    for name in ('latitude','longitude','area_m2'):
        array=getattr(q,name).clone();array[0,0]=float('nan');changes[name]=array
    replace(q,**changes).validate(sensors[0])
