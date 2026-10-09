"""Synthetic structural S2 fixtures, not actual IGRA or weather skill evidence."""
from datetime import timedelta
import json
import hashlib

import numpy as np
import pytest
import torch

from global_weather.grid import build_pyramid
from global_weather.profile_training import (
    ProfileDataset, ProfileObservationModel, START, TRAIN_END, VARIABLES,
    bounded_records, checked, configuration, prepare, profile_loss, VAL_END,
)
from global_weather.vertical import PROFILE_UNITS


def records(when, split='train', shift=0.):
    values=[260.+shift,.003+shift*.00001,5.+shift,-2.+shift,40000.+shift*100]
    return [dict(source='radiosonde',provider='NOAA_IGRA2',valid=True,
                 observation_id=f'fixture/{when.isoformat()}/{variable}',profile_id=f'fixture/{when.isoformat()}',
                 observed_at=when.isoformat(),available_at=(when+timedelta(hours=2)).isoformat(),
                 latitude=10.,longitude=179.,pressure_pa=70000.,variable=variable,value=values[i],
                 units=PROFILE_UNITS[i],group_split=split,revision=0,provider_qc={'software_fixture':True},
                 time_basis='reported_level_time',position_basis='reported_level_position',
                 archive_sha256='0'*64,format_sha256='1'*64)
            for i,variable in enumerate(VARIABLES)]


def source_file(tmp_path, validation_shift=0.):
    rows=records(START+timedelta(days=2),shift=0.)+records(START+timedelta(days=3),shift=2.)
    rows+=records(TRAIN_END+timedelta(days=3),split='validation',shift=validation_shift)
    path=tmp_path/'records.jsonl'; path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    admission(path)
    return path


def admission(path):
    (path.parent/'manifest.json').write_text(json.dumps({'schema':'igra-observation-archive-1',
        'provider':'NOAA_IGRA2','data_kind':'real','observations_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'software_fixture':True}))


def test_preparation_unique_train_norms_exclude_validation(tmp_path):
    one=tmp_path/'one'; two=tmp_path/'two'; one.mkdir(); two.mkdir()
    first=prepare(source_file(one,10),one/'dataset')
    second=prepare(source_file(two,100),two/'dataset')
    assert first['statistics']==second['statistics']
    assert all(row['count']==2 for row in first['statistics'])
    assert first['records']==15
    assert first['norm_interpretation'].startswith('pooled actual train pressures')
    assert first['scientific_acceptance'] is False


def test_history_availability_and_full_profile_split_guards(tmp_path):
    path=source_file(tmp_path); prepare(path,tmp_path/'dataset'); dataset=ProfileDataset(tmp_path/'dataset')
    when=START+timedelta(days=2)
    assert not dataset.records(when-timedelta(hours=1),when,issue=when)
    assert len(dataset.records(when-timedelta(hours=1),when,issue=when+timedelta(hours=2)))==5
    for split in ('train','validation','test'):
        issues=dataset.issues(split,3)
        assert len(issues)==3
        assert all(issue.hour==6 for issue in issues)
    assert dataset.issues('train',3)[-1]+timedelta(hours=114)<TRAIN_END


def test_profile_source_revision_units_roles_and_mutation_fail_closed(tmp_path):
    row=records(START+timedelta(days=2))[0]
    for patch in ({'provider':'ERA5'},{'revision':1},{'withdrawn':True},{'units':'C'}, {'group_split':None}):
        with pytest.raises(ValueError): checked(dict(row,**patch))
    path=source_file(tmp_path); prepare(path,tmp_path/'dataset')
    dataset=ProfileDataset(tmp_path/'dataset')
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'): dataset.verify()


def test_profile_physical_model_37_levels_masks_and_active_gradients():
    torch.manual_seed(3)
    model=ProfileObservationModel(build_pyramid(0)[0],[260,.003,5,-2,40000],[20,.001,10,10,20000],hidden=8,pressure_bounds=[[100,100000]]*5)
    issue=START+timedelta(days=2,hours=6)
    inputs=records(issue-timedelta(hours=6)); targets=records(issue+timedelta(hours=17),shift=3.)
    frames=model(inputs,issue)
    assert len(frames)==25 and frames[-1].lead_hours==72
    assert frames[-1].profiles.shape==(12,37,6)
    assert torch.isnan(frames[-1].profiles[...,5]).all()
    assert not frames[-1].profile_variable_mask[...,5].any()
    assert not frames[-1].surface_mask.any()
    objective,counts=profile_loss(model,frames,targets)
    assert counts==[1]*5 and torch.isfinite(objective)
    objective.backward()
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None,name
        assert torch.isfinite(parameter.grad).all(),name
        assert parameter.grad.abs().sum()>0,name


def test_future_unavailable_input_has_no_effect():
    torch.manual_seed(5)
    model=ProfileObservationModel(build_pyramid(0)[0],[260,.003,5,-2,40000],[20,.001,10,10,20000],hidden=8,pressure_bounds=[[100,100000]]*5)
    issue=START+timedelta(days=2,hours=6)
    future=records(issue+timedelta(hours=2))
    unavailable=records(issue-timedelta(hours=1))
    with torch.no_grad():
        empty=model([],issue); forecast=model(future+unavailable,issue)
    for expected,actual in zip(empty,forecast):
        assert torch.equal(expected.profiles[...,:5],actual.profiles[...,:5])


def test_fixed_bounded_sampling_and_limits():
    rows=records(START+timedelta(days=2))+records(START+timedelta(days=3),shift=2)
    assert bounded_records(rows,5)==bounded_records(list(reversed(rows)),5)
    assert set(row['variable'] for row in bounded_records(rows,5))==set(VARIABLES)
    assert len(bounded_records(rows,5))==5
    with pytest.raises(ValueError): configuration({'epochs':101})
    with pytest.raises(ValueError): configuration({'max_records_per_window':20001})
    with pytest.raises(ValueError): configuration({'unknown':True})


def test_variable_pressure_support_does_not_hide_supported_temperature():
    bounds=[[100,100000],[70000,90000],[100,100000],[100,100000],[100,100000]]
    model=ProfileObservationModel(build_pyramid(0)[0],[260,.003,5,-2,40000],[20,.001,10,10,20000],hidden=8,pressure_bounds=bounds)
    with torch.no_grad(): frame=model([],START)[0]
    assert frame.profile_mask.all()
    assert frame.profile_variable_mask[...,0].all()
    assert not frame.profile_variable_mask[:,-1,1].any()
    assert torch.isnan(frame.profiles[:,-1,1]).all()
    assert torch.isfinite(frame.profiles[:,-1,0]).all()


def test_training_checkpoint_resume_identity_and_separate_test(tmp_path,monkeypatch):
    from global_weather.profile_training import train,evaluate
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    rows=[]
    for start,split in ((START,'train'),(TRAIN_END,'validation'),(VAL_END,'test')):
        rows+=records(start+timedelta(days=2),split,0.)+records(start+timedelta(days=3),split,2.)
    source=tmp_path/'records.jsonl'; source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    admission(source)
    prepare(source,tmp_path/'dataset')
    config={'mesh_level':0,'hidden':8,'epochs':1,'threads':1,
            'max_train_issues':1,'max_validation_issues':1,'max_test_issues':1,'max_records_per_window':50}
    train(tmp_path/'dataset',tmp_path/'training',config)
    ref=json.loads((tmp_path/'training'/'best.json').read_text())
    train(tmp_path/'dataset',tmp_path/'training',config)
    assert json.loads((tmp_path/'training'/'best.json').read_text())==ref
    evaluate(tmp_path/'dataset',tmp_path/'training',tmp_path/'test.json')
    report=json.loads((tmp_path/'test.json').read_text())
    assert report['split']=='test' and report['scientific_acceptance'] is False
    assert sum(cell['count'] for row in report['metrics'] for cell in row if cell)==5
    with pytest.raises(ValueError,match='resume'):
        train(tmp_path/'dataset',tmp_path/'training',dict(config,hidden=12))
    from global_weather import profile_training as training
    original_identity=training.identity
    monkeypatch.setattr(training,'identity',lambda *args:dict(original_identity(*args),commit='f'*40))
    with pytest.raises(ValueError,match='identity'):
        training.load_model(tmp_path/'dataset',tmp_path/'training')
    frozen,_=training.load_frozen(tmp_path/'dataset',tmp_path/'training')
    assert not frozen.training and not any(p.requires_grad for p in frozen.parameters())
    original_fingerprint=training._model_definition_sha256
    monkeypatch.setattr(training,'_model_definition_sha256',lambda commit=None: 'changed' if commit is None else original_fingerprint(commit))
    with pytest.raises(ValueError,match='identity'):
        training.load_frozen(tmp_path/'dataset',tmp_path/'training')


def test_explicit_optimizer_continuation_preserves_parent_and_rejects_drift(tmp_path,monkeypatch):
    from global_weather import profile_training as training
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    rows=[]
    for start,split in ((START,'train'),(TRAIN_END,'validation'),(VAL_END,'test')):
        rows+=records(start+timedelta(days=2),split,0.)+records(start+timedelta(days=3),split,2.)
    source=tmp_path/'records.jsonl'; source.write_text(''.join(json.dumps(row)+'\n' for row in rows)); admission(source)
    dataset=tmp_path/'dataset'; prepare(source,dataset)
    config={'mesh_level':0,'hidden':8,'epochs':1,'threads':1,
            'max_train_issues':1,'max_validation_issues':1,'max_test_issues':1,'max_records_per_window':50}
    parent=tmp_path/'parent'; training.train(dataset,parent,config)
    original={p.name:training.digest(p) for p in parent.iterdir() if p.is_file()}
    completion=json.loads((parent/'complete.json').read_text()); commit=completion['identity']['commit']
    child=tmp_path/'child'
    training.train(dataset,child,dict(config,epochs=2),continue_from=parent,parent_commit=commit)
    lineage=json.loads((child/'continuation.json').read_text())
    assert lineage['parent_epoch']==1
    assert 'optimizer' in lineage['preserved'] and lineage['model_selection_source']=='measured_validation_only'
    first=torch.load(child/'epoch-0001/state.pt',weights_only=True)
    old=torch.load(training.checkpoint_path(parent,lineage['parent_reference']),weights_only=True)
    assert all(first['optimizer']['state'][k]['step']==v['step']+1 for k,v in old['optimizer']['state'].items())
    training.train(dataset,child,dict(config,epochs=2))
    assert original=={p.name:training.digest(p) for p in parent.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='architecture'):
        training.train(dataset,tmp_path/'bad',dict(config,hidden=12),continue_from=parent,parent_commit=commit)
    with pytest.raises(ValueError,match='identity'):
        training.train(dataset,tmp_path/'bad-source',config,continue_from=parent,parent_commit='0'*40)
    (parent/'latest.json').write_text((parent/'latest.json').read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        training.train(dataset,child,dict(config,epochs=2))


def test_source_fingerprint_covers_loss_and_dependencies(monkeypatch):
    from global_weather import profile_training as training
    actual=training._model_definition_sha256()
    original=training.subprocess.check_output
    def changed(arguments,**kwargs):
        if arguments[:2]==['git','show']:
            relative=arguments[2].split(':',1)[1]
            data=(training.Path(training.__file__).resolve().parents[1]/relative).read_bytes()
            if relative.endswith('/model.py'): data+=b'\n# dependency changed\n'
            return data.decode() if kwargs.get('text') else data
        return original(arguments,**kwargs)
    monkeypatch.setattr(training.subprocess,'check_output',changed)
    assert training._model_definition_sha256('0'*40)!=actual
