"""Synthetic software checks only; these fixtures are not actual observations."""
from datetime import timedelta
import json

import numpy as np
import pytest

from global_weather.observation_data import (
    START, TRAIN_END, VAL_END, VARIABLES, UNITS, ObservationDataset,
    _checked, _unique, admit, prepare,
)
from global_weather.pipeline.io import read_json


def row(time, variable=0, station='USW00000001', value=280., delay=1):
    return {'provider':'NOAA_GHCNh','valid':True,'variable':VARIABLES[variable],
            'units':UNITS[variable],'value':value,'latitude':10.,'longitude':20.,
            'observation_id':f'GHCNh/{station}/{time.isoformat()}/{VARIABLES[variable]}',
            'observed_at':time.isoformat(),'available_at':(time+timedelta(hours=delay)).isoformat(),
            'revision':0,'provider_qc':{'measured':{'Quality_Code':'1','Source_Code':'223',
                                                'Measurement_Code':''}}}


def cache(tmp_path):
    root=tmp_path/'cache';root.mkdir();records=[]
    for hour in range(100):
        for variable in range(6):
            records.append(row(START+timedelta(hours=hour),variable,value=280.+hour+variable))
    # Extreme holdout values and a future-only station cannot affect admission/norms.
    records.append(row(TRAIN_END+timedelta(hours=1),value=1e7))
    records.append(row(TRAIN_END+timedelta(hours=2),station='USW00000002'))
    (root/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    return root


def test_train_only_admission_and_norms_with_causal_inputs(tmp_path):
    root=cache(tmp_path);admission=tmp_path/'admission.json'
    selected=admit([root],admission,minimum_coverage=.001)
    assert [s['id'] for s in selected['stations']]==['USW00000001']
    manifest=prepare([root],admission,tmp_path/'dataset')
    assert prepare([root],admission,tmp_path/'dataset')==manifest
    ds=ObservationDataset(tmp_path/'dataset')
    assert ds.mean[0]==pytest.approx(329.5)
    assert manifest['source_roles']['static'] is None
    assert manifest['global_profile_acceptance'].startswith('BLOCKED')
    sample=ds.sample(ds.subset('train')[0])
    assert sample['input'].shape==(12,1,6) and sample['target'].shape==(24,1,6)
    assert not sample['input_mask'][-1].any()  # Hour-24 records arrive at hour 25.
    assert sample['input_mask'][:-1].all()
    assert sample['target'][0,0,0]==307.  # Real fixture measurement at +3 h.
    assert np.isfinite(sample['normalized_input']).all()
    assert sample['normalized_input'][-1].tolist()==[[0.]*6]
    train_last=ds.samples[ds.subset('train')[-1]]['hour']
    val_first=ds.samples[ds.subset('validation')[0]]['hour']
    val_last=ds.samples[ds.subset('validation')[-1]]['hour']
    test_first=ds.samples[ds.subset('test')[0]]['hour']
    assert val_first-train_last>=84 and test_first-val_last>=84
    assert ds.samples[ds.subset('validation')[0]]['issue'].startswith('2022-01-')
    (tmp_path/'dataset/norm.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):
        ObservationDataset(tmp_path/'dataset')


def test_hourly_operator_boundary_and_qc():
    assert _checked(row(START+timedelta(hours=2)))[1]==2
    assert _checked(row(START+timedelta(hours=2,minutes=1)))[1]==3
    bad=row(START);bad['provider_qc']['measured']['Quality_Code']='4'
    assert _checked(bad) is None
    missing=row(START);missing.pop('provider_qc')
    with pytest.raises(ValueError,match='QC'):_checked(missing)


def test_identical_duplicates_not_counted_and_conflicting_version_blocked(tmp_path):
    root=tmp_path/'cache';root.mkdir();r=row(START)
    path=root/'records.jsonl';path.write_text(json.dumps(r)+'\n'+json.dumps(r)+'\n')
    import sqlite3
    _unique(root,tmp_path/'unique.sqlite')
    with sqlite3.connect(tmp_path/'unique.sqlite') as db:
        assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==1
    r['value']+=1
    with path.open('a') as stream:stream.write(json.dumps(r)+'\n')
    with pytest.raises(ValueError,match='Conflicting'):_unique(root,tmp_path/'conflict.sqlite')


def test_unapproved_sources_and_no_invented_norms(tmp_path):
    root=tmp_path/'era5';root.mkdir()
    (root/'records.jsonl').write_text(json.dumps(row(START))+'\n')
    with pytest.raises(ValueError,match='Only immutable'):
        admit(root,tmp_path/'bad.json',minimum_coverage=.00001)
    root=tmp_path/'cache';root.mkdir()
    (root/'records.jsonl').write_text(json.dumps(row(START))+'\n')
    admission=tmp_path/'admission.json';admit(root,admission,minimum_coverage=.00001)
    with pytest.raises(ValueError,match='no invented norms'):
        prepare(root,admission,tmp_path/'dataset')


def test_admission_sources_cannot_change_after_selection(tmp_path):
    root=cache(tmp_path);admission=tmp_path/'admission.json'
    admit(root,admission,minimum_coverage=.001)
    with (root/'records.jsonl').open('a') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='Admission train data changed'):
        prepare(root,admission,tmp_path/'dataset')


def test_revised_or_revoked_records_require_issue_aware_operator(tmp_path):
    root=tmp_path/'cache';root.mkdir();r=row(START)
    r['revision']=1
    path=root/'records.jsonl';path.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError,match='revision zero'):
        _unique(root,tmp_path/'revision.sqlite')
    r['revision']=0;r['valid']=False
    path.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError,match='Revoked'):
        _unique(root,tmp_path/'revoked.sqlite')
