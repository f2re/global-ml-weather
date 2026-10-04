"""Full prepared-data integration using explicitly analytic, synthetic inputs."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest
from global_weather.products.core import Field, utc
from global_weather.products.algorithms import spectral_index
from global_weather.products.io import save_product, sha256
from global_weather.products.ingest import export_product
from global_weather.pipeline.fixture import create_fixture
from global_weather.pipeline.io import read_json, reference, atomic_json
from global_weather.pipeline.dataset import PreparedDataset
from global_weather.pipeline.fit import fit_normalization
from global_weather.products.catalog import variable_spec


def add_products(root):
    path=create_fixture(root,horizon_hours=72)
    data=read_json(path);registry=read_json(root/'registry.json')
    for i,s in enumerate(data['samples']):
        issue=utc(s['issue_time']);when=(issue-timedelta(hours=24)).isoformat()
        meta=dict(source='meteor_msu_mr',platform='SYNTHETIC',calibration_family='analytic-v1',data_kind='synthetic')
        def f(x,quantity,units='1',**metadata):
            return Field(np.array([x],float),np.array([True]),quantity,units,'analytic-grid',when,when,
                         sha256(root/'registry.json'),{**meta,**metadata})
        p=spectral_index(f(.4+i*.03,'surface_reflectance',spectral_role='nir'),
                f(.2,'surface_reflectance',spectral_role='red'),f(1.,'eligibility_mask'),
                f(30.,'solar_zenith_angle','degree'),product='ndvi',available_at=when,data_kind='synthetic')
        pf=root/f'ndvi-{i}.npz';save_product(pf,p)
        gf=root/f'geometry-{i}.npz';np.savez(gf,latitude=np.array([60.]),longitude=np.array([30.]),
             view_zenith_deg=np.array([20.]),footprint_km=np.array([4.]),grid_id=np.array('analytic-grid'))
        of=root/f'product-{i}.jsonl';report=export_product(pf,gf,of,history_hours=48)
        registry.update(report['registry'])
        source=root/s['observations']['path'];merged=root/f'combined-{i}.jsonl'
        merged.write_text(source.read_text()+of.read_text())
        s['observations']=reference(root,merged)
    rp=root/'products-registry.json';atomic_json(rp,registry)
    data['registry']=reference(root,rp);data['normalization']=None
    unscaled=root/'products-unscaled.json';atomic_json(unscaled,data)
    return unscaled


def test_derived_norms_fit_only_training_and_reach_prepared_input(tmp_path):
    source=add_products(tmp_path/'data');output=source.with_name('products-dataset.json')
    fit_normalization(source,output)
    ds=PreparedDataset(output);name=next(k for k,v in ds.registry.items() if v.get('product')=='ndvi')
    means=[(.4+i*.03-.2)/(.4+i*.03+.2) for i in range(3)]
    assert ds.norm.get(name,'1').mean[0]==pytest.approx(np.mean(means))
    assert ds.norm.get(name,'1').std[0]==pytest.approx(np.std(means))
    record=ds.packed(ds.samples[0]);k=record.vocabulary.index(name)
    assert (record.variables==k).any() and record.features[record.variables==k,1].item()==2.
    assert ds.history_hours==48


def test_slow_context_expands_holdout_guard(tmp_path):
    source=add_products(tmp_path/'data');m=read_json(source);registry=read_json(tmp_path/'data'/'products-registry.json')
    name=next(k for k,v in registry.items() if v.get('product'))
    registry[name]['history_hours']=168
    rp=tmp_path/'data'/'long-registry.json';atomic_json(rp,registry);m['registry']=reference(rp.parent,rp)
    out=rp.with_name('long-history.json');atomic_json(out,m)
    with pytest.raises(ValueError,match='пересекаются'):PreparedDataset(out,require_norm=False)


def test_synthetic_product_cannot_be_used_as_real(tmp_path):
    source=add_products(tmp_path/'data');m=read_json(source);m['data_kind']='real'
    out=source.with_name('real-label.json');atomic_json(out,m)
    ds=PreparedDataset(out,require_norm=False)
    with pytest.raises(ValueError,match='Происхождение'):ds.records(ds.samples[0])


def test_products_train_forecast_full_horizon(tmp_path):
    source=add_products(tmp_path/'data');ds=source.with_name('products-dataset.json');fit_normalization(source,ds)
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(epochs=1,hidden=16,latent_slots=4,horizon_hours=72,
                    seed=17,learning_rate=.0001,patience=5,threads=1,memory_budget_mib=1024)))
    run=tmp_path/'training';out=tmp_path/'forecast'
    for args in [['train','--dataset',str(ds),'--config',str(config),'--output',str(run)],
                 ['forecast','--dataset',str(ds),'--run',str(run),'--sample','sample-4','--horizon-hours','72','--output',str(out)]]:
        result=subprocess.run([sys.executable,'-m','global_weather.pipeline',*args],capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
    assert list(out.glob('*.npz'))


@pytest.mark.parametrize('source,platform',[('arktika_m','ARCM1'),('electro_l','ELEKTRO-L-3')])
def test_existing_numeric_capsule_bridge_preserves_values(tmp_path,source,platform):
    from global_weather.products.bridge import from_capsule
    from global_weather.products.io import load_field, arrays
    root=tmp_path/'capsule';root.mkdir()
    measured='2020-01-01T00:00:00+00:00';ready='2020-01-01T00:10:00+00:00'
    np.savez(root/'pixels.npz',values=np.array([[250.,260.]],dtype=np.float32),
        valid=np.ones((1,2),bool),latitude=np.array([[60.,60.]]),longitude=np.array([[30.,31.]]),
        view_zenith_deg=np.array([[20.,21.]]),footprint_km=np.array([[4.,4.]]))
    meta=dict(schema='physical-raster-v1',quantity='brightness_temperature',units='K',source=source,
        platform=platform,instrument='MSU-GS',channel_id='9',calibration_reference='analytic-test-only',
        observed_at=measured,available_at=ready,arrays_sha256=sha256(root/'pixels.npz'),
        shape=[1,2],valid_pixels=2,crs='EPSG:4326',transform=[1,0,29.5,0,-1,60.5],
        geometry_reference='analytic-test-only',data_kind='synthetic')
    (root/'manifest.json').write_text(json.dumps(meta))
    out=tmp_path/'field.npz';geo=tmp_path/'geometry.npz'
    from_capsule(root,out,data_kind='synthetic',geometry_output=geo)
    f=load_field(out)
    assert f.values.tolist()==[[250.,260.]] and f.observed_at==measured
    assert f.metadata['spectral_role']=='thermal_window' and arrays(geo)['latitude'].shape==(1,2)
    with pytest.raises(ValueError):from_capsule(root,tmp_path/'bad.npz',data_kind='real')
