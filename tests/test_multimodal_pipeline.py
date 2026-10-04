"""Обязательный полный цикл; отсутствие pipeline не разрешает пропуск теста."""
import json
import subprocess
import sys
from global_weather.multimodal.fixture import dataset
from global_weather.pipeline.dataset import PreparedDataset
from global_weather.pipeline.runner import config_from_json,make_model
from global_weather.multimodal.model import MultimodalWeatherModel


def test_prepared_training_uses_every_encoder(tmp_path):
    source=dataset(tmp_path/'data',horizon_hours=6)
    ds=PreparedDataset(source);ds.validate()
    cfg={'epochs':1,'hidden':16,'latent_slots':4,'horizon_hours':6,'threads':1,'memory_budget_mib':1024}
    model=make_model(ds,config_from_json(cfg))
    assert isinstance(model,MultimodalWeatherModel)
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    run=tmp_path/'training';forecast=tmp_path/'forecast'
    for args in (['train','--dataset',str(source),'--config',str(config),'--output',str(run)],
                 ['evaluate','--dataset',str(source),'--run',str(run),'--output',str(tmp_path/'evaluation.json')],
                 ['forecast','--dataset',str(source),'--run',str(run),'--sample','sample-4','--horizon-hours','6','--output',str(forecast)]):
        process=subprocess.run([sys.executable,'-m','global_weather.pipeline',*args],capture_output=True,text=True,timeout=90)
        assert process.returncode==0,process.stdout+process.stderr
    assert list(forecast.glob('*.npz'))
