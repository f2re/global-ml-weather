"""Synthetic end-to-end execution, including an actual trained 72-hour rollout."""
from pathlib import Path
import json
import tempfile
from global_weather.pipeline.fixture import create_fixture
from global_weather.pipeline.runner import TrainConfig, train, evaluate, forecast
from global_weather.pipeline.io import atomic_json


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        dataset=create_fixture(root/'data',horizon_hours=72)
        cfg=TrainConfig(epochs=1,horizon_hours=72,hidden=16,latent_slots=4,memory_budget_mib=1024)
        result=train(dataset,root/'run',cfg)
        scored=evaluate(dataset,root/'run',root/'test.json')
        issued=forecast(dataset,root/'run','sample-4',root/'forecast',horizon_hours=72)
        assert result['epochs_completed']==1 and not result['test_set_used_for_selection']
        assert len(issued['lead_hours'])==25 and issued['targets_read'] is False
        assert scored['split']=='test' and scored['data_kind']=='synthetic'
        assert any(x['lead_hours']==72 for x in scored['scores'])
        report={'status':'synthetic_pipeline_passed','forecast_leads':issued['lead_hours'],
                'metric_rows':len(scored['scores']),'test_used_for_selection':False,
                'data_kind':'synthetic','meteorologically_validated':False}
        atomic_json(Path('outputs/pipeline/check.json'),report)
        print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
