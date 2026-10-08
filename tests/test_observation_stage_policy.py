"""Direct observation-only instruction overrides the imported teacher proposal."""
import pytest
from global_weather.observation_training import TrainingStage, default_training_stages, stage_plan_payload


def test_no_real_training_stage_reads_era5():
    stages = default_training_stages()
    for stage in stages:
        assert not any('era5' in name for name in stage.inputs)
        if stage.trainable_modules != ('none',):
            assert not any('era5' in name for name in stage.targets)
            assert stage.era5_role != 'teacher'
    assert stage_plan_payload()['era5_training_allowed'] is False


def test_explicit_teacher_stage_is_rejected():
    with pytest.raises(ValueError, match='GLOBAL-WEATHER-OBS-1'):
        TrainingStage('S0', 'invalid teacher', ('processor',), ('era5',), ('era5_future',), (), 'teacher', False).validate()
