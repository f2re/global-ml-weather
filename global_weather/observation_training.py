"""Literature-backed staged training for observation-driven weather models.

ERA5 is a dense teacher and diagnostic reference in the early stages.  It is
not an operational observation and is forbidden as a deployment-time input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable

from .observation_identity import observation_group_identity


@dataclass(frozen=True)
class TrainingStage:
    id: str
    purpose: str
    trainable_modules: tuple[str, ...]
    inputs: tuple[str, ...]
    targets: tuple[str, ...]
    required_evidence: tuple[str, ...]
    era5_role: str
    deployment_equivalent: bool

    def validate(self):
        if self.era5_role not in ('teacher', 'secondary_validation', 'forbidden'):
            raise ValueError('Неизвестная роль ERA5.')
        if self.deployment_equivalent and 'era5' in self.inputs:
            raise ValueError('Рабочий этап не может требовать ERA5 на входе.')
        if not self.id or not self.trainable_modules or not self.inputs or not self.targets:
            raise ValueError('Неполное описание этапа обучения.')
        return self


def default_training_stages():
    """Return the ordered training programme, not proof that it has run."""
    stages = (
        TrainingStage(
            'S0-theory',
            'Проверить динамическое ядро, маски, устойчивость и горизонт на плотных полях.',
            ('processor', 'decoder'), ('era5',), ('era5_future',),
            ('held_out_era5_period', 'physical_diagnostics', '72h_rollout'),
            'teacher', False),
        TrainingStage(
            'S1-observation-encoder',
            'Научить кодировщик восстанавливать состояние из разреженных реальных наблюдений.',
            ('observation_encoder',), ('raw_observations',), ('era5_analysis',),
            ('modality_dropout', 'coordinate_time_metadata', 'withheld_profiles'),
            'teacher', False),
        TrainingStage(
            'S2-observation-space',
            'Дообучить всю систему по будущим фактическим наблюдениям в их координатах и времени.',
            ('observation_encoder', 'processor', 'observation_decoder'),
            ('raw_observations',), ('future_raw_observations',),
            ('observation_operator', 'profile_group_holdout', 'missingness_stress'),
            'secondary_validation', True),
        TrainingStage(
            'S3-cycling',
            'Выполнять циклический анализ с предыдущим прогнозом и асинхронными наблюдениями.',
            ('analysis_update', 'processor', 'observation_decoder'),
            ('background_forecast', 'raw_observations'), ('future_raw_observations',),
            ('actual_observation_times', 'late_arrival_policy', 'checkpoint_resume'),
            'secondary_validation', True),
        TrainingStage(
            'S4-independent-validation',
            'Проверить модель на скрытых профилях, платформах, районах и временных интервалах.',
            ('none',), ('raw_observations', 'model_checkpoint'),
            ('withheld_raw_observations', 'era5_diagnostic'),
            ('no_gradient', 'unseen_platforms', 'unseen_regions', 'unseen_times'),
            'secondary_validation', True),
    )
    for stage in stages:
        stage.validate()
    return stages


def stage_plan_payload():
    stages = [asdict(stage) for stage in default_training_stages()]
    body = {'schema': 'observation-training-stages-1', 'stages': stages,
            'operational_input': 'raw_observations_plus_background',
            'era5_operational_input': False, 'execution_implemented': False}
    body['fingerprint'] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    return body


def partition_observation_groups(records: Iterable[dict], *, seed=17,
                                 validation_fraction=.1, test_fraction=.1):
    """Split complete profiles/platform events, never individual levels."""
    for value, label in ((validation_fraction, 'validation'), (test_fraction, 'test')):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f'Неверная доля {label}.')
    if validation_fraction + test_fraction >= 1:
        raise ValueError('Контрольные доли не оставляют обучающую часть.')
    if type(seed) is not int:
        raise ValueError('Зерно разбиения должно быть целым.')
    groups = {}
    for record in records:
        group = observation_group_identity(record)
        groups.setdefault(group, []).append(record)
    result = {'train': [], 'validation': [], 'test': []}
    for group, rows in groups.items():
        value = int.from_bytes(hashlib.sha256(f'{seed}/{group}'.encode()).digest()[:8], 'big') / 2**64
        role = ('test' if value < test_fraction else
                'validation' if value < test_fraction + validation_fraction else 'train')
        result[role].extend(rows)
    return result


def validate_stage_transition(completed_ids, next_id):
    order = [stage.id for stage in default_training_stages()]
    if next_id not in order:
        raise ValueError('Неизвестный этап обучения.')
    completed = set(completed_ids)
    expected = set(order[:order.index(next_id)])
    if not expected.issubset(completed):
        raise ValueError('Не завершены обязательные предыдущие этапы.')
    return True


if __name__ == '__main__':
    print(json.dumps(stage_plan_payload(), ensure_ascii=False, indent=2))
