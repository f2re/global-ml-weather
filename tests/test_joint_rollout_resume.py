from copy import deepcopy

import pytest

from global_weather.joint_observation_training import (
    MESH8_72H_RESUME_SOURCE_SHA256,
    _is_authorized_mesh8_72h_resume,
)


@pytest.fixture
def reviewed_transition():
    previous = {
        'epoch': 4500,
        'best_epoch': 4500,
        'identity': {
            'experiment_id': 'joint-mesh8-training-4500',
            'experiment_role': 'mesh8_research_training',
            'grid_level': 8,
            'config': {'rollout_hours': 3, 'learning_rate': 1e-4},
        },
    }
    ref = {'epoch': 4500, 'sha256': MESH8_72H_RESUME_SOURCE_SHA256}
    expected = {
        'experiment_id': 'joint-mesh8-training-72h',
        'experiment_role': 'mesh8_research_training',
        'grid_level': 8,
        'config': {'rollout_hours': 72, 'learning_rate': 5e-5},
    }
    return previous, ref, expected


def test_allows_only_the_reviewed_mesh8_3h_to_72h_checkpoint(reviewed_transition):
    assert _is_authorized_mesh8_72h_resume(*reviewed_transition)


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        (('identity', 'config', 'rollout_hours'), 6),
        (('identity', 'experiment_id'), 'joint-mesh8-training-4000'),
        (('epoch',), 4499),
        (('best_epoch',), 4499),
    ],
)
def test_rejects_other_source_horizons_and_epochs(reviewed_transition, path, value):
    previous, ref, expected = deepcopy(reviewed_transition)
    target = previous
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not _is_authorized_mesh8_72h_resume(previous, ref, expected)


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        (('config', 'rollout_hours'), 48),
        (('config', 'learning_rate'), 1e-5),
        (('experiment_id',), 'joint-mesh8-training-1000'),
        (('experiment_role',), 'mesh8_end_to_end_qualification'),
        (('grid_level',), 7),
    ],
)
def test_rejects_other_target_experiments(reviewed_transition, path, value):
    previous, ref, expected = deepcopy(reviewed_transition)
    target = expected
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not _is_authorized_mesh8_72h_resume(previous, ref, expected)


def test_rejects_a_different_checkpoint_hash(reviewed_transition):
    previous, ref, expected = deepcopy(reviewed_transition)
    ref['sha256'] = '0' * 64
    assert not _is_authorized_mesh8_72h_resume(previous, ref, expected)
