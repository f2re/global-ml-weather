from pathlib import Path
import pytest
from global_weather.profile_verification import verify


def test_external_verification_refuses_unfinished_training(tmp_path):
    training=tmp_path/'training'; training.mkdir()
    with pytest.raises(ValueError,match='freeze'):
        verify(tmp_path/'dataset',training,Path('missing-pressure.nc'),Path('missing-surface.nc'),
               '2022-08-01T00:00:00Z',tmp_path/'diagnostic')
    assert not (tmp_path/'diagnostic').exists()
