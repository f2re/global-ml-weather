"""R11 memory/time profile of one training sample per curriculum stage (program check on CPU)."""
from global_weather.assimilation.profiling import profile
from global_weather.assimilation.selftest import SELFTEST_CONFIG


def test_profile_reports_each_distinct_stage_without_training():
    config = dict(SELFTEST_CONFIG)
    config['stages'] = [dict(SELFTEST_CONFIG['stages'][0], rollout_hours=3, ensemble_members=1, cycling=False),
                        dict(SELFTEST_CONFIG['stages'][0], rollout_hours=6),
                        dict(SELFTEST_CONFIG['stages'][0], rollout_hours=6)]
    report = profile(config, synthetic=True, device='cpu')
    assert report['schema'] == 'physical-assimilation-profile-1' and report['data_kind'] == 'synthetic'
    assert [row['rollout_hours'] for row in report['stages']] == [3, 6]
    for row in report['stages']:
        assert row['targets'] > 0 and row['forward_seconds'] > 0 and row['backward_seconds'] > 0
        assert row['peak_rss_bytes'] > 0 and 'cuda_peak_reserved_bytes' not in row
    assert report['parameters'] > 0
