"""These enforce documentation wiring, not the truth of an agent's report."""
import json
from pathlib import Path
import pytest
from global_weather.connectors.ecosystem import UPSTREAM,SOURCE_BLOBS

ROOT=Path(__file__).resolve().parents[1]
ROLES=('coordinator','data-steward','radiometry','normalization','physics','model-engineer','executor','verification','release-auditor')

@pytest.mark.parametrize('role',ROLES)
def test_every_role_reads_mandatory_contract(role):
    for prefix in ('agents','.claude/agents'):
        text=(ROOT/prefix/(role+'.md')).read_text()
        assert 'GLOBAL-WEATHER-OPS-1' in text
        assert 'agents/OPERATING_CONTRACT.md' in text
        assert '05-ecosystem-compatibility.md' in text
        assert 'bypassPermissions' not in text


def test_source_lock_matches_code():
    cfg=json.loads((ROOT/'configs/ecosystem_sources.json').read_text())
    assert cfg['sources']==UPSTREAM and cfg['source_blobs']==SOURCE_BLOBS
    assert cfg['installed_binaries_verified'] is False
    assert cfg['real_satellite_samples_tested'] is False


def test_root_enforces_ops_and_no_force():
    text=(ROOT/'AGENTS.md').read_text()
    assert 'GLOBAL-WEATHER-OPS-1' in text and 'без force' in text
    assert 'tests/test_ecosystem.py' in text and 'итоговый' in text.lower().replace('ё','е')
