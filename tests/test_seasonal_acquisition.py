"""Offline transport-contract tests; these do not assert NOAA data acquisition."""
import io
import json
import pytest
from global_weather import seasonal_acquisition as acquisition


class Response(io.BytesIO):
    def __init__(self, url, body, length=None):
        super().__init__(body)
        self.url = url
        self.headers = {'Content-Length': str(len(body) if length is None else length)}

    def geturl(self):
        return self.url


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setattr(acquisition, 'FILES', {'air': 8})
    calls = []
    def opened(url):
        calls.append(url)
        return Response(url, b'CDF\x01test')
    monkeypatch.setattr(acquisition, '_open', opened)
    return calls


def test_network_requires_permission_and_pinned_reuse_detects_drift(tmp_path, transport):
    with pytest.raises(ValueError, match='allow-network'):
        acquisition.acquire(tmp_path)
    assert not transport
    report = acquisition.acquire(tmp_path, allow_network=True)
    assert report['received_body_bytes'] == 8
    assert report['attempts'][0]['status'] == 'passed'
    assert report['attempts'][0]['finished_utc']
    assert acquisition.acquire(tmp_path)['files'] == report['files']
    assert len(transport) == 1
    (tmp_path / 'air.mon.ltm.1991-2020.nc').write_bytes(b'CDF\x01evil')
    with pytest.raises(ValueError, match='changed'):
        acquisition.acquire(tmp_path, allow_network=True)
    assert len(transport) == 1


def test_partial_failures_preserve_files_and_have_three_attempt_limit(tmp_path, transport, monkeypatch):
    monkeypatch.setattr(acquisition, '_open', lambda url: Response(url, b'CDF', length=8))
    with pytest.raises(RuntimeError, match='attempt limit'):
        acquisition.acquire(tmp_path, allow_network=True)
    report = json.loads((tmp_path / 'acquisition.json').read_text())
    assert report['received_body_bytes'] == 9
    assert len(report['attempts']) == 3
    assert all(row['status'] == 'failed' for row in report['attempts'])
    assert len(list(tmp_path.glob('*.part'))) == 3
    with pytest.raises(RuntimeError, match='attempt limit'):
        acquisition.acquire(tmp_path, allow_network=True)


def test_byte_budget_is_not_exceeded_even_by_read_chunk(tmp_path, transport, monkeypatch):
    monkeypatch.setattr(acquisition, 'MAX_BYTES', 5)
    with pytest.raises(ValueError, match='total budget'):
        acquisition.acquire(tmp_path, allow_network=True)
    report = json.loads((tmp_path / 'acquisition.json').read_text())
    assert report['received_body_bytes'] == 5
    assert (tmp_path / 'air.mon.ltm.1991-2020.nc.attempt-1.part').stat().st_size == 5


def test_redirect_is_rejected_before_contacting_destination():
    with pytest.raises(ValueError, match='before following'):
        acquisition._RejectRedirect().redirect_request(None, None, 302, 'redirect', {}, 'https://unapproved.example/')


def test_catalog_length_mismatch_never_reads_body(tmp_path, transport, monkeypatch):
    monkeypatch.setattr(acquisition, '_open', lambda url: Response(url, b'CDF\x01test', length=999))
    with pytest.raises(RuntimeError, match='attempt limit'):
        acquisition.acquire(tmp_path, allow_network=True)
    report = json.loads((tmp_path / 'acquisition.json').read_text())
    assert report['received_body_bytes'] == 0
    assert not list(tmp_path.glob('*.part'))


def test_deadline_is_wall_clock_not_only_socket(monkeypatch):
    timers = []
    handlers = []
    monkeypatch.setattr(acquisition.signal, 'signal', lambda sig, handler: handlers.append(handler))
    monkeypatch.setattr(acquisition.signal, 'setitimer', lambda *args: timers.append(args) or (0, 0))
    with acquisition._deadline():
        with pytest.raises(TimeoutError, match='90 seconds'):
            handlers[0](None, None)
    assert timers[0][1] == 90
    assert timers[-1][1:] == (0, 0)
