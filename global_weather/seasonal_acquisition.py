"""Explicit bounded acquisition of five public NOAA monthly climate means."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import signal
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from .observation_training import digest, save
from .profile_normalization import _lock

BASE = 'https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/Monthlies/pressure/'
FILES = {'air': 9709277, 'shum': 5045372, 'uwnd': 10927383, 'vwnd': 11405646, 'hgt': 8874993}
MAX_BYTES = 64 * 1024**2


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('NOAA climate redirects are forbidden before following.')


def _open(url: str):
    return build_opener(_RejectRedirect()).open(url, timeout=90)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _deadline():
    # The acquisition CLI has a single main-thread executor on the Linux node.
    # Socket timeout alone does not bound repeated reads; ITIMER_REAL also
    # interrupts a stalled read and enforces the 90-second attempt wall time.
    def expired(signum, frame):
        raise TimeoutError('NOAA climate attempt exceeded 90 seconds.')
    previous = signal.signal(signal.SIGALRM, expired)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, 90)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *prior_timer)
        signal.signal(signal.SIGALRM, previous)


def _regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('Climate cache file must be a single regular file.')


def acquire(output: str | Path, *, allow_network: bool = False) -> dict:
    root = Path(output)
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError('Climate cache symlinks are forbidden.')
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'acquisition.json'
    with _lock(manifest):
        if manifest.exists():
            _regular(manifest)
            report = json.loads(manifest.read_text())
        else:
            report = {'schema': 'noaa-monthly-acquisition-1', 'provider': 'NOAA_NCEP1',
                      'period': ['1991-01-01', '2020-12-31'], 'maximum_body_bytes': MAX_BYTES,
                      'created_utc': _now(), 'received_body_bytes': 0, 'files': {}, 'attempts': []}
        if (report.get('schema') != 'noaa-monthly-acquisition-1'
                or report.get('provider') != 'NOAA_NCEP1'
                or report.get('period') != ['1991-01-01', '2020-12-31']
                or report.get('maximum_body_bytes') != MAX_BYTES
                or type(report.get('received_body_bytes')) is not int
                or not 0 <= report['received_body_bytes'] <= MAX_BYTES
                or not isinstance(report.get('files'), dict)
                or not isinstance(report.get('attempts'), list)):
            raise ValueError('Climate acquisition identity or byte budget differs.')
        started = time.monotonic()
        for variable, size in FILES.items():
            name = variable + '.mon.ltm.1991-2020.nc'
            path = root / name
            url = BASE + name
            if name in report['files']:
                _regular(path)
                reference = report['files'][name]
                if (reference.get('url') != url or reference.get('bytes') != size
                        or path.stat().st_size != size or digest(path) != reference.get('sha256')):
                    raise ValueError('Pinned cached climate bytes changed.')
                continue
            if path.exists() or path.is_symlink():
                raise ValueError('Unregistered existing climate file; preserve for inspection.')
            if not allow_network:
                raise ValueError('Missing climate cache; use explicit --allow-network acquisition.')
            attempts = sum(item['name'] == name for item in report['attempts'])
            succeeded = False
            for number in range(attempts + 1, 4):
                part = root / (name + f'.attempt-{number}.part')
                if part.exists() or part.is_symlink():
                    raise ValueError('Ambiguous climate download; preserve partial file.')
                attempt = {'name': name, 'number': number, 'url': url, 'status': 'running', 'started_utc': _now()}
                report['attempts'].append(attempt)
                save(manifest, report)
                try:
                    if time.monotonic() - started >= 1800:
                        raise TimeoutError('Climate acquisition phase deadline exceeded.')
                    with _deadline(), _open(url) as response:
                        if response.geturl() != url or urlsplit(response.geturl()).hostname != 'psl.noaa.gov':
                            raise ValueError('Unexpected NOAA download redirect.')
                        length = response.headers.get('Content-Length')
                        if length is not None and int(length) != size:
                            raise ValueError('NOAA climate response length differs from reviewed catalog.')
                        descriptor = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                        received = 0
                        with os.fdopen(descriptor, 'wb') as stream:
                            while True:
                                remaining = min(size - received, MAX_BYTES - report['received_body_bytes'])
                                if remaining <= 0:
                                    break
                                block = response.read(min(512 * 1024, remaining))
                                if not block:
                                    break
                                received += len(block)
                                report['received_body_bytes'] += len(block)
                                save(manifest, report)
                                if (received > size or report['received_body_bytes'] > MAX_BYTES
                                        or time.monotonic() - started >= 1800):
                                    raise ValueError('Climate download byte or time budget exhausted.')
                                stream.write(block)
                            stream.flush(); os.fsync(stream.fileno())
                    if received != size:
                        raise ValueError('Incomplete NOAA climate file.')
                    with part.open('rb') as stream:
                        magic = stream.read(4)
                    if not (magic.startswith(b'CDF') or magic == b'\x89HDF'):
                        raise ValueError('NOAA response is not NetCDF.')
                    checksum = digest(part)
                    part.rename(path)
                    report['files'][name] = {'url': url, 'bytes': size, 'sha256': checksum}
                    attempt.update(status='passed', finished_utc=_now())
                    save(manifest, report)
                    succeeded = True
                    break
                except (OSError, ValueError, TimeoutError) as exc:
                    attempt.update(status='failed', finished_utc=_now(), exception=type(exc).__name__, reason=str(exc))
                    save(manifest, report)
                    if report['received_body_bytes'] >= MAX_BYTES:
                        raise ValueError('Climate download total budget exhausted.') from exc
            if not succeeded:
                raise RuntimeError('Climate acquisition attempt limit reached: ' + name)
        report['status'] = 'verified_cached_files'
        report['verified_utc'] = _now()
        save(manifest, report)
        return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-network', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(acquire(args.output, allow_network=args.allow_network), indent=2))


if __name__ == '__main__':
    main()
