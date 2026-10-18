"""Bounded acquisition of held-out GHCNh archives for train-admitted stations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from .providers.ghcnh import fetch_station
from .pipeline.io import atomic_json as save


def acquire(admission, cache, output, *, network=False, years=(2022,), max_bytes=64*1024**3):
    admitted = json.loads(Path(admission).read_text())
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    report = {'source': 'NOAA GHCNh', 'years': list(years), 'results': [], 'station_selection': 'train_admitted'}
    for yr in years:
        for station in admitted['stations']:
            station_id = station['id'] if isinstance(station, dict) else station
            used = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
            if used >= max_bytes or shutil.disk_usage(cache).free < 32*1024**3:
                raise RuntimeError('Acquisition storage budget exhausted')
            try:
                path, receipt = fetch_station(station_id, yr, cache, network=network,
                                               max_bytes=min(100_000_000, max_bytes-used))
                report['results'].append({'station': station_id, 'year': yr, 'status': 'downloaded', 'receipt': receipt})
            except (ValueError, OSError) as exc:
                # Keep selected station unchanged; missing validation/test never selects a replacement.
                report['results'].append({'station': station_id, 'year': yr, 'status': 'unavailable', 'reason': str(exc)})
            save(output, report)
            print(station_id, yr, report['results'][-1]['status'], flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True); parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--allow-network', action='store_true')
    parser.add_argument('--years', type=int, nargs='+', default=[2022])
    args = parser.parse_args(argv)
    acquire(args.admission, args.cache, args.output, network=args.allow_network, years=tuple(args.years))


if __name__ == '__main__': main()
