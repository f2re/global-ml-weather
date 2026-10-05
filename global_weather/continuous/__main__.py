"""Register date ranges and inspect C1 metadata. This CLI does not train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from .store import CampaignStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('outputs/lab'))
    actions = parser.add_subparsers(dest='action', required=True)
    add = actions.add_parser('add-range', help='Добавить даты к постоянной программе')
    add.add_argument('--start-date', required=True)
    add.add_argument('--end-date', required=True)
    add.add_argument('--request-key')
    actions.add_parser('state', help='Прочитать состояние и границы реализации')
    for name in ('ranges', 'events', 'samples'):
        sub = actions.add_parser(name)
        sub.add_argument('--after', type=int, default=0)
        sub.add_argument('--limit', type=int, default=100)
    blocks = actions.add_parser('blocks')
    blocks.add_argument('--after-date')
    blocks.add_argument('--limit', type=int, default=100)
    args = parser.parse_args(argv)
    try:
        store = CampaignStore(args.workspace)
        if args.action == 'add-range':
            result = store.add_range(args.start_date, args.end_date, request_key=args.request_key)
        elif args.action == 'state':
            result = store.state()
        elif args.action == 'blocks':
            result = store.blocks(after_date=args.after_date, limit=args.limit)
        else:
            # Selection is from argparse's fixed actions, never a module or shell command.
            reader = {'ranges': store.ranges, 'events': store.events, 'samples': store.samples}[args.action]
            result = reader(after=args.after, limit=args.limit)
    except (ValueError, sqlite3.Error, OSError) as exc:
        parser.exit(2, f'Ошибка журнала программы: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
