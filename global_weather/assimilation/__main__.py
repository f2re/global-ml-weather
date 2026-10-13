"""Command dispatcher: train, selftest, prepare-store, prepare-static, frozen (forecast | score)."""
from __future__ import annotations

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {'train': 'training', 'selftest': 'selftest', 'prepare-store': 'sources', 'prepare-static': 'static',
                'frozen': 'forecast'}
    if not argv or argv[0] not in commands:
        raise SystemExit('usage: python -m global_weather.assimilation {' + ','.join(commands) + '} [options]')
    from importlib import import_module
    import_module('global_weather.assimilation.' + commands[argv[0]]).main(argv[1:])


if __name__ == '__main__':
    main()
