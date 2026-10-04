"""Fixed task dispatch for one serial worker. Request data never provide a module."""
import argparse
from pathlib import Path
from ..pipeline.io import read_json


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--inbox', type=Path, required=True)
    a = p.parse_args(argv)
    kind = read_json(a.run_dir/'request.json').get('kind')
    if kind in ('validate_dataset', 'train_dataset', 'evaluate_dataset', 'forecast_dataset'):
        from .pipeline_jobs import main as execute
    elif kind in ('baseline', 'adaptive', 'tests', 'inspect'):
        from .worker import main as execute
    else:
        raise ValueError('Неизвестное действие исполнителя.')
    execute(['--run-dir', str(a.run_dir), '--inbox', str(a.inbox)])


if __name__ == '__main__':
    main()
