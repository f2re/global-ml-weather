"""Local preparation, fixed norms, reproducible training and research inference."""
import argparse
import json
from pathlib import Path
from .io import read_json, atomic_json, reference, local_path
from .dataset import PreparedDataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('demo-dataset'); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mesh-level', type=int, default=0); p.add_argument('--horizon-hours', type=int, default=3)
    p = sub.add_parser('assemble'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('validate'); p.add_argument('--dataset', type=Path, required=True); p.add_argument('--output', type=Path)
    p = sub.add_parser('fit-norms'); p.add_argument('--dataset', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path)
    p = sub.add_parser('train'); p.add_argument('--dataset', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path); p.add_argument('--resume', action='store_true')
    p = sub.add_parser('evaluate'); p.add_argument('--dataset', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--split', choices=['validation', 'test'], default='test')
    p = sub.add_parser('forecast'); p.add_argument('--dataset', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--sample', required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--horizon-hours', type=int, default=72)
    p = sub.add_parser('prepare-targets'); p.add_argument('--pressure-netcdf', type=Path, required=True)
    p.add_argument('--surface-netcdf', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--issue-time', required=True); p.add_argument('--mesh-level', type=int, required=True)
    p.add_argument('--horizon-hours', type=int, default=72); p.add_argument('--step-hours', type=int, default=3)
    p.add_argument('--confirm-utc', action='store_true'); p.add_argument('--precipitation-kind', choices=['hourly_increment'])
    p = sub.add_parser('prepare-static'); p.add_argument('--netcdf', type=Path, required=True)
    p.add_argument('--mesh-level', type=int, required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'demo-dataset':
        from .fixture import create_fixture
        result = {'status': 'synthetic_fixture', 'dataset': str(create_fixture(args.output, mesh_level=args.mesh_level, horizon_hours=args.horizon_hours)),
                  'meteorologically_validated': False}
    elif args.command == 'assemble':
        root = args.plan.absolute().parent; m = read_json(args.plan)
        if args.output.absolute().parent != root or args.output.exists():
            raise ValueError('Создайте новый манифест рядом с планом, без перезаписи.')
        for key in ('registry', 'static', 'normalization'):
            if m.get(key) is not None:
                m[key] = reference(root, local_path(root, m[key]))
        for sample in m['samples']:
            for key in ('observations', 'targets'):
                if sample.get(key) is not None:
                    sample[key] = reference(root, local_path(root, sample[key]))
        atomic_json(args.output, m)
        result = {'status': 'manifest_sealed', 'path': str(args.output), 'admitted': False}
    elif args.command == 'validate':
        ds = PreparedDataset(args.dataset, inference=True)
        result = ds.validate(targets=not ds.is_input)
        if args.output: atomic_json(args.output, result)
    elif args.command == 'fit-norms':
        from .fit import fit_normalization
        result = fit_normalization(args.dataset, args.output, base_path=args.base)
    elif args.command == 'train':
        from .runner import train, TrainConfig, config_from_json
        cfg = config_from_json(read_json(args.config)) if args.config else TrainConfig()
        result = train(args.dataset, args.output, cfg, resume=args.resume)
    elif args.command == 'evaluate':
        from .runner import evaluate
        result = evaluate(args.dataset, args.run, args.output, split=args.split)
        result = {k: v for k, v in result.items() if k != 'scores'} | {'metrics_file': str(args.output)}
    elif args.command == 'forecast':
        from .runner import forecast
        result = forecast(args.dataset, args.run, args.sample, args.output, horizon_hours=args.horizon_hours)
    elif args.command == 'prepare-targets':
        from .era5 import prepare_targets
        result = prepare_targets(args.pressure_netcdf, args.surface_netcdf, args.output,
                                  issue_time=args.issue_time, mesh_level=args.mesh_level,
                                  horizon_hours=args.horizon_hours, step_hours=args.step_hours,
                                  confirm_utc=args.confirm_utc, precipitation_kind=args.precipitation_kind)
    else:
        from .era5 import prepare_static
        result = prepare_static(args.netcdf, args.output, mesh_level=args.mesh_level)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
