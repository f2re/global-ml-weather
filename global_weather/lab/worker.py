"""Whitelisted child process tasks. Network and arbitrary code are not exposed."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import torch
from .contracts import RunSpec, atomic_json, sha256, safe_child, now
from .metrics import physical_diagnostics


def emit(stage, **details):
    print(json.dumps(dict(time=now(), stage=stage, **details), ensure_ascii=False, allow_nan=False), flush=True)


def provenance():
    root = Path(__file__).resolve().parents[2]
    try:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, timeout=3)
        revision = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        revision = None
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*.py'))
             if p.relative_to(root).parts[0] in ('global_weather', 'tests')}
    versions = {}
    for name in ('numpy', 'torch', 'scipy', 'fastapi'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(git_revision=revision, source_sha256=files, python=sys.version.split()[0], versions=versions,
                platform=sys.platform, cpu_threads=1, device='cpu')


def simulate(spec, output):
    from global_weather.grid import build_pyramid, latlon
    from global_weather.cli import synthetic_records
    from global_weather.observations import pack_observations
    from global_weather.vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS
    from global_weather.model import GlobalWeatherModel
    from global_weather.training import Targets, train_step
    torch.set_num_threads(1)
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    issue = datetime(2026, 10, 3, tzinfo=timezone.utc)
    grids = build_pyramid(spec.mesh_level)
    grid = grids[0]
    grid.save(output/'grid.npz')
    records = synthetic_records(grid, issue)
    if spec.remove_source != 'none': records = [r for r in records if r['source'] != spec.remove_source]
    obs = pack_observations(records, grid, np.array(PRESSURE_HPA)*100, issue)
    kwargs = dict(observation_schema=obs.schema_fingerprint, hidden=spec.hidden, step_hours=3)
    if spec.kind == 'adaptive':
        from global_weather.adaptive import AdaptiveWeatherModel
        model = AdaptiveWeatherModel(grids, obs.vocabulary, latent_slots=spec.latent_slots,
                                     allow_unscaled_synthetic=True, **kwargs)
    else:
        model = GlobalWeatherModel(grids, obs.vocabulary, **kwargs)
    static = torch.zeros(grid.n_cells)
    emit('prepared', cells=grid.n_cells, observations=obs.accepted_records)
    history = []
    if spec.optimizer_steps:
        with torch.no_grad(): teacher = list(model(obs, static, static, horizon_hours=3))[-1]
        profiles = teacher.profiles[None].clone(); profiles[..., 0] += 1.
        targets = Targets((3,), profiles, teacher.profile_mask[None, ..., None].expand_as(profiles).clone(),
                          teacher.surface[None].clone(), teacher.surface_mask[None].clone(), model.grid_fingerprint, model.pressure_pa)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for i in range(spec.optimizer_steps):
            result = train_step(model, opt, obs, static, static, targets)
            history.append(result)
            emit('optimizer', step=i+1, loss=result['loss'])
    model.eval()
    diagnostics = []
    with torch.inference_mode():
        for frame in model(obs, static, static, horizon_hours=spec.horizon_hours):
            pp, ss = frame.profiles.numpy(), frame.surface.numpy()
            dd = physical_diagnostics(pp, ss, model.pressure_pa.numpy(), grid.areas_m2)
            if not dd['finite']: raise FloatingPointError('Неконечный прогноз: испытание остановлено.')
            diagnostics.append(dict(lead_hours=frame.lead_hours, **dd))
            np.savez_compressed(output/f'frame_{frame.lead_hours:03d}.npz', profiles=pp, surface=ss,
                                profile_mask=frame.profile_mask.numpy(), surface_mask=frame.surface_mask.numpy())
            emit('forecast', lead_hours=frame.lead_hours, total_hours=spec.horizon_hours)
    report = dict(schema_version=1, status='synthetic', meteorologically_validated=False,
                  normalization='synthetic_engineering_scales_NOT_ERA5', static_fields='synthetic_zero_terrain_and_land',
                  issue_time=issue.isoformat(), spec=spec.model_dump(), cells=grid.n_cells,
                  grid_fingerprint=grid.fingerprint, pressure_hpa=list(PRESSURE_HPA),
                  profile_variables=list(PROFILE_VARIABLES), profile_units=list(PROFILE_UNITS),
                  surface_variables=list(SURFACE_VARIABLES), surface_units=list(SURFACE_UNITS),
                  input_records=obs.accepted_records, parameters=sum(p.numel() for p in model.parameters()),
                  optimization=history, diagnostics=diagnostics, lead_hours=[d['lead_hours'] for d in diagnostics],
                  skill={'rmse': None, 'reason': 'Нет независимых метеорологических целей.'}, provenance=provenance())
    atomic_json(output/'report.json', report)
    # Immutable, portable execution protocol with checksums of all produced arrays.
    atomic_json(output/'artifacts.json', {p.name: sha256(p) for p in output.iterdir() if p.suffix in ('.npz', '.json') and p.name != 'artifacts.json'})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--inbox', type=Path, required=True)
    args = parser.parse_args(argv)
    # Limits are defense in depth, not an OS sandbox for untrusted code.
    resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128*1024*1024, 128*1024*1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    spec = RunSpec.model_validate_json((args.run_dir/'request.json').read_text())
    emit('started', kind=spec.kind)
    begin = time.monotonic()
    if spec.kind in ('baseline', 'adaptive'):
        simulate(spec, args.run_dir)
    elif spec.kind == 'tests':
        root = Path(__file__).resolve().parents[2]
        paths = ['tests/test_core.py', 'tests/test_adaptive.py']
        if not all((root/p).is_file() for p in paths): raise ValueError('Нужна полная рабочая копия репозитория с тестами.')
        code = subprocess.call([sys.executable, '-m', 'pytest', '-q', *paths], cwd=root)
        atomic_json(args.run_dir/'report.json', dict(status='software_tests', exit_code=code, provenance=provenance(), meteorologically_validated=False))
        if code: raise RuntimeError('Тесты завершились с ошибкой.')
    else:
        from global_weather.connectors.local import inspect_file
        report = inspect_file(safe_child(args.inbox, spec.input_file))
        report.update(provenance=provenance(), meteorologically_validated=False)
        atomic_json(args.run_dir/'report.json', report)
    emit('completed', elapsed_seconds=round(time.monotonic()-begin, 3), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    main()
