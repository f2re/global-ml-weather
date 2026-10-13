#!/usr/bin/env python3
"""Run the frozen mesh8 R10 Q4->Q5 qualification under bounded supervision.

The command sequence is fixed in source: preflight, one bounded training run,
frozen-checkpoint inference and final evidence verification. Data paths may be
selected by the operator, but no executable commands are accepted from config
or manifests. A successful run is technical qualification only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

MINIMUM_FREE_BYTES = 64 * 1024**3


def now():
    return datetime.now(timezone.utc)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait()


def bounded(command, *, root, run_root, stdout, stderr, deadline, env):
    process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr,
                               env=env, start_new_session=True)
    try:
        while process.poll() is None:
            if now() >= deadline:
                raise TimeoutError('R10 execution deadline exceeded.')
            if shutil.disk_usage(run_root).free < MINIMUM_FREE_BYTES:
                raise RuntimeError('R10 free-disk reserve fell below 64 GiB.')
            time.sleep(2)
        return process.returncode
    finally:
        if process.poll() is None:
            stop(process)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--norm', required=True)
    parser.add_argument('--climatology', required=True)
    parser.add_argument('--static', required=True)
    parser.add_argument('--config', default='configs/joint_observation_training.json')
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--deadline', required=True, help='Offset-aware ISO8601 UTC deadline')
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    actual_sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=root, text=True)
    if actual_sha != args.expected_sha or dirty.strip():
        raise SystemExit('R10 runner requires exact expected SHA and a clean tracked tree.')
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline <= now():
        raise SystemExit('R10 deadline must be a future offset-aware timestamp.')

    run_root = Path(args.run_root).expanduser().resolve()
    if run_root.is_symlink() or any(parent.is_symlink() for parent in run_root.parents):
        raise SystemExit('R10 run root may not use symlinks.')
    run_root.mkdir(parents=True, exist_ok=True)
    logs = run_root/'logs'; logs.mkdir(exist_ok=True)
    evidence_path = run_root/'execution-evidence.json'
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {
        'schema': 'mesh8-r10-execution-evidence-1', 'source_commit': actual_sha,
        'started_utc': now().isoformat(), 'deadline_utc': deadline.isoformat(),
        'scientific_acceptance': False, 'steps': []}
    if evidence.get('source_commit') != actual_sha or evidence.get('deadline_utc') != deadline.isoformat():
        raise SystemExit('Existing R10 evidence belongs to another source/deadline.')
    evidence['status'] = 'running'; save(evidence_path, evidence)

    python = sys.executable
    common = ['--dataset', str(Path(args.dataset).resolve()), '--norm', str(Path(args.norm).resolve()),
              '--climatology', str(Path(args.climatology).resolve()), '--static', str(Path(args.static).resolve())]
    preflight = [python, '-m', 'global_weather.mesh8_preflight', *common,
                 '--config', str(Path(args.config).resolve()), '--run-root', str(run_root),
                 '--expected-sha', actual_sha]
    training = [python, '-m', 'global_weather.joint_observation_training', *common,
                '--config', str(Path(args.config).resolve()), '--output', str(run_root/'training')]
    inference = [python, '-m', 'global_weather.joint_observation_inference',
                 '--dataset', str(Path(args.dataset).resolve()), '--training', str(run_root/'training'),
                 '--static', str(Path(args.static).resolve()), '--output', str(run_root/'forecast'),
                 '--split', 'validation', '--limit', '2', '--horizon-hours', '3', '--device', 'cuda:0',
                 '--warmups', '1', '--repeats', '3', '--threads', '8', '--spacing-degrees', '5',
                 '--source-revision', actual_sha]
    final = [python, '-m', 'global_weather.mesh8_final_check', '--run-root', str(run_root),
             '--expected-sha', actual_sha]
    commands = [('preflight', preflight), ('training', training), ('inference', inference), ('final-check', final)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8')

    try:
        for name, command in commands:
            if name == 'training' and (run_root/'training/complete.json').is_file():
                evidence['steps'].append({'name': name, 'status': 'resume_existing_complete',
                                          'finished_utc': now().isoformat()})
                save(evidence_path, evidence); continue
            if name == 'inference' and (run_root/'forecast/batch.json').is_file():
                batch = json.loads((run_root/'forecast/batch.json').read_text())
                if batch.get('status') == 'complete':
                    evidence['steps'].append({'name': name, 'status': 'resume_existing_complete',
                                              'finished_utc': now().isoformat()})
                    save(evidence_path, evidence); continue
                raise RuntimeError('Existing Q5 forecast is incomplete; preserve it and use a new run root.')
            record = {'name': name, 'command': command, 'started_utc': now().isoformat()}
            evidence['steps'].append(record); save(evidence_path, evidence)
            stdout_path, stderr_path = logs/(name+'.stdout.log'), logs/(name+'.stderr.log')
            with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
                code = bounded(command, root=root, run_root=run_root, stdout=stdout, stderr=stderr,
                               deadline=deadline, env=env)
            record.update(returncode=code, status='passed' if code == 0 else 'failed',
                          finished_utc=now().isoformat(),
                          stdout={'path': str(stdout_path.relative_to(run_root)), 'sha256': digest(stdout_path)},
                          stderr={'path': str(stderr_path.relative_to(run_root)), 'sha256': digest(stderr_path)})
            save(evidence_path, evidence)
            if code != 0:
                raise RuntimeError(name+' failed; inspect preserved logs.')
        final_report = json.loads((run_root/'final-verification.json').read_text())
        if final_report.get('status') != 'r10_q4_q5_technically_qualified':
            raise RuntimeError('Final R10 verifier did not qualify Q4/Q5 evidence.')
        evidence.update(status='completed_technical_qualification', finished_utc=now().isoformat(),
                        final_verification_sha256=digest(run_root/'final-verification.json'))
        save(evidence_path, evidence)
        return 0
    except BaseException as error:
        evidence.update(status='blocked', finished_utc=now().isoformat(),
                        failure={'type': type(error).__name__, 'message': str(error)})
        save(evidence_path, evidence)
        raise


if __name__ == '__main__':
    sys.exit(main())
