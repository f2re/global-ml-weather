"""CLI regressions for an unrelated editable package ahead of the checkout."""
from pathlib import Path
import os
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def foreign_package(tmp_path, monkeypatch):
    foreign = tmp_path / 'foreign'
    package = foreign / 'global_weather'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text("raise RuntimeError('stale editable installation')\n")
    monkeypatch.setenv('PYTHONPATH', str(foreign))
    return foreign


def test_geometry_cli_prefers_own_checkout(tmp_path, foreign_package):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/mesh8_geometry_smoke.py'), '--help'],
        cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert '--output' in result.stdout


def test_runner_children_import_own_checkout(tmp_path, foreign_package):
    runner = runpy.run_path(str(ROOT / 'scripts/mesh8-r10-runner.py'))
    env = runner['execution_environment'](ROOT)
    assert str(foreign_package) in env['PYTHONPATH'].split(os.pathsep)
    result = subprocess.run(
        [sys.executable, '-c', 'import global_weather; print(global_weather.__file__)'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == ROOT / 'global_weather/__init__.py'
