"""Dataset tasks for the existing serial queue. No browser-controlled code or URLs."""
from __future__ import annotations
import argparse
from pathlib import Path
import resource
from typing import Literal
from pydantic import Field, field_validator, model_validator
from .contracts import RunSpec, safe_child
from ..pipeline.io import atomic_json, read_json

KINDS = ('validate_dataset', 'train_dataset', 'evaluate_dataset', 'forecast_dataset')
PERMISSIONS = {'coordinator': {'validate_dataset'}, 'data-steward': {'validate_dataset'},
               'normalization': {'validate_dataset'}, 'radiometry': {'validate_dataset'},
               'physics': {'validate_dataset', 'evaluate_dataset'},
               'model-engineer': {'validate_dataset', 'train_dataset', 'forecast_dataset'},
               'executor': set(KINDS), 'verification': {'validate_dataset', 'evaluate_dataset', 'forecast_dataset'},
               'release-auditor': {'validate_dataset', 'evaluate_dataset'}}


class PipelineSpec(RunSpec):
    kind: Literal['validate_dataset', 'train_dataset', 'evaluate_dataset', 'forecast_dataset']
    dataset_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    training_run: str = ''
    sample_id: str = ''
    epochs: int = Field(default=2, ge=1, le=5)

    @model_validator(mode='after')
    def no_ignored_synthetic_options(self):
        if self.optimizer_steps or self.remove_source != 'none' or self.input_file:
            raise ValueError('Параметры синтетического испытания не применяются к выборке.')
        return self

    @field_validator('training_run')
    @classmethod
    def run_name(cls, value):
        import re
        if value and not re.fullmatch('[a-f0-9]{32}', value):
            raise ValueError('Требуется идентификатор запуска из очереди.')
        return value

    @field_validator('sample_id')
    @classmethod
    def sample_name(cls, value):
        import re
        if value and not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
            raise ValueError('Требуется ID примера из выборки.')
        return value


def authorize(role, kind):
    if role not in PERMISSIONS or kind not in PERMISSIONS[role]:
        raise ValueError('Роль не разрешает это действие с выборкой.')


def dataset_path(workspace, name):
    root = Path(workspace)/'datasets'
    if root.is_symlink():
        raise ValueError('Каталог данных не должен быть ссылкой.')
    folder = safe_child(root, name)
    if not folder.is_dir():
        raise ValueError('Набор не установлен оператором в каталоге datasets.')
    return safe_child(folder, 'dataset.json')


def manifest_summary(workspace, name):
    path = dataset_path(workspace, name); m = read_json(path)
    if not isinstance(m, dict):
        raise ValueError('Манифест должен быть объектом JSON.')
    if m.get('schema') not in ('global-weather-dataset-1', 'global-weather-input-1'):
        raise ValueError('Неизвестная схема набора.')
    if not isinstance(m.get('samples'), list) or any(not isinstance(x, dict) for x in m['samples']):
        raise ValueError('Некорректный состав примеров.')
    return {'id': name, 'sample_count': len(m['samples']), 'data_kind': m.get('data_kind'), 'mesh_level': m.get('mesh_level'),
            'horizon_hours': m.get('horizon_hours'), 'step_hours': m.get('step_hours'),
            'normalized': m.get('normalization') is not None,
            'samples': [{'id': x.get('id'), 'split': x.get('split')} for x in m.get('samples', [])[:100]]}


def execute(spec, output, workspace):
    from ..pipeline.dataset import PreparedDataset
    from ..pipeline.runner import train, evaluate, forecast, TrainConfig
    path = dataset_path(workspace, spec.dataset_id)
    ds = PreparedDataset(path, max_cells=162, max_samples=16, inference=spec.kind in ('forecast_dataset', 'validate_dataset'))
    if ds.step != 3:
        raise ValueError('Веб-испытание использует шаг 3 часа. Другие шаги доступны через CLI.')
    if spec.horizon_hours > ds.horizon:
        raise ValueError('Горизонт больше подготовленного диапазона.')
    if spec.kind == 'validate_dataset':
        return ds.validate(targets=not ds.is_input)
    if spec.kind == 'train_dataset':
        cfg = TrainConfig(epochs=spec.epochs, hidden=spec.hidden, latent_slots=spec.latent_slots,
                          horizon_hours=spec.horizon_hours, seed=spec.seed, memory_budget_mib=1024)
        return train(path, output/'experiment', cfg)
    if not spec.training_run:
        raise ValueError('Выберите завершённое обучение.')
    from .queue import RunQueue
    queue = RunQueue(workspace)
    parent = queue.get(spec.training_run)
    if parent['status'] != 'completed' or parent['spec']['kind'] != 'train_dataset':
        raise ValueError('Выбранный запуск не является завершённым обучением.')
    run = safe_child(Path(workspace)/'runs', spec.training_run)/'experiment'
    if spec.kind == 'evaluate_dataset':
        result = evaluate(path, run, output/'evaluation.json', split='test')
        return {'status': 'held_out_evaluation', **result}
    if not spec.sample_id:
        raise ValueError('Выберите входной пример.')
    result = forecast(path, run, spec.sample_id, output/'prediction', horizon_hours=spec.horizon_hours)
    for p in (output/'prediction').iterdir():
        if p.name == 'grid.npz' or p.name.startswith('frame_'):
            p.rename(output/p.name)
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--inbox', type=Path, required=True)
    args = p.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128*1024**2, 128*1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    spec = PipelineSpec.model_validate_json((args.run_dir/'request.json').read_text())
    result = execute(spec, args.run_dir, args.inbox.parent)
    atomic_json(args.run_dir/'report.json', result)
    from ..pipeline.io import sha256
    atomic_json(args.run_dir/'artifacts.json', {p.name: sha256(p) for p in args.run_dir.iterdir()
                                               if p.suffix in ('.npz', '.json') and p.name != 'artifacts.json'})


if __name__ == '__main__':
    main()
