"""Role registry and deterministic dispatch; not autonomous LLM agents."""
import argparse
import json
from pathlib import Path
from .contracts import RunSpec, atomic_json
from .pipeline_jobs import KINDS, PERMISSIONS, PipelineSpec

ROLES = [
 dict(id='coordinator', title='Координатор исследования', responsibility='Гипотезы, последовательность работ, зависимости и статусы доказательств', actions=['inspect'], instruction='agents/coordinator.md'),
 dict(id='data-steward', title='Данные и происхождение', responsibility='Каталоги, лицензии, SHA256, времена наблюдения и поступления, неизменяемые исходники', actions=['inspect', 'download-graphcast', 'download-noaa', 'plan-era5'], instruction='agents/data-steward.md'),
 dict(id='radiometry', title='Радиометрия и геометрия', responsibility='Каналы, калибровка, параллакс, антенные пятна; право отправить данные в карантин', actions=['inspect'], instruction='agents/radiometry.md'),
 dict(id='normalization', title='Климатические нормы', responsibility='Фиксированные μ/σ, уровни, интервалы, обучающий период и совместимость', actions=['inspect'], instruction='agents/normalization.md'),
 dict(id='physics', title='Физическая обоснованность', responsibility='Уравнения, размерности, балансы, маски, ограничения и независимые проверки', actions=['tests'], instruction='agents/physics.md'),
 dict(id='model-engineer', title='Архитектура модели', responsibility='Кодировщики, направленный граф, сжатая вертикаль, память и сравнение архитектур', actions=['baseline', 'adaptive'], instruction='agents/model-engineer.md'),
 dict(id='executor', title='Исполнение кода', responsibility='Запуск разрешённых испытаний, лимиты, коды возврата и воспроизводимость', actions=['baseline', 'adaptive', 'tests', 'inspect'], instruction='agents/executor.md'),
 dict(id='verification', title='Независимая верификация', responsibility='Сравнение с целями и контрольным прогнозом; проверка утечек и отказов источников', actions=['tests', 'inspect'], instruction='agents/verification.md'),
 dict(id='release-auditor', title='Аудит выпуска', responsibility='Тесты, браузерные сценарии, безопасность, полнота ограничений и документация', actions=['tests'], instruction='agents/release-auditor.md'),
]
for role in ROLES:
    role['actions'] += sorted(PERMISSIONS[role['id']])


def authorize(role, action):
    item = next((r for r in ROLES if r['id'] == role), None)
    if item is None or action not in item['actions']: raise ValueError('Роль не разрешает это действие.')
    return item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=[r['id'] for r in ROLES])
    parser.add_argument('--action', choices=['baseline', 'adaptive', 'tests', 'inspect', 'download-graphcast', 'download-noaa', 'plan-era5', *KINDS])
    parser.add_argument('--workspace', type=Path, default=Path('outputs/lab'))
    parser.add_argument('--input-file', default='')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--network', action='store_true')
    parser.add_argument('--station'); parser.add_argument('--year', type=int); parser.add_argument('--date')
    parser.add_argument('--dataset-id', default='')
    parser.add_argument('--training-run', default=''); parser.add_argument('--sample-id', default='')
    parser.add_argument('--epochs', type=int, default=2); parser.add_argument('--horizon-hours', type=int, default=3)
    args = parser.parse_args(argv)
    if not args.role:
        print(json.dumps(ROLES, ensure_ascii=False, indent=2)); return
    role = authorize(args.role, args.action)
    if not args.execute:
        print(json.dumps(dict(role=role, status='plan_only', note='Исполнение требует --execute. Это не LLM-вызов.'), ensure_ascii=False, indent=2)); return
    if args.action in ('download-graphcast', 'download-noaa', 'plan-era5'):
        from global_weather.connectors.acquire import main as acquire
        import uuid
        from .contracts import now
        directory = args.workspace.resolve()/'acquisition'/uuid.uuid4().hex
        directory.mkdir(parents=True)
        record = dict(role=args.role, action=args.action, status='started', started=now(), network_permitted=args.network)
        atomic_json(directory/'agent.json', record)
        try:
            if args.action == 'plan-era5':
                command = ['era5-request', '--date', args.date or '', '--output', str(directory/'request.json')]
            else:
                if not args.network: raise ValueError('Загрузка требует явного --network.')
                command = ['graphcast' if args.action == 'download-graphcast' else 'noaa-isd', '--network', '--output', str(directory/'upstream' if args.action == 'download-graphcast' else directory/'observations.csv')]
                if args.action == 'download-noaa': command += ['--station', args.station or '', '--year', str(args.year or 0)]
            acquire(command)
            record.update(status='completed', finished=now())
        except BaseException as exc:
            record.update(status='failed', finished=now(), error_type=type(exc).__name__)
            atomic_json(directory/'agent.json', record)
            raise
        atomic_json(directory/'agent.json', record)
        print(json.dumps(record, ensure_ascii=False, indent=2)); return
    from .queue import RunQueue, TERMINAL
    import time
    queue = RunQueue(args.workspace)
    if args.action in KINDS:
        spec = PipelineSpec(kind=args.action, dataset_id=args.dataset_id,
                            training_run=args.training_run, sample_id=args.sample_id,
                            epochs=args.epochs, horizon_hours=args.horizon_hours)
        from .pipeline_jobs import manifest_summary
        summary = manifest_summary(queue.root, args.dataset_id)
        if type(summary['mesh_level']) is not int or not 0 <= summary['mesh_level'] <= 2 or summary['sample_count'] > 16:
            raise ValueError('Задание превышает ограничения стенда.')
        spec = spec.model_copy(update={'mesh_level': summary['mesh_level']})
    else:
        spec = RunSpec(kind=args.action, input_file=args.input_file)
    queue.start()
    try:
        run = queue.create(spec)
        atomic_json(queue.runs/run['id']/'agent.json', dict(role=args.role, instruction=role['instruction'], backend='deterministic_allowlist'))
        while queue.get(run['id'])['status'] not in TERMINAL: time.sleep(.2)
        result = queue.get(run['id']); print(json.dumps(result, ensure_ascii=False, indent=2))
        if result['status'] != 'completed': raise SystemExit(1)
    finally: queue.close()


if __name__ == '__main__': main()
