"""Prepared-data endpoints use the parent application's Host and CSRF middleware."""
from pathlib import Path
from fastapi.responses import FileResponse
from .pipeline_jobs import PipelineSpec, authorize, manifest_summary


def register(app, queue):
    @app.get('/training')
    def page():
        return FileResponse(Path(__file__).with_name('static')/'training.html')

    @app.get('/api/pipeline/datasets')
    def datasets():
        root = queue.root/'datasets'
        if not root.is_dir() or root.is_symlink():
            return []
        result = []
        for child in sorted(root.iterdir())[:100]:
            if not child.is_dir() or child.is_symlink():
                continue
            try:
                result.append(manifest_summary(queue.root, child.name))
            except (ValueError, OSError, KeyError):
                result.append({'id': child.name, 'status': 'invalid_manifest'})
        return result

    @app.post('/api/pipeline/runs')
    def create(spec: PipelineSpec, role: str = 'executor'):
        authorize(role, spec.kind)
        summary = manifest_summary(queue.root, spec.dataset_id)
        if type(summary['mesh_level']) is not int or not 0 <= summary['mesh_level'] <= 2:
            raise ValueError('Веб-обучение ограничено сеткой из 12–162 ячеек.')
        if summary['sample_count'] > 16:
            raise ValueError('Веб-обучение ограничено 16 примерами.')
        changes = {'mesh_level': summary['mesh_level']}
        if spec.kind in ('evaluate_dataset', 'forecast_dataset'):
            parent = queue.get(spec.training_run)
            if parent['status'] != 'completed' or parent['spec']['kind'] != 'train_dataset':
                raise ValueError('Выберите завершённое обучение.')
            changes.update(hidden=parent['spec']['hidden'], latent_slots=parent['spec']['latent_slots'])
            if spec.kind == 'evaluate_dataset':
                changes['horizon_hours'] = parent['spec']['horizon_hours']
        if spec.kind == 'validate_dataset':
            changes['horizon_hours'] = summary['horizon_hours']
        spec = spec.model_copy(update=changes)
        return queue.create(spec)
