"""Loopback-only offline test interface. One process, no cloud scripts or shell API."""
from contextlib import asynccontextmanager
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .contracts import RunSpec, safe_child
from .queue import RunQueue
from .agents import ROLES, authorize
from ..connectors import CATALOG
from ..connectors.local import inventory, MAX_BYTES


def create_app(workspace=None, *, testing=False, arktika_root=None):
    queue = RunQueue(workspace or os.environ.get('WEATHER_LAB_HOME', 'outputs/lab'))
    from ..autonomous.service import Experiments
    autonomous = Experiments(queue.root)
    token = secrets.token_urlsafe(32)
    @asynccontextmanager
    async def lifespan(app):
        queue.start()
        try:
            autonomous.start()
            try: yield
            finally: autonomous.close()
        finally: queue.close()
    app = FastAPI(title='Испытания глобальной модели', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.queue = queue
    app.state.autonomous = autonomous
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost'] + (['testserver'] if testing else []))

    @app.middleware('http')
    async def guard(request: Request, call_next):
        if request.method not in ('GET', 'HEAD'):
            origin = request.headers.get('origin')
            own = f'{request.url.scheme}://{request.headers.get("host", "")}'
            if (origin and origin != own) or not secrets.compare_digest(request.headers.get('x-lab-csrf', ''), token):
                return PlainTextResponse('Запрос отклонён: источник или CSRF.', status_code=403)
            length = request.headers.get('content-length')
            if length and (not length.isdigit() or int(length) > MAX_BYTES):
                return PlainTextResponse('Превышен предел запроса.', status_code=413)
        response = await call_next(request)
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc): return PlainTextResponse(str(exc), status_code=400)
    @app.exception_handler(KeyError)
    async def not_found(request, exc): return PlainTextResponse('Не найдено.', status_code=404)

    @app.get('/api/bootstrap')
    def bootstrap():
        return dict(csrf=token, roles=ROLES, connectors=CATALOG,
                    status='research_untrained', network_execution=False,
                    limits=dict(mesh_level=3, hidden=32, horizon_hours=72, max_queued=6),
                    protocols=['01-scientific-method.md', '02-data-and-normalization.md', '03-execution-security.md',
                               '04-verification-release.md', '05-ecosystem-compatibility.md',
                               '06-russian-documentation.md', '07-training-and-inference.md'])
    @app.get('/api/runs')
    def runs(): return queue.list()
    @app.post('/api/runs')
    def create(spec: RunSpec, role: str = 'executor'):
        authorize(role, spec.kind)
        return queue.create(spec)
    @app.get('/api/runs/{run_id}')
    def run(run_id: str): return queue.get(run_id)
    @app.post('/api/runs/{run_id}/cancel')
    def cancel(run_id: str): return queue.cancel(run_id)
    @app.get('/api/runs/{run_id}/log', response_class=PlainTextResponse)
    def log(run_id: str):
        queue.get(run_id); path = safe_child(queue.runs, run_id)/'execution.log'
        if not path.exists(): return ''
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size-100_000)); return f.read(100_000).decode('utf-8', errors='replace')
    @app.get('/api/runs/{run_id}/report')
    def report(run_id: str):
        current = queue.get(run_id)
        if current['status'] != 'completed': raise HTTPException(409, 'Результат доступен после успешного завершения.')
        path = safe_child(queue.runs, run_id)/'report.json'
        if not path.exists(): raise HTTPException(404, 'Нет отчёта.')
        return json.loads(path.read_text(encoding='utf-8'))
    @app.get('/api/runs/{run_id}/grid')
    def grid(run_id: str):
        queue.get(run_id)
        path = safe_child(queue.runs, run_id)/'grid.npz'
        if not path.exists(): raise HTTPException(404, 'Сетка ещё не готова.')
        with np.load(path, allow_pickle=False) as g:
            return dict(xyz=g['xyz'].tolist(), vertices=g['polygon_vertices'].tolist(),
                        offsets=g['polygon_offsets'].tolist(), indices=g['polygon_indices'].tolist())
    @app.get('/api/runs/{run_id}/frame')
    def frame(run_id: str, lead: int = 0, variable: str = 't2m', level: int = 0, cell: int = 0):
        from ..vertical import PROFILE_VARIABLES, SURFACE_VARIABLES
        r = report(run_id)
        if r.get('status') not in ('synthetic', 'research_forecast') or lead not in r.get('lead_hours', []): raise HTTPException(400, 'Недопустимый срок.')
        if not 0 <= cell < r['cells'] or not 0 <= level < 37: raise HTTPException(400, 'Недопустимая ячейка или уровень.')
        path = safe_child(queue.runs, run_id)/f'frame_{lead:03d}.npz'
        with np.load(path, allow_pickle=False) as f:
            if variable in PROFILE_VARIABLES:
                k = PROFILE_VARIABLES.index(variable); values = f['profiles'][:, level, k]; mask = f['profile_mask'][:, level]; unit = r['profile_units'][k]
            elif variable in SURFACE_VARIABLES:
                k = SURFACE_VARIABLES.index(variable); values = f['surface'][:, k]; mask = f['surface_mask'][:, k]; unit = r['surface_units'][k]
            else: raise HTTPException(400, 'Неизвестная величина.')
            profile = f['profiles'][cell]; pm = f['profile_mask'][cell]
            return dict(values=[float(v) if ok and np.isfinite(v) else None for v, ok in zip(values, mask)], unit=unit,
                        profile=[[float(v) if ok and np.isfinite(v) else None for v in row] for row, ok in zip(profile, pm)],
                        pressure_hpa=r['pressure_hpa'], cell=cell, lead_hours=lead)
    @app.get('/api/runs/{run_id}/download/{name}')
    def artifact(run_id: str, name: str):
        queue.get(run_id)
        if name not in ('report.json', 'artifacts.json', 'execution.json', 'grid.npz') and not re.fullmatch(r'frame_\d{3}\.npz', name):
            raise HTTPException(404, 'Этот артефакт не выдаётся.')
        path = safe_child(safe_child(queue.runs, run_id), name)
        if not path.is_file(): raise HTTPException(404, 'Нет артефакта.')
        return FileResponse(path, filename=name)
    @app.get('/api/inbox')
    def inbox(): return inventory(queue.inbox)
    @app.post('/api/inbox/{name}')
    async def upload(name: str, request: Request):
        path = safe_child(queue.inbox, name)
        if path.suffix.lower() not in ('.jsonl', '.json', '.csv', '.cbor', '.nc'): raise HTTPException(400, 'Недопустимый формат.')
        if path.exists(): raise HTTPException(409, 'Файл уже существует; исходники не перезаписываются.')
        temporary = path.with_suffix(path.suffix+'.part'); total = 0; owned = False
        try:
            with temporary.open('xb') as stream:
                owned = True
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > MAX_BYTES: raise HTTPException(413, 'Превышен предел 32 МиБ.')
                    stream.write(chunk)
            if path.exists(): raise HTTPException(409, 'Файл уже существует.')
            temporary.replace(path)
        except FileExistsError:
            raise HTTPException(409, 'Файл уже загружается.')
        except Exception:
            if owned: temporary.unlink(missing_ok=True)
            raise
        return dict(name=name, bytes=total, status='not_inspected')
    @app.get('/api/protocol/{name}', response_class=PlainTextResponse)
    def protocol(name: str):
        if name not in bootstrap()['protocols']: raise HTTPException(404)
        path = Path(__file__).resolve().parents[2]/'docs'/'protocols'/name
        if not path.is_file(): raise HTTPException(404, 'Документ находится в полной рабочей копии.')
        return path.read_text(encoding='utf-8')
    from .pipeline_ui import register
    register(app, queue)
    from .autonomous_ui import register as register_autonomous
    register_autonomous(app, autonomous, arktika_root)
    static = Path(__file__).with_name('static')
    app.mount('/static', StaticFiles(directory=static), name='static')
    @app.get('/')
    def index():
        text = (static/'index.html').read_text(encoding='utf-8')
        text = text.replace('GLOBAL-ML-WEATHER / 0.3', 'GLOBAL-ML-WEATHER / 0.4')
        text = text.replace('<b>Обученных весов нет</b>', '<b>Точность не подтверждена</b>')
        text = text.replace('</nav>', '</nav><a class="textlink" href="/training">Обучение и проверка выборки →</a>', 1)
        text = text.replace('</nav>', '<a href="/experiments">Реальные данные и обучение</a></nav>', 1)
        return HTMLResponse(text)
    return app


def main(argv=None):
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('outputs/lab'))
    parser.add_argument('--arktika-root', type=Path, help='Разрешённый каталог экспортированных продуктов arktika-worker; только чтение')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535: parser.error('Нужен непривилегированный порт.')
    uvicorn.run(create_app(args.workspace, arktika_root=args.arktika_root), host='127.0.0.1', port=args.port, workers=1, access_log=False)


if __name__ == '__main__': main()
