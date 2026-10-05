"""Local UI routes: explicit network consent, no caller-controlled paths/commands."""
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from ..autonomous.plan import parse_plan
from ..devices import inventory
from ..providers.satellites import SatelliteCatalog


def register(app, service, arktika_root=None):
    catalog=SatelliteCatalog(service.root/'satellites',arktika_root,service.credentials.read,service.credentials.save)
    app.state.satellites=catalog
    @app.get('/experiments',response_class=HTMLResponse)
    def page():return (Path(__file__).with_name('static')/'experiments.html').read_text(encoding='utf-8')
    @app.get('/api/autonomous/state')
    def state():return {'hardware':inventory(),'credentials':service.credentials.status(),'arktika_configured':arktika_root is not None,'runs':service.list()}
    @app.post('/api/autonomous/credentials')
    def credentials(value:dict):return service.credentials.save(value)
    @app.post('/api/autonomous/plan')
    def plan(value:dict):return parse_plan(value).checked()
    @app.post('/api/autonomous/runs')
    def create(value:dict):return service.create(value)
    @app.get('/api/autonomous/runs/{identity}')
    def get(identity:str):return service.get(identity)
    @app.post('/api/autonomous/runs/{identity}/cancel')
    def cancel(identity:str):return service.cancel(identity)
    @app.post('/api/autonomous/runs/{identity}/resume')
    def resume(identity:str):return service.resume(identity)
    @app.get('/api/autonomous/runs/{identity}/log')
    def log(identity:str):
        p=service.directory(identity)/'execution.log'
        if not p.exists():return {'text':''}
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-50000));text=f.read().decode('utf-8',errors='replace')
        for v in service.credentials.read().values():
            if v:text=text.replace(v,'[СКРЫТО]')
        return {'text':text}
    @app.get('/api/autonomous/runs/{identity}/evaluation')
    def evaluation(identity:str):
        from ..pipeline.io import read_json
        p=service.directory(identity)/'work/evaluation/evaluation.json'
        if not p.exists():raise HTTPException(409,'Независимая проверка ещё не завершена.')
        return read_json(p)
    @app.post('/api/satellites/collections')
    def collections(value:dict):return catalog.collections(network=value.get('network') is True)
    @app.post('/api/satellites/search')
    def search(value:dict):
        allowed={'start','end','platform','collection','network','max_pages'}
        if set(value)-allowed:raise ValueError('Неизвестные поля поиска.')
        return catalog.search(**value)
    @app.post('/api/satellites/download')
    def download(value:dict):return catalog.download(value['id'],network=value.get('network') is True)
    @app.get('/api/satellites/local')
    def local(day:str='',platform:str=''):return catalog.browse_local(day=day,platform=platform)
    @app.post('/api/satellites/import')
    def import_local(value:dict):return catalog.import_local(value['id'])
