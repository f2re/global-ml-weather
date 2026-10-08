"""Browser test of prepared data, training, test evaluation and trained forecast."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from playwright.sync_api import sync_playwright, expect
from global_weather.pipeline.fixture import create_fixture

root=Path(__file__).resolve().parents[1]
target=root/'outputs/pipeline-browser';target.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    ws=Path(tmp)
    create_fixture(ws/'datasets/demo',horizon_hours=3)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    with (target/'server.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','global_weather.lab.app','--workspace',tmp,'--port',str(port)],cwd=root,stdout=log,stderr=log)
        try:
            for _ in range(200):
                try:urllib.request.urlopen(base+'/api/bootstrap',timeout=1).close();break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('Server did not start')
            with sync_playwright() as p:
                executable=os.environ.get('GLOBAL_WEATHER_CHROMIUM')
                if executable is not None and executable not in ('/usr/bin/chromium','/usr/bin/chromium-browser'):
                    raise ValueError('Use a verified system Chromium path for remote tests.')
                browser_env=dict(os.environ)
                compatibility=Path('/home/user/global-weather-browser-runtime/compat/usr/lib/x86_64-linux-gnu')
                if (compatibility/'libffi.so.6').is_file():
                    browser_env['LD_LIBRARY_PATH']=str(compatibility)
                browser=p.chromium.launch(headless=True, executable_path=executable, env=browser_env)
                page=browser.new_page(viewport={'width':1500,'height':1050});errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                try:
                    page.goto(base+'/training')
                    expect(page.locator('#dataset-info')).to_contain_text('СИНТЕТИЧЕСКИЕ')
                    page.select_option('#dataset','demo')
                    def submit(kind):
                        page.select_option('#action',kind)
                        with page.expect_response(lambda r:r.url.endswith('/api/pipeline/runs') and r.request.method=='POST') as response:
                            page.click('#submit-training')
                        assert response.value.status==200,response.value.text()
                        rid=response.value.json()['id']
                        end=time.monotonic()+90
                        while time.monotonic()<end:
                            status=page.request.get(base+'/api/runs/'+rid).json()
                            if status['status'] in ('completed','failed','cancelled','timed_out','interrupted'):break
                            page.wait_for_timeout(150)
                        assert status['status']=='completed',page.request.get(base+'/api/runs/'+rid+'/log').text()
                        expect(page.locator('#pipeline-state')).to_have_text('Завершено',timeout=10000)
                        return rid
                    submit('validate_dataset')
                    page.select_option('#action','train_dataset')
                    page.locator('#epochs').fill('1')
                    trained=submit('train_dataset')
                    expect(page.locator('#epoch-table')).to_contain_text('Ошибка обучения')
                    expect(page.locator('#trained-run option[value="'+trained+'"]')).to_have_count(1)
                    page.select_option('#action','evaluate_dataset')
                    page.select_option('#trained-run',trained)
                    evaluated=submit('evaluate_dataset')
                    expect(page.locator('#score-table')).to_contain_text('RMSE')
                    page.select_option('#action','forecast_dataset')
                    page.select_option('#sample','sample-4')
                    issued=submit('forecast_dataset')
                    report=page.request.get(base+'/api/runs/'+issued+'/report').json()
                    assert report['data_kind']=='synthetic' and report['targets_read'] is False
                    page.screenshot(path=str(target/'training-desktop.png'),full_page=True)
                    page.set_viewport_size({'width':390,'height':844})
                    assert page.locator('html').evaluate('(e)=>e.scrollWidth<=window.innerWidth')
                    page.screenshot(path=str(target/'training-mobile.png'),full_page=True)
                    page.set_viewport_size({'width':1500,'height':1050})
                    page.goto(base+'/?run='+issued)
                    expect(page.locator('#view-title')).to_contain_text('Прогноз по весам',timeout=15000)
                    expect(page.locator('#lead')).to_be_enabled()
                    page.locator('#lead').fill('3');page.locator('#lead').dispatch_event('change')
                    expect(page.locator('#lead-label')).to_have_text('+3 ч')
                    expect(page.locator('#experiments > .notice')).to_contain_text('синтетических')
                    assert len(page.request.get(base+'/api/runs/'+issued+'/frame?lead=3').json()['profile'])==37
                    page.screenshot(path=str(target/'trained-map.png'),full_page=True)
                    assert not errors,errors
                    (target/'check.json').write_text(json.dumps({'status':'passed','model':'trained_on_synthetic','page_errors':errors,'runs':[trained,evaluated,issued]},ensure_ascii=False))
                except Exception:
                    page.screenshot(path=str(target/'failure.png'),full_page=True)
                    (target/'failure.html').write_text(page.content())
                    (target/'page-errors.json').write_text(json.dumps(errors))
                    raise
                finally:browser.close()
        finally:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
