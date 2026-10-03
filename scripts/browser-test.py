"""End-to-end UI test against the real server; optional Playwright dependency."""
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

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    with open(Path(temporary)/'server.log','w') as log:
        process=subprocess.Popen([sys.executable,'-m','global_weather.lab.app','--workspace',temporary,'--port',str(port)],cwd=root,stdout=log,stderr=log)
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(base+'/api/bootstrap',timeout=1).close();break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('Server did not start.')
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True)
                page=browser.new_page(viewport={'width':1600,'height':1100})
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(base);expect(page.locator('#connection')).to_have_text('Сервер доступен')
                page.select_option('#kind','adaptive');page.select_option('#mesh','0');page.select_option('#horizon','6')
                page.click('#start');expect(page.locator('#run-state')).to_have_text('Завершено',timeout=60000)
                expect(page.locator('#lead')).to_be_enabled()
                page.locator('#lead').fill('3');page.locator('#lead').dispatch_event('change')
                page.select_option('#variable','temperature');page.select_option('#profile-var','2');page.wait_for_timeout(500)
                page.click('[data-page=sources]');assert page.locator('#connector-list .card').count()==6
                page.locator('#upload-file').set_input_files({'name':'sample.csv','mimeType':'text/csv','buffer':b'a,b\n1,2\n'})
                page.click('#upload');page.wait_for_selector('#inbox-list:text("sample.csv")')
                page.click('[data-page=agents]');assert page.locator('#agent-list .card').count()==9
                page.click('[data-page=protocols]');page.locator('#protocol-list button').first.click()
                expect(page.locator('#protocol-text')).to_contain_text('Научная постановка')
                page.click('[data-page=experiments]')
                target=root/'outputs'/'browser';target.mkdir(parents=True,exist_ok=True)
                page.screenshot(path=str(target/'desktop.png'),full_page=True)
                page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(300)
                assert page.locator('html').evaluate('(el) => el.scrollWidth <= window.innerWidth')
                page.screenshot(path=str(target/'mobile.png'),full_page=True)
                assert not errors,errors
                browser.close()
                print(json.dumps({'page_errors':errors,'mobile_overflow':False,'backend':'actual_local_http','model':'adaptive'},ensure_ascii=False))
        finally:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
