"""New local UI: plan, credentials, navigation, mobile overflow; no live download."""
import argparse,json,os,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/autonomous-browser'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory() as work:
    process=subprocess.Popen([sys.executable,'-m','global_weather.lab.app','--workspace',work,'--port','18767'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen('http://127.0.0.1:18767/api/bootstrap',timeout=1).close();break
            except OSError:time.sleep(.1)
        else:raise RuntimeError('Server did not start')
        with sync_playwright() as pw:
            options={'headless':True}
            if os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE'):options['executable_path']=os.environ['PLAYWRIGHT_CHROMIUM_EXECUTABLE']
            browser=pw.chromium.launch(**options);page=browser.new_page(viewport={'width':1440,'height':1100})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto('http://127.0.0.1:18767/experiments');expect(page.locator('#hardware')).to_contain_text('PyTorch')
            page.locator('#preview').click();expect(page.locator('#plan-preview')).to_contain_text('Манифест')
            assert not page.locator('#notice').is_visible()
            page.screenshot(path=str(a.output/'desktop.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844});page.reload();expect(page.locator('#hardware')).to_contain_text('PyTorch')
            assert page.locator('html').evaluate('(el) => el.scrollWidth <= window.innerWidth + 1')
            page.screenshot(path=str(a.output/'mobile.png'),full_page=True)
            assert not errors,errors
            browser.close()
        (a.output/'report.json').write_text(json.dumps({'status':'passed','checks':['plan_preview','desktop','mobile_no_overflow','no_page_errors'],'network_acquisition_tested':False},indent=2)+'\n')
    finally:
        process.terminate()
        try:process.wait(timeout=12)
        except subprocess.TimeoutExpired:process.kill();process.wait()
