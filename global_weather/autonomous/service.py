"""Persistent serial executor. Browser input never becomes a shell command."""
from __future__ import annotations
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
from ..pipeline.io import atomic_json, read_json
from .plan import parse_plan


class Credentials:
    fields={'cds_key','gptl_token','gptl_refresh','gptl_client_id','gptl_redirect_uri'}
    def __init__(self, path):self.path=Path(path);self.lock=threading.Lock()

    def read(self):
        if not self.path.exists():return {}
        if self.path.is_symlink() or self.path.stat().st_size>200000 or self.path.stat().st_mode&0o077:
            raise ValueError('Файл доступа должен иметь права 0600 и не быть ссылкой.')
        value=read_json(self.path)
        if not isinstance(value,dict) or set(value)-self.fields or any(not isinstance(v,str) for v in value.values()):raise ValueError('Неверный формат файла доступа.')
        return value

    def save(self, values):
        if not isinstance(values,dict) or set(values)-self.fields:raise ValueError('Неизвестные поля доступа.')
        for value in values.values():
            if not isinstance(value,str) or len(value)>65536 or any(c.isspace() or ord(c)<32 for c in value):
                raise ValueError('Введите одно значение токена без пробелов и переводов строк.')
        with self.lock:
            existing=self.read()
            if 'gptl_token' in values and 'gptl_refresh' not in values:existing.pop('gptl_refresh',None)
            existing.update(values)
            self.path.parent.mkdir(parents=True,exist_ok=True)
            if any(p.is_symlink() for p in (self.path,*self.path.parents)):raise ValueError('Ссылка вместо хранилища доступа.')
            temporary=self.path.with_name('.credentials-'+secrets.token_hex(8))
            try:
                fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'w') as f:json.dump(existing,f);f.flush();os.fsync(f.fileno())
                os.replace(temporary,self.path)
            finally:temporary.unlink(missing_ok=True)
        return self.status()

    def status(self):return {key:bool(value) for key,value in self.read().items()}


class Experiments:
    def __init__(self, workspace):
        self.root=Path(workspace).absolute()/'autonomous'
        if any(p.is_symlink() for p in (self.root,*self.root.parents)):raise ValueError('Недопустимый каталог экспериментов.')
        self.root.mkdir(parents=True,exist_ok=True)
        self.credentials=Credentials(self.root/'credentials.json')
        self.lock=threading.RLock();self.stop=threading.Event();self.thread=None;self.process=None;self.owner=None

    def directory(self, identity):
        import re
        if not re.fullmatch('[a-f0-9]{32}',identity):raise ValueError('Неверный идентификатор эксперимента.')
        p=self.root/identity
        if p.is_symlink() or not p.is_dir():raise ValueError('Эксперимент не найден.')
        return p

    def create(self, value):
        spec=parse_plan(value);plan=spec.checked()
        with self.lock:
            if sum(r['status'] in ('queued','running') for r in self.list())>=8:raise ValueError('Очередь экспериментов заполнена.')
            identity=uuid.uuid4().hex;d=self.root/identity;d.mkdir()
            atomic_json(d/'request.json',asdict(spec));atomic_json(d/'status.json',{'id':identity,'status':'queued','created':time.time()})
        return self.get(identity)

    def get(self, identity):
        d=self.directory(identity);result=read_json(d/'status.json')
        result['request']=read_json(d/'request.json')
        p=d/'work/progress.json'
        result['progress']=read_json(p) if p.exists() else None
        p=d/'work/result.json'
        result['result']=read_json(p) if p.exists() else None
        return result

    def list(self):
        dirs=sorted((p for p in self.root.iterdir() if p.is_dir() and not p.is_symlink() and len(p.name)==32),key=lambda p:p.stat().st_mtime,reverse=True)
        return [self.get(p.name) for p in dirs[:200] if (p/'status.json').exists()]

    def cancel(self, identity):
        with self.lock:
            d=self.directory(identity);row=read_json(d/'status.json')
            if row['status'] in ('queued','running'):
                (d/'cancel').touch(exist_ok=True)
                if row['status']=='queued':row['status']='cancelled';atomic_json(d/'status.json',row)
        return self.get(identity)

    def resume(self, identity):
        with self.lock:
            d=self.directory(identity);row=read_json(d/'status.json')
            if row['status'] not in ('failed','cancelled','interrupted','timed_out'):raise ValueError('Этот эксперимент нельзя продолжить.')
            parse_plan(read_json(d/'request.json')).checked()
            (d/'cancel').unlink(missing_ok=True)
            row.update(status='queued',reason=None);atomic_json(d/'status.json',row)
        return self.get(identity)

    def start(self):
        if self.thread:return
        self.owner=(self.root/'service.lock').open('a')
        try:fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close();self.owner=None;raise ValueError('Другой исполнитель уже владеет каталогом.')
        for r in self.list():
            if r['status']=='running':
                row=read_json(self.directory(r['id'])/'status.json');row['status']='interrupted';atomic_json(self.directory(r['id'])/'status.json',row)
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()

    def close(self):
        self.stop.set()
        if self.process and self.process.poll() is None:
            try:os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        if self.thread:self.thread.join(timeout=10)
        if self.process and self.process.poll() is None:
            try:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=3)
            except ProcessLookupError:pass
        if self.thread:self.thread.join(timeout=3)
        if self.owner:self.owner.close();self.owner=None

    def loop(self):
        while not self.stop.wait(.3):
            job=next((r for r in reversed(self.list()) if r['status']=='queued'),None)
            if not job:continue
            d=self.directory(job['id']);package=Path(__file__).resolve().parents[2]
            with self.lock:
                row=read_json(d/'status.json')
                if row['status']!='queued':continue
                row.update(status='running',started=time.time());atomic_json(d/'status.json',row)
            try:
                env={k:os.environ[k] for k in ('PATH','HOME','LANG','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','HTTPS_PROXY','HTTP_PROXY','NO_PROXY','SSL_CERT_FILE','REQUESTS_CA_BUNDLE') if k in os.environ}
                env.update(PYTHONPATH=str(package),PYTHONUNBUFFERED='1')
                command=[sys.executable,'-m','global_weather.autonomous','worker','--job',str(d),'--credentials',str(self.credentials.path)]
                with (d/'execution.log').open('ab') as log:
                    self.process=subprocess.Popen(command,cwd=package,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    began=time.monotonic();cancelled_at=None;timed_out=False
                    while self.process.poll() is None:
                        if (d/'cancel').exists() or self.stop.is_set() or time.monotonic()-began>job['request']['max_runtime_hours']*3600:
                            timed_out=time.monotonic()-began>job['request']['max_runtime_hours']*3600
                            if cancelled_at is None:
                                cancelled_at=time.monotonic();os.killpg(self.process.pid,signal.SIGTERM)
                            elif time.monotonic()-cancelled_at>5:os.killpg(self.process.pid,signal.SIGKILL)
                        if (d/'execution.log').stat().st_size>32*1024**2:
                            os.killpg(self.process.pid,signal.SIGTERM)
                        time.sleep(.2)
                    code=self.process.wait()
                row.update(status='interrupted' if self.stop.is_set() else 'timed_out' if timed_out else 'cancelled' if (d/'cancel').exists() else 'completed' if code==0 else 'failed',exit_code=code,finished=time.time())
                error=d/'error.json'
                if error.exists():row['reason']=read_json(error).get('reason')
            except Exception:
                row.update(status='failed',reason='Не удалось запустить исполнителя; проверьте журнал и окружение.')
            finally:
                self.process=None;atomic_json(d/'status.json',row)
