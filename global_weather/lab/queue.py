"""Persistent single-worker queue with cancellation and interruption recovery."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from .contracts import RunSpec, atomic_json, safe_child, now

TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted', 'timed_out'}


class RunQueue:
    def __init__(self, root, *, timeout=300):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.runs = self.root/'runs'; self.runs.mkdir(exist_ok=True)
        self.inbox = self.root/'inbox'; self.inbox.mkdir(exist_ok=True)
        self.timeout = timeout; self.stop = threading.Event(); self.process = None; self.thread = None
        self.guard = threading.Lock(); self.active_id = None; self.lockfile = None
        self.database = self.root/'experiments.sqlite'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, status TEXT NOT NULL, spec TEXT NOT NULL, created TEXT NOT NULL, started TEXT, finished TEXT, exit_code INTEGER, reason TEXT)')

    def connect(self):
        conn = sqlite3.connect(self.database, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def start(self):
        if self.thread: return
        self.lockfile = (self.root/'worker.lock').open('a')
        try: fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lockfile.close(); self.lockfile = None
            raise RuntimeError('Каталог уже используется другим стендом. Нужен один процесс uvicorn.')
        with self.connect() as db:
            db.execute("UPDATE runs SET status='interrupted', finished=?, reason='Сервер был остановлен; автоматический повтор запрещён.' WHERE status IN ('running','queued')", (now(),))
        self.thread = threading.Thread(target=self._loop, daemon=True, name='experiment-worker'); self.thread.start()

    def close(self):
        self.stop.set()
        with self.guard:
            if self.process and self.process.poll() is None: self._terminate(self.process)
        if self.thread: self.thread.join(timeout=8)
        if self.lockfile:
            fcntl.flock(self.lockfile, fcntl.LOCK_UN); self.lockfile.close()
        self.thread = None

    def create(self, spec: RunSpec):
        if spec.kind == 'inspect' and not safe_child(self.inbox, spec.input_file).is_file():
            raise ValueError('Выберите существующий файл из входного каталога.')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT count(*) FROM runs WHERE status IN ('queued','running')").fetchone()[0] >= 6:
                raise ValueError('Очередь заполнена: максимум шесть испытаний.')
            run_id = uuid.uuid4().hex
            directory = self.runs/run_id; directory.mkdir()
            atomic_json(directory/'request.json', spec.model_dump())
            db.execute('INSERT INTO runs(id,status,spec,created) VALUES (?,?,?,?)', (run_id, 'queued', spec.model_dump_json(), now()))
        return self.get(run_id)

    def get(self, run_id):
        safe_child(self.runs, run_id)
        with self.connect() as db: row = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None: raise KeyError(run_id)
        data = dict(row); data['spec'] = json.loads(data['spec'])
        return data

    def list(self):
        with self.connect() as db: ids = [r[0] for r in db.execute('SELECT id FROM runs ORDER BY created DESC LIMIT 200')]
        return [self.get(i) for i in ids]

    def cancel(self, run_id):
        with self.guard:
            current = self.get(run_id)
            if current['status'] in TERMINAL: return current
            with self.connect() as db:
                db.execute("UPDATE runs SET status='cancelled', finished=?, reason='Остановлено пользователем.' WHERE id=?", (now(), run_id))
            if self.active_id == run_id and self.process and self.process.poll() is None: self._terminate(self.process)
        return self.get(run_id)

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=2)
        except ProcessLookupError: pass

    def _loop(self):
        while not self.stop.wait(.1):
            with self.connect() as db:
                row = db.execute("SELECT id FROM runs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None: continue
            run_id = row['id']
            try: self._execute(run_id)
            except Exception as exc:
                with self.connect() as db:
                    db.execute("UPDATE runs SET status='failed', finished=?, reason=? WHERE id=? AND status NOT IN ('cancelled','interrupted')", (now(), type(exc).__name__ + ': ошибка исполнения; см. протокол.', run_id))

    def _execute(self, run_id):
        directory = self.runs/run_id
        package_root = Path(__file__).resolve().parents[2]
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LD_LIBRARY_PATH', 'SYSTEMROOT') if k in os.environ}
        home = directory/'home'; home.mkdir(exist_ok=True)
        env.update(HOME=str(home), PYTHONPATH=str(package_root), PYTHONUNBUFFERED='1',
                   CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        reason = None; final = None
        with (directory/'execution.log').open('wb') as log:
            with self.guard:
                if self.get(run_id)['status'] != 'queued': return
                with self.connect() as db: db.execute("UPDATE runs SET status='running', started=? WHERE id=?", (now(), run_id))
                command = [sys.executable, '-m', 'global_weather.lab.dispatch', '--run-dir', str(directory), '--inbox', str(self.inbox)]
                self.process = subprocess.Popen(command, cwd=package_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, start_new_session=True, shell=False)
                self.active_id = run_id; process = self.process
            start = time.monotonic()
            while process.poll() is None:
                if self.stop.is_set(): final, reason = 'interrupted', 'Сервер остановлен.'
                elif time.monotonic()-start > self.timeout: final, reason = 'timed_out', 'Превышен предел времени.'
                elif (directory/'execution.log').stat().st_size > 2*1024*1024: final, reason = 'failed', 'Превышен предел журнала.'
                if final:
                    self._terminate(process); break
                time.sleep(.1)
            code = process.wait()
        with self.guard:
            self.process = None; self.active_id = None
            with self.connect() as db:
                db.execute("UPDATE runs SET status=?, finished=?, exit_code=?, reason=? WHERE id=? AND status='running'",
                           (final or ('completed' if code == 0 else 'failed'), now(), code, reason, run_id))
        atomic_json(directory/'execution.json', self.get(run_id))
