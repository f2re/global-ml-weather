"""Durable single-writer optimizer steps, with the SQLite head as authority.

Only trusted Python training code may call this module. It does not download,
certify data, or provide an HTTP method for marking an example as trained.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import stat
import threading
import uuid
import zipfile

import numpy as np
import torch

from .contracts import canonical, fingerprint
from ..devices import device_identity


class CheckpointError(ValueError):
    """Fail closed: neither old weights with a new cursor nor silent reset."""


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def regular(path):
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CheckpointError('Контрольная точка должна быть обычным файлом без ссылок.')
    return info


def safe_directory(path):
    for item in (Path(path), *Path(path).parents):
        info = item.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise CheckpointError('Недопустимый каталог контрольных точек.')


@contextmanager
def checked_file(path, maximum):
    regular(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= maximum:
            raise CheckpointError('Файл контрольной точки пуст, слишком велик или небезопасен.')
        yield stream


def finite_copy(value):
    """CPU-only serializable state, without executable classes or NaN."""
    if isinstance(value, torch.Tensor):
        v = value.detach().cpu().clone()
        checked = v.coalesce().values() if v.is_sparse else v
        if not torch.isfinite(checked).all():
            raise CheckpointError('Неконечный тензор в состоянии обучения.')
        return v
    if isinstance(value, dict):
        result = type(value)((k, finite_copy(v)) for k, v in value.items())
        if hasattr(value, '_metadata'):
            result._metadata = finite_copy(value._metadata)
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(finite_copy(v) for v in value)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise CheckpointError('Неподдерживаемое или неконечное состояние обучения.')


def rng_state(device):
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (n[0], torch.from_numpy(n[1].astype(np.int64)), *n[2:]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}


def restore_rng(state, device):
    n = state['numpy']
    numpy_state = (n[0], n[1].numpy().astype(np.uint32), *n[2:])
    # Validate CPU RNG formats using isolated generators before mutating globals.
    random.Random().setstate(state['python'])
    np.random.RandomState().set_state(numpy_state)
    torch.Generator().set_state(state['torch'])
    if device.type == 'cuda' and len(state['cuda']) != torch.cuda.device_count():
        raise CheckpointError('Изменилось число генераторов CUDA.')
    random.setstate(state['python'])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda'])


def typename(value):
    return None if value is None else type(value).__module__ + '.' + type(value).__qualname__


def tensor_signature(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'layout': str(value.layout)}
    if isinstance(value, dict):
        return {k: tensor_signature(v) for k, v in value.items()}
    return value


class StepTrainer:
    """One step callback, one optimizer application, one committed generation.

    The callback is trusted application code, not a user-supplied module name.
    It must finish one optimizer step. The executor clears gradients at both
    step boundaries. Explicit scheduler state and optional scaler are retained.
    After an exception the session is poisoned and must be reopened.
    """

    def __init__(self, store, model, optimizer, *, identity, scheduler=None, scaler=None,
                 max_checkpoint_bytes=512*1024**2, failpoint=None):
        if not isinstance(identity, dict) or identity.get('data_kind') not in ('real', 'synthetic'):
            raise ValueError('Нужно явно указать происхождение данных в контракте исполнения.')
        if len(canonical(identity).encode()) > 65536:
            raise ValueError('Контракт исполнения слишком велик.')
        if type(max_checkpoint_bytes) is not int or max_checkpoint_bytes <= 0:
            raise ValueError('Неверный предел размера контрольной точки.')
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)):
            raise ValueError('Для этого исполнителя проверены Adam, AdamW и SGD.')
        self.store, self.model, self.optimizer = store, model, optimizer
        self.scheduler, self.scaler = scheduler, scaler
        self.maximum = max_checkpoint_bytes
        self.failpoint = failpoint or (lambda point: None)
        self.directory = store.root / 'checkpoints'
        self.active = False
        self.poisoned = False
        self.lock = None
        self.current = None
        parameters = list(model.parameters())
        if not parameters:
            raise ValueError('Модель не содержит параметров.')
        self.device = parameters[0].device
        if any(p.device != self.device for p in parameters) or any(b.device != self.device for b in model.buffers()):
            raise ValueError('Модель должна находиться на одном устройстве.')
        names = {id(p): n for n, p in model.named_parameters()}
        try:
            groups = [{**{k: v for k, v in g.items() if k != 'params'},
                       'params': [names[id(p)] for p in g['params']]} for g in optimizer.param_groups]
        except KeyError as exc:
            raise ValueError('Оптимизатор содержит параметры другой модели.') from exc
        from ..pipeline.runner import software
        self.identity = {'application': identity, 'model_type': typename(model),
                         'state_schema': tensor_signature(model.state_dict()),
                         'module_config': {n: m.extra_repr() for n, m in model.named_modules()},
                         'optimizer_type': typename(optimizer), 'optimizer_groups': groups,
                         'scheduler_type': typename(scheduler), 'scaler_type': typename(scaler),
                         'software': software(), 'runtime': device_identity(self.device),
                         'threads': torch.get_num_threads(),
                         'deterministic': torch.are_deterministic_algorithms_enabled(),
                         'cudnn_benchmark': torch.backends.cudnn.benchmark,
                         'cudnn_deterministic': torch.backends.cudnn.deterministic,
                         'matmul_precision': torch.get_float32_matmul_precision(),
                         'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                         'cudnn_tf32': torch.backends.cudnn.allow_tf32}
        # Normalize tuples before comparing to persisted JSON.
        self.identity = json.loads(canonical(self.identity))

    def __enter__(self):
        if self.active or self.poisoned:
            raise CheckpointError('Создайте новую сессию исполнителя.')
        self.store._safe_paths()
        self.directory.mkdir(mode=0o700, exist_ok=True)
        safe_directory(self.directory)
        for path in (self.store.root, self.store.root.parent, self.store.root.parent.parent):
            fsync_directory(path)
        fd = os.open(self.store.root/'trainer.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self.lock = os.fdopen(fd, 'a+b')
        try:
            regular(self.store.root/'trainer.lock')
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CheckpointError('Другой исполнитель уже обучает эту программу.') from exc
            self.pid = os.getpid()
            self.thread_id = threading.get_ident()
            self.token = uuid.uuid4().hex
            with self.store.transaction(write=True) as db:
                campaign = self.store._campaign(db)
                if campaign is None:
                    raise CheckpointError('Сначала зарегистрируйте диапазон программы.')
                self.identity['campaign_contract'] = campaign['contract_hash']
                self.identity_hash = fingerprint(self.identity)
                old = db.execute('SELECT * FROM trainer_contract WHERE singleton=1').fetchone()
                if old is not None and (old['identity_hash'] != self.identity_hash
                                        or old['identity_json'] != canonical(self.identity)):
                    raise CheckpointError('Изменились модель, нормы, оптимизатор или численная среда.')
                if old is None:
                    db.execute('INSERT INTO trainer_contract VALUES (1,?,?)',
                               (canonical(self.identity), self.identity_hash))
                db.execute('''INSERT INTO trainer_owner VALUES (1,1,?)
                    ON CONFLICT(singleton) DO UPDATE SET epoch=epoch+1,token=excluded.token''', (self.token,))
            self.active = True
            self.current = self._head()
            if self.current is None:
                self._publish([], None, {})  # Durable initial state before the first gradient.
            else:
                self._restore()
            self.prune()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        self.active = False
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def _owned(self, db):
        if not self.active or self.poisoned or os.getpid() != self.pid or self.lock is None or threading.get_ident() != self.thread_id:
            raise CheckpointError('Сессия завершена или требует восстановления после ошибки.')
        self.store._safe_paths()
        safe_directory(self.directory)
        a, b = regular(self.store.root/'trainer.lock'), os.fstat(self.lock.fileno())
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            raise CheckpointError('Файл владения исполнителем заменён.')
        row = db.execute('SELECT token FROM trainer_owner WHERE singleton=1').fetchone()
        if row is None or row['token'] != self.token:
            raise CheckpointError('Исполнитель утратил владение программой.')
        head = db.execute('SELECT generation_id FROM checkpoint_head WHERE singleton=1').fetchone()
        expected = None if self.current is None else self.current['id']
        if (None if head is None else head[0]) != expected:
            raise CheckpointError('Контрольная точка изменилась у другого исполнителя.')

    def _head(self):
        with self.store.transaction() as db:
            row = db.execute('''SELECT manifest_json FROM checkpoint_head h
                JOIN checkpoint_details d ON d.generation_id=h.generation_id WHERE h.singleton=1''').fetchone()
            count, last = db.execute('SELECT count(*),max(step_number) FROM checkpoint_details').fetchone()
            if row is None:
                if count:
                    raise CheckpointError('Утрачен текущий указатель при существующей истории шагов.')
                return None
            manifest = json.loads(row[0])
            uses = db.execute('SELECT count(*) FROM training_events').fetchone()[0]
            if (manifest['step_number'] != last or count != last+1
                    or manifest['committed_uses_total'] != uses):
                raise CheckpointError('Веса и журнал применений не согласованы; продолжение запрещено.')
            return manifest

    def _capture(self):
        self.optimizer.zero_grad(set_to_none=True)
        return finite_copy({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                            'scaler': self.scaler.state_dict() if self.scaler else None,
                            'rng': rng_state(self.device),
                            'training_modes': {n: m.training for n, m in self.model.named_modules()}})

    def _read(self, manifest, *, load=True):
        identity = manifest['id']
        with self.store.transaction() as db:
            registered = db.execute('SELECT manifest_hash FROM checkpoint_generations WHERE id=?', (identity,)).fetchone()
        if registered is None or registered[0] != hashlib.sha256(canonical(manifest).encode()).hexdigest():
            raise CheckpointError('Паспорт поколения повреждён в журнале.')
        if not re.fullmatch('[a-f0-9]{32}', identity):
            raise CheckpointError('Недопустимая контрольная точка в журнале.')
        folder = self.directory/identity
        try:
            safe_directory(folder)
            with checked_file(folder/'manifest.json', 1024**2) as f:
                if f.read() != canonical(manifest).encode():
                    raise CheckpointError('Паспорт контрольной точки не совпадает с журналом.')
            with checked_file(folder/'state.pt', self.maximum) as f:
                digest = hashlib.sha256()
                while chunk := f.read(1024**2):
                    digest.update(chunk)
                if digest.hexdigest() != manifest['state_sha256'] or f.tell() != manifest['state_bytes']:
                    raise CheckpointError('Контрольная сумма состояния обучения не совпадает.')
                f.seek(0)
                with zipfile.ZipFile(f) as z:
                    if sum(x.file_size for x in z.infolist()) > self.maximum:
                        raise CheckpointError('Состояние превышает бюджет распаковки.')
                if not load:
                    return None
                f.seek(0)
                state = torch.load(f, map_location='cpu', weights_only=True)
        except (OSError, zipfile.BadZipFile) as exc:
            raise CheckpointError('Текущая контрольная точка недоступна. Нужна согласованная резервная копия.') from exc
        if manifest['identity_hash'] != self.identity_hash:
            raise CheckpointError('Контрольная точка имеет другой контракт исполнения.')
        finite_copy(state)
        return state

    def _restore(self):
        state = self._read(self.current)
        expected = self.model.state_dict()
        incoming = state['model']
        if incoming.keys() != expected.keys() or tensor_signature(incoming) != tensor_signature(expected):
            raise CheckpointError('Ключи, формы или типы модели не совпадают.')
        # Project weather models keep grid and normalization in immutable buffers.
        if type(self.model).__module__.startswith('global_weather.'):
            for name, reference in self.model.named_buffers():
                a, b = reference.detach().cpu(), incoming[name]
                same = (torch.equal(a.coalesce().indices(), b.coalesce().indices())
                        and torch.equal(a.coalesce().values(), b.coalesce().values())) if a.is_sparse else torch.equal(a, b)
                if not same:
                    raise CheckpointError('Изменены фиксированные сетка или нормы модели.')
        groups = state['optimizer']['param_groups']
        if len(groups) != len(self.optimizer.param_groups) or any(
            len(a['params']) != len(b['params']) for a, b in zip(groups, self.optimizer.param_groups)):
            raise CheckpointError('Не совпадает состав оптимизатора.')
        # Built-in supported optimizers use parameter-shaped moments and scalar step.
        for saved, live in zip(groups, self.optimizer.param_groups):
            for key, parameter in zip(saved['params'], live['params']):
                for name, value in state['optimizer']['state'].get(key, {}).items():
                    if isinstance(value, torch.Tensor) and name != 'step' and value.shape != parameter.shape:
                        raise CheckpointError('Повреждена форма состояния оптимизатора.')
        self.model.load_state_dict(incoming, strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        if self.scaler is not None:
            self.scaler.load_state_dict(state['scaler'])
        for name, module in self.model.named_modules():
            module.training = state['training_modes'][name]
        restore_rng(state['rng'], self.device)
        self.optimizer.zero_grad(set_to_none=True)

    def _check_uses(self, db, use_ids):
        committed = []
        for use in use_ids:
            row = db.execute('''SELECT role FROM usage_plan u JOIN samples s ON s.id=u.sample_id
                JOIN assignments a USING(issue_time) WHERE u.id=?''', (use,)).fetchone()
            if row is None or row['role'] != 'train':
                raise CheckpointError('Пример отсутствует или является контрольным.')
            old = db.execute('SELECT generation_id FROM training_events WHERE use_id=?', (use,)).fetchone()
            if old:
                committed.append(old[0])
        if committed and len(committed) != len(use_ids):
            raise CheckpointError('Часть пакета уже обучена; разделите работу по журналу.')
        return committed

    def step(self, use_ids, operation, *, cursor):
        if not isinstance(use_ids, (list, tuple)) or not 1 <= len(use_ids) <= 4096:
            raise ValueError('Нужен ограниченный непустой пакет зарегистрированных применений.')
        if any(not isinstance(u, str) or not re.fullmatch('[a-f0-9]{64}', u) for u in use_ids) or len(set(use_ids)) != len(use_ids):
            raise ValueError('Неверные или повторные идентификаторы применений.')
        if not isinstance(cursor, dict) or len(canonical(cursor).encode()) > 65536:
            raise ValueError('Курсор должен быть ограниченным объектом JSON.')
        with self.store.transaction() as db:
            self._owned(db)
            done = self._check_uses(db, use_ids)
        if done:
            return {'status': 'already_committed', 'generation_ids': done}
        from ..pipeline.runner import software
        if software() != self.identity['software']:
            raise CheckpointError('Исходный код или численная среда изменились во время обучения.')
        applications = []
        hook = self.optimizer.register_step_post_hook(lambda *args: applications.append(True))
        try:
            self.optimizer.zero_grad(set_to_none=True)
            metrics = operation()
            if len(applications) != 1:
                raise CheckpointError('Операция должна завершить ровно один шаг оптимизатора.')
            if not isinstance(metrics, dict) or len(canonical(metrics).encode()) > 65536:
                raise CheckpointError('Метрики шага должны быть конечным ограниченным объектом JSON.')
            if self.scheduler is not None:
                self.scheduler.step()
            self.failpoint('after_optimizer')
            if software() != self.identity['software']:
                raise CheckpointError('Исходный код изменился во время шага.')
            self._publish(list(use_ids), cursor, metrics)
            self.prune()
            return {'status': 'committed', 'generation_id': self.current['id'],
                    'step_number': self.current['step_number'], 'metrics': metrics}
        except BaseException:
            # A possibly applied but uncommitted gradient must not survive into
            # the next callback. Reopening restores the authoritative generation.
            self.poisoned = True
            raise
        finally:
            hook.remove()

    def _publish(self, uses, cursor, metrics):
        state = self._capture()
        estimated = sum(v.numel()*v.element_size() for v in self.model.state_dict().values() if isinstance(v, torch.Tensor))
        if shutil.disk_usage(self.store.root).free < max(32*1024**2, estimated*4):
            raise CheckpointError('Недостаточно места для безопасного сохранения состояния.')
        identity = uuid.uuid4().hex
        pending, destination = self.directory/('.pending-'+identity), self.directory/identity
        pending.mkdir(mode=0o700)
        # Never overwrite a published generation. An orphan is cleaned only by
        # a subsequent successful owner, after verifying the committed head.
        with (pending/'state.pt').open('xb') as f:
            os.chmod(pending/'state.pt', 0o600)
            torch.save(state, f)
            self.failpoint('after_state_write')
            f.flush()
            os.fsync(f.fileno())
        size = regular(pending/'state.pt').st_size
        if size > self.maximum:
            raise CheckpointError('Состояние превышает разрешённый размер контрольной точки.')
        with checked_file(pending/'state.pt', self.maximum) as f:
            digest = hashlib.sha256()
            while chunk := f.read(1024**2):
                digest.update(chunk)
        manifest = {'schema': 'continuous-checkpoint-1', 'id': identity,
                    'identity_hash': self.identity_hash,
                    'parent_id': None if self.current is None else self.current['id'],
                    'step_number': 0 if self.current is None else self.current['step_number']+1,
                    'state_bytes': size, 'state_sha256': digest.hexdigest(),
                    'cursor': cursor, 'use_ids': uses, 'metrics': metrics,
                    'committed_uses_total': len(uses)+(self.current['committed_uses_total'] if self.current else 0)}
        data = canonical(manifest).encode()
        with (pending/'manifest.json').open('xb') as f:
            os.chmod(pending/'manifest.json', 0o600)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        fsync_directory(pending)
        pending.rename(destination)
        fsync_directory(self.directory)
        self.failpoint('after_publish')
        with self.store.transaction(write=True) as db:
            self._owned(db)
            if self._check_uses(db, uses):
                raise CheckpointError('Примеры уже применены другим шагом.')
            db.execute('INSERT INTO checkpoint_generations VALUES (?,?,?)',
                       (identity, hashlib.sha256(data).hexdigest(), self.store._now().isoformat()))
            db.execute('INSERT INTO checkpoint_details VALUES (?,?,?,?)',
                       (identity, manifest['step_number'], manifest['parent_id'], data.decode()))
            for use in uses:
                db.execute('INSERT INTO training_events(use_id,generation_id) VALUES (?,?)', (use, identity))
            db.execute('''INSERT INTO checkpoint_head VALUES (1,?)
                ON CONFLICT(singleton) DO UPDATE SET generation_id=excluded.generation_id''', (identity,))
            self.store._event(db, 'optimizer_step_committed' if uses else 'model_initialized',
                              {'generation_id': identity, 'step_number': manifest['step_number'], 'uses': len(uses)})
            self.failpoint('before_commit')
        self.current = manifest
        self.failpoint('after_commit')

    def prune(self):
        """Keep current/previous and explicit pins; ledger rows remain immutable."""
        with self.store.transaction() as db:
            self._owned(db)
            keep = {r[0] for r in db.execute('SELECT generation_id FROM checkpoint_details ORDER BY step_number DESC LIMIT 2')}
            keep.update(r[0] for r in db.execute('SELECT generation_id FROM checkpoint_pins'))
        # Verify both retained recovery generations before any file removal.
        with self.store.transaction() as db:
            for identity in keep:
                row = db.execute('SELECT manifest_json FROM checkpoint_details WHERE generation_id=?', (identity,)).fetchone()
                self._read(json.loads(row[0]), load=False)
        for child in self.directory.iterdir():
            if child.name in keep:
                continue
            if not re.fullmatch(r'(?:\.pending-)?[a-f0-9]{32}', child.name):
                continue
            safe_directory(child)
            for path in child.iterdir():
                if path.name not in ('state.pt', 'manifest.json'):
                    raise CheckpointError('Неизвестные файлы в каталоге поколения; очистка остановлена.')
                regular(path)
            shutil.rmtree(child)
        fsync_directory(self.directory)

    def pin(self):
        """Internal explicit retention of a selected checkpoint or stage boundary."""
        with self.store.transaction(write=True) as db:
            self._owned(db)
            db.execute('INSERT OR IGNORE INTO checkpoint_pins VALUES (?)', (self.current['id'],))

    def backup(self, destination):
        """A separate consistent workspace, not a backup of weather source arrays.

        Do not run another optimizer thread in this session. Range registration
        may proceed; SQLite's backup API provides a consistent database snapshot.
        """
        with self.store.transaction() as db:
            self._owned(db)
        self.prune()
        destination = Path(destination).absolute()
        safe_directory(destination.parent)
        if destination.is_relative_to(self.store.root.parent):
            raise CheckpointError('Резервная копия должна находиться вне активного рабочего каталога.')
        if destination.exists() or destination.is_symlink():
            raise CheckpointError('Каталог резервной копии уже существует.')
        temporary = destination.with_name('.backup-'+uuid.uuid4().hex)
        temporary.mkdir(mode=0o700)
        root = temporary/'continuous'
        root.mkdir(mode=0o700)
        try:
            with self.store._connection() as source, sqlite3.connect(root/'learning.sqlite3') as target:
                source.backup(target)
            # sqlite3's context manager commits but does not close a connection.
            target.close()
            os.chmod(root/'learning.sqlite3', 0o600)
            folders = root/'checkpoints'
            folders.mkdir(mode=0o700)
            for folder in self.directory.iterdir():
                safe_directory(folder)
                if not re.fullmatch('[a-f0-9]{32}', folder.name):
                    continue
                out = folders/folder.name
                out.mkdir(mode=0o700)
                for name in ('state.pt', 'manifest.json'):
                    with checked_file(folder/name, self.maximum) as src, (out/name).open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush()
                        os.fsync(dst.fileno())
                    os.chmod(out/name, 0o600)
                fsync_directory(out)
            with (root/'learning.sqlite3').open('rb') as f:
                os.fsync(f.fileno())
            fsync_directory(folders)
            fsync_directory(root)
            fsync_directory(temporary)
            temporary.rename(destination)
            fsync_directory(destination.parent)
            return {'status': 'checkpoint_and_ledger_backed_up', 'generation_id': self.current['id'],
                    'source_weather_arrays_included': False}
        except BaseException:
            # The failed copy is never exposed as a completed destination.
            shutil.rmtree(temporary)
            raise
