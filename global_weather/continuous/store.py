"""Transactional campaign/range registry and the C1 data-use ledger.

This module never trains, downloads, or declares an example trained. C2 must
publish checkpoints and usage events in one coordinated commit before the
reserved training ledger can acquire entries. The old trainer is unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
import os
from pathlib import Path
import sqlite3
import stat
import uuid

from .contracts import (canonical, fingerprint, check_range, calendar_date,
                        default_contract, sample_identity, temporal_role, utc_time,
                        identifier, SPLIT_VERSION)

APPLICATION_ID = 0x47574331
SCHEMA_VERSION = 1

DDL = (
    '''CREATE TABLE campaign (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1), id TEXT UNIQUE NOT NULL,
        contract_json TEXT NOT NULL, contract_hash TEXT NOT NULL, created_at TEXT NOT NULL)''',
    '''CREATE TABLE ranges (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
        request_key TEXT UNIQUE NOT NULL, start_day INTEGER NOT NULL, end_day INTEGER NOT NULL,
        new_days INTEGER NOT NULL CHECK(new_days>=0), created_at TEXT NOT NULL,
        CHECK(start_day<=end_day))''',
    '''CREATE TABLE intervals (
        start_day INTEGER PRIMARY KEY, end_day INTEGER NOT NULL CHECK(start_day<=end_day))''',
    '''CREATE TABLE assignments (
        issue_time TEXT PRIMARY KEY, nominal_role TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('train','validation','test','guard')))''',
    '''CREATE TABLE assets (
        id TEXT PRIMARY KEY, descriptor_json TEXT NOT NULL UNIQUE)''',
    '''CREATE TABLE samples (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
        issue_time TEXT NOT NULL REFERENCES assignments(issue_time),
        descriptor_json TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)''',
    'CREATE INDEX samples_time ON samples(issue_time)',
    '''CREATE TABLE sample_assets (
        sample_id TEXT NOT NULL REFERENCES samples(id), asset_id TEXT NOT NULL REFERENCES assets(id),
        kind TEXT NOT NULL CHECK(kind IN ('inputs','targets')),
        PRIMARY KEY(sample_id,asset_id,kind))''',
    '''CREATE TABLE usage_plan (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
        sample_id TEXT NOT NULL REFERENCES samples(id),
        stage TEXT NOT NULL CHECK(stage='base'), pass_number INTEGER NOT NULL CHECK(pass_number>=0),
        created_at TEXT NOT NULL, UNIQUE(sample_id,stage,pass_number))''',
    # No writer is exposed for these two tables in C1. A receipt is not proof
    # that a gradient has been applied. Filling them belongs to C2, not HTTP.
    '''CREATE TABLE checkpoint_generations (
        id TEXT PRIMARY KEY, manifest_hash TEXT NOT NULL, committed_at TEXT NOT NULL)''',
    '''CREATE TABLE training_events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        use_id TEXT UNIQUE NOT NULL REFERENCES usage_plan(id),
        generation_id TEXT NOT NULL REFERENCES checkpoint_generations(id))''',
    '''CREATE TABLE events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
        payload_json TEXT NOT NULL, created_at TEXT NOT NULL)''',
    '''CREATE TRIGGER train_only BEFORE INSERT ON usage_plan
        WHEN (SELECT role FROM assignments JOIN samples USING(issue_time)
              WHERE samples.id=NEW.sample_id) != 'train'
        BEGIN SELECT RAISE(ABORT,'control sample cannot be scheduled for training'); END''',
)
IMMUTABLE = ('campaign', 'ranges', 'assignments', 'assets', 'samples',
             'sample_assets', 'usage_plan', 'checkpoint_generations', 'training_events', 'events')


class CampaignStore:
    """One persistent campaign per explicitly chosen local workspace."""

    def __init__(self, workspace, *, clock=None):
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        workspace = Path(workspace).absolute()
        self.root = workspace / 'continuous'
        self.path = self.root / 'learning.sqlite3'
        self._safe_paths()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._safe_paths()
        # Reserve a private regular file without following an existing link.
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            self._safe_paths()
        else:
            os.close(fd)
        self._initialize()

    def _safe_paths(self):
        for path in (self.path, *self.path.parents,
                     *(Path(str(self.path) + suffix) for suffix in ('-wal', '-shm', '-journal'))):
            if path.is_symlink():
                raise ValueError('Символическая ссылка в хранилище программы запрещена.')
            if path.exists() and path not in self.path.parents:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError('Хранилище должно содержать обычные файлы без жёстких ссылок.')

    def _now(self):
        result = self.clock()
        if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
            raise ValueError('Часы программы должны возвращать время с часовым поясом.')
        return result.astimezone(timezone.utc)

    def _connect(self):
        self._safe_paths()
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA busy_timeout=10000')
        return db

    def _initialize(self):
        with self._connection() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            app_id = db.execute('PRAGMA application_id').fetchone()[0]
            if version not in (0, SCHEMA_VERSION) or app_id not in (0, APPLICATION_ID):
                raise ValueError('Неизвестная версия базы. Автоматическая перезапись запрещена.')
            mode = db.execute('PRAGMA journal_mode=WAL').fetchone()[0]
            if mode.lower() != 'wal':
                raise ValueError('SQLite не включила WAL; выберите локальное хранилище.')
            db.execute('BEGIN IMMEDIATE')
            try:
                # Re-read inside the write transaction: another constructor
                # may have initialized the same empty database in the meantime.
                version = db.execute('PRAGMA user_version').fetchone()[0]
                if version not in (0, SCHEMA_VERSION):
                    raise ValueError('Версия базы изменилась при открытии.')
                if version == 0:
                    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table'").fetchone():
                        raise ValueError('Неизвестное содержимое базы без версии схемы.')
                    for statement in DDL:
                        db.execute(statement)
                    for table in IMMUTABLE:
                        for operation in ('UPDATE', 'DELETE'):
                            db.execute(f'''CREATE TRIGGER immutable_{table}_{operation.lower()}
                                BEFORE {operation} ON {table} BEGIN
                                SELECT RAISE(ABORT,'immutable campaign history'); END''')
                    db.execute(f'PRAGMA application_id={APPLICATION_ID}')
                    db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
                if db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID:
                    raise ValueError('База не принадлежит программе обучения.')
                db.execute('COMMIT')
            except BaseException:
                if db.in_transaction:
                    db.execute('ROLLBACK')
                raise

    @contextmanager
    def _connection(self):
        db = self._connect()
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self, *, write=False):
        with self._connection() as db:
            if (db.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION
                    or db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID):
                raise ValueError('Версия или принадлежность базы изменилась.')
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield db
                db.execute('COMMIT')
            except BaseException:
                if db.in_transaction:
                    db.execute('ROLLBACK')
                raise

    def _campaign(self, db, *, create=False):
        row = db.execute('SELECT * FROM campaign WHERE singleton=1').fetchone()
        if row is None and create:
            contract = default_contract()
            db.execute('INSERT INTO campaign VALUES (1,?,?,?,?)',
                       (uuid.uuid4().hex, canonical(contract), fingerprint(contract), self._now().isoformat()))
            row = db.execute('SELECT * FROM campaign WHERE singleton=1').fetchone()
            self._event(db, 'campaign_created', {'campaign_id': row['id']})
        if row is not None:
            import json
            payload = json.loads(row['contract_json'])
            if payload.get('split', {}).get('version') != SPLIT_VERSION:
                raise ValueError('Версия временного разбиения требует явной миграции.')
            if fingerprint(payload) != row['contract_hash']:
                raise ValueError('Контракт модели повреждён.')
            return dict(id=row['id'], contract=payload, contract_hash=row['contract_hash'],
                        created_at=row['created_at'])
        return None

    def _event(self, db, kind, payload):
        db.execute('INSERT INTO events(kind,payload_json,created_at) VALUES (?,?,?)',
                   (kind, canonical(payload), self._now().isoformat()))

    @staticmethod
    def _receipt(row, campaign_id, *, replayed=False):
        return {'id': row['id'], 'sequence': row['seq'], 'campaign_id': campaign_id,
                'start_date': date.fromordinal(row['start_day']).isoformat(),
                'end_date': date.fromordinal(row['end_day']).isoformat(),
                'requested_days': row['end_day'] - row['start_day'] + 1,
                'new_days_at_registration': row['new_days'],
                'existing_days_at_registration': row['end_day'] - row['start_day'] + 1 - row['new_days'],
                'replayed': replayed, 'created_at': row['created_at'],
                'status': 'registered_waiting_for_executor', 'training_started': False}

    def add_range(self, start_date, end_date, *, request_key=None):
        first, last = check_range(start_date, end_date, self._now().date())
        key = (fingerprint({'start': first, 'end': last}) if request_key is None
               else identifier(request_key, 'ключ повторного запроса'))
        with self.transaction(write=True) as db:
            campaign = self._campaign(db, create=True)
            old = db.execute('SELECT * FROM ranges WHERE request_key=?', (key,)).fetchone()
            if old:
                if (old['start_day'], old['end_day']) != (first, last):
                    raise ValueError('Этот ключ запроса уже использован с другими датами.')
                return self._receipt(old, campaign['id'], replayed=True)
            overlaps = db.execute('SELECT * FROM intervals WHERE start_day<=? AND end_day>=? ORDER BY start_day',
                                  (last + 1, first - 1)).fetchall()
            existing = sum(max(0, min(last, r['end_day']) - max(first, r['start_day']) + 1) for r in overlaps)
            new_days = last - first + 1 - existing
            lower = min([first] + [r['start_day'] for r in overlaps])
            upper = max([last] + [r['end_day'] for r in overlaps])
            for row in overlaps:
                db.execute('DELETE FROM intervals WHERE start_day=?', (row['start_day'],))
            db.execute('INSERT INTO intervals VALUES (?,?)', (lower, upper))
            identity = uuid.uuid4().hex
            db.execute('INSERT INTO ranges(id,request_key,start_day,end_day,new_days,created_at) VALUES (?,?,?,?,?,?)',
                       (identity, key, first, last, new_days, self._now().isoformat()))
            self._event(db, 'range_registered', {'request_id': identity, 'start_date': start_date,
                                               'end_date': end_date, 'new_days': new_days})
            row = db.execute('SELECT * FROM ranges WHERE id=?', (identity,)).fetchone()
            return self._receipt(row, campaign['id'])

    def state(self):
        with self.transaction() as db:
            counts = {name: db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                      for name, table in [('requests', 'ranges'), ('samples', 'samples'),
                                          ('planned_uses', 'usage_plan'), ('committed_uses', 'training_events')]}
            days = db.execute('SELECT coalesce(sum(end_day-start_day+1),0) FROM intervals').fetchone()[0]
            return {'schema': 'continuous-state-1', 'campaign': self._campaign(db),
                    'requested_days': days, **counts,
                    'last_event': db.execute('SELECT coalesce(max(seq),0) FROM events').fetchone()[0],
                    'training_ready': False, 'execution_status': 'awaiting_C2_C5_executor',
                    'model_initialized': False, 'historical_independence_verified': False}

    @staticmethod
    def _pagination(after, limit):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError('Курсор должен быть неотрицательным; размер страницы — от 1 до 200.')

    def ranges(self, *, after=0, limit=100):
        self._pagination(after, limit)
        with self.transaction() as db:
            campaign = self._campaign(db)
            rows = db.execute('SELECT * FROM ranges WHERE seq>? ORDER BY seq LIMIT ?', (after, limit + 1)).fetchall()
            return {'items': [self._receipt(r, campaign['id']) for r in rows[:limit]],
                    'next_after': rows[limit - 1]['seq'] if len(rows) > limit else None}

    def events(self, *, after=0, limit=100):
        import json
        self._pagination(after, limit)
        with self.transaction() as db:
            rows = db.execute('SELECT * FROM events WHERE seq>? ORDER BY seq LIMIT ?', (after, limit + 1)).fetchall()
            items = [dict(sequence=r['seq'], kind=r['kind'], payload=json.loads(r['payload_json']),
                          created_at=r['created_at']) for r in rows[:limit]]
            return {'items': items, 'next_after': rows[limit - 1]['seq'] if len(rows) > limit else None}

    def blocks(self, *, after_date=None, limit=100):
        self._pagination(0, limit)
        after = calendar_date(after_date).toordinal() if after_date else 0
        with self.transaction() as db:
            campaign = self._campaign(db)
            if campaign is None:
                return {'items': [], 'next_after_date': None}
            days = []
            # Intervals, not millions of daily rows. Skip unrequested gaps in SQL.
            for interval in db.execute('SELECT * FROM intervals WHERE end_day>? ORDER BY start_day', (after,)):
                start = max(after + 1, interval['start_day'])
                count = min(interval['end_day'] - start + 1, limit + 1 - len(days))
                days.extend(range(start, start + count))
                if len(days) > limit:
                    break
            contract = campaign['contract']
            items = []
            for day in days[:limit]:
                stamp = date.fromordinal(day).isoformat()
                roles = dict.fromkeys(('train', 'validation', 'test', 'guard'), 0)
                for hour in contract['issue_hours']:
                    _, role = temporal_role(utc_time(f'{stamp}T{hour:02d}:00:00Z'),
                                            contract['history_hours'], contract['horizon_hours'],
                                            contract['split']['seed'])
                    roles[role] += 1
                sample_count = db.execute('SELECT count(*) FROM samples WHERE issue_time>=? AND issue_time<?',
                                          (stamp, date.fromordinal(day + 1).isoformat())).fetchone()[0]
                items.append({'id': fingerprint({'campaign': campaign['id'], 'day': stamp}),
                              'date': stamp, 'issue_roles': roles, 'registered_samples': sample_count,
                              'catalog_status': 'not_checked', 'preparation_status': 'not_started',
                              'training_status': 'not_started'})
            return {'items': items, 'next_after_date': items[-1]['date'] if len(days) > limit else None}

    def register_sample(self, descriptor):
        """Internal preparation API: records provenance, never physical admission."""
        with self.transaction(write=True) as db:
            campaign = self._campaign(db)
            if campaign is None:
                raise ValueError('Сначала зарегистрируйте диапазон программы.')
            contract = campaign['contract']
            payload = sample_identity(descriptor, contract)
            issue = utc_time(payload['issue_time'])
            day = issue.date().toordinal()
            if not db.execute('SELECT 1 FROM intervals WHERE start_day<=? AND end_day>=?', (day, day)).fetchone():
                raise ValueError('Срок выпуска находится вне запрошенных диапазонов.')
            nominal, role = temporal_role(issue, contract['history_hours'], contract['horizon_hours'],
                                          contract['split']['seed'])
            old = db.execute('SELECT * FROM assignments WHERE issue_time=?', (payload['issue_time'],)).fetchone()
            if old is not None and (old['nominal_role'], old['role']) != (nominal, role):
                raise ValueError('Постоянная роль выпуска не совпадает с политикой программы.')
            db.execute('INSERT OR IGNORE INTO assignments VALUES (?,?,?)', (payload['issue_time'], nominal, role))
            identity = fingerprint(payload)
            existed = db.execute('SELECT id FROM samples WHERE id=?', (identity,)).fetchone() is not None
            if not existed:
                db.execute('INSERT INTO samples(id,issue_time,descriptor_json,created_at) VALUES (?,?,?,?)',
                           (identity, payload['issue_time'], canonical(payload), self._now().isoformat()))
                for kind in ('inputs', 'targets'):
                    for asset in payload[kind]:
                        asset_id = fingerprint(asset)
                        db.execute('INSERT OR IGNORE INTO assets VALUES (?,?)', (asset_id, canonical(asset)))
                        db.execute('INSERT INTO sample_assets VALUES (?,?,?)', (identity, asset_id, kind))
                self._event(db, 'sample_registered', {'sample_id': identity, 'role': role})
            return {'id': identity, 'role': role, 'replayed': existed,
                    'physically_admitted': False, 'training_committed': False}

    def plan_use(self, sample_id, *, pass_number=0):
        """Idempotent intent only. No interface can label this intent as trained."""
        if not isinstance(sample_id, str) or len(sample_id) != 64:
            raise ValueError('Неверный идентификатор примера.')
        if type(pass_number) is not int or not 0 <= pass_number <= 10000:
            raise ValueError('Неверный номер разрешённого прохода.')
        with self.transaction(write=True) as db:
            campaign = self._campaign(db)
            sample = db.execute('SELECT samples.id,role FROM samples JOIN assignments USING(issue_time) WHERE samples.id=?',
                                (sample_id,)).fetchone()
            if sample is None:
                raise ValueError('Пример не зарегистрирован.')
            if sample['role'] != 'train':
                raise ValueError('Контрольные и защитные примеры не допускаются к обучению.')
            identity = fingerprint({'campaign': campaign['id'], 'stage': 'base',
                                    'sample': sample_id, 'pass': pass_number})
            old = db.execute('SELECT id FROM usage_plan WHERE id=?', (identity,)).fetchone()
            if not old:
                db.execute('INSERT INTO usage_plan(id,sample_id,stage,pass_number,created_at) VALUES (?,?,?,?,?)',
                           (identity, sample_id, 'base', pass_number, self._now().isoformat()))
                self._event(db, 'usage_planned', {'use_id': identity, 'sample_id': sample_id, 'pass_number': pass_number})
            return {'id': identity, 'sample_id': sample_id, 'pass_number': pass_number,
                    'status': 'planned_not_trained', 'replayed': old is not None}

    def samples(self, *, after=0, limit=100):
        self._pagination(after, limit)
        with self.transaction() as db:
            rows = db.execute('''SELECT samples.seq,samples.id,issue_time,role,
                (SELECT count(*) FROM usage_plan WHERE sample_id=samples.id) AS planned_uses,
                (SELECT count(*) FROM training_events JOIN usage_plan ON use_id=usage_plan.id
                 WHERE sample_id=samples.id) AS committed_uses
                FROM samples JOIN assignments USING(issue_time)
                WHERE samples.seq>? ORDER BY samples.seq LIMIT ?''', (after, limit + 1)).fetchall()
            return {'items': [dict(r) for r in rows[:limit]],
                    'next_after': rows[limit - 1]['seq'] if len(rows) > limit else None}
