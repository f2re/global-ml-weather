"""Additive C2 migration; immutable C1 history is never rewritten."""

DDL = (
    '''CREATE TABLE trainer_contract (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
        identity_json TEXT NOT NULL, identity_hash TEXT NOT NULL)''',
    '''CREATE TABLE checkpoint_details (
        generation_id TEXT PRIMARY KEY REFERENCES checkpoint_generations(id),
        step_number INTEGER UNIQUE NOT NULL CHECK(step_number>=0),
        parent_id TEXT REFERENCES checkpoint_generations(id), manifest_json TEXT NOT NULL)''',
    '''CREATE TABLE checkpoint_head (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
        generation_id TEXT NOT NULL REFERENCES checkpoint_generations(id))''',
    '''CREATE TABLE trainer_owner (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1), epoch INTEGER NOT NULL,
        token TEXT NOT NULL)''',
    '''CREATE TABLE checkpoint_pins (
        generation_id TEXT PRIMARY KEY REFERENCES checkpoint_generations(id))''',
    '''CREATE TRIGGER committed_train_only BEFORE INSERT ON training_events
        WHEN (SELECT role FROM assignments JOIN samples USING(issue_time)
              JOIN usage_plan ON usage_plan.sample_id=samples.id
              WHERE usage_plan.id=NEW.use_id) != 'train'
        BEGIN SELECT RAISE(ABORT,'control sample cannot be committed'); END''',
)


def migrate(db):
    """Caller owns the existing initializer lock and write transaction."""
    # C1 exposed no writer. Refuse unexplained ledger entries rather than
    # manufacturing a checkpoint or silently declaring old weights usable.
    if db.execute('SELECT 1 FROM checkpoint_generations LIMIT 1').fetchone():
        raise ValueError('В базе C1 есть неизвестные контрольные точки; требуется проверка.')
    for statement in DDL:
        db.execute(statement)
    for table in ('trainer_contract', 'checkpoint_details', 'checkpoint_pins'):
        for operation in ('UPDATE', 'DELETE'):
            db.execute(f'''CREATE TRIGGER immutable_{table}_{operation.lower()}
                BEFORE {operation} ON {table} BEGIN
                SELECT RAISE(ABORT,'immutable training history'); END''')
