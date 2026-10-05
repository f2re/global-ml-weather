"""Date-only C1 API. Registers work; does not claim that training has started."""
from __future__ import annotations

from fastapi import Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
import sqlite3

from ..continuous.store import CampaignStore


class RangeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    start_date: str = Field(min_length=10, max_length=10)
    end_date: str = Field(min_length=10, max_length=10)


def register(app, workspace):
    store = CampaignStore(workspace)
    app.state.continuous = store

    def call(method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except sqlite3.OperationalError as exc:
            raise HTTPException(503, 'Журнал временно недоступен; повторите запрос с прежним ключом.') from exc
        except sqlite3.DatabaseError as exc:
            raise HTTPException(503, 'Ошибка целостности журнала; данные не перезаписаны.') from exc

    @app.post('/api/learning/ranges')
    def add_range(body: RangeRequest, idempotency_key: str | None = Header(default=None, max_length=128)):
        return call(store.add_range, body.start_date, body.end_date, request_key=idempotency_key)

    @app.get('/api/learning/state')
    def state():
        return call(store.state)

    @app.get('/api/learning/ranges')
    def ranges(after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
        return call(store.ranges, after=after, limit=limit)

    @app.get('/api/learning/blocks')
    def blocks(after_date: str | None = Query(None, max_length=10), limit: int = Query(100, ge=1, le=200)):
        return call(store.blocks, after_date=after_date, limit=limit)

    @app.get('/api/learning/events')
    def events(after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
        return call(store.events, after=after, limit=limit)

    @app.get('/api/learning/samples')
    def samples(after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
        return call(store.samples, after=after, limit=limit)
