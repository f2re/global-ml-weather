"""Provenance-hardened admission and selection for coupled online feedback.

This layer sits above feedback.py.  It distinguishes a planned use from a
registered target, a physically admitted fact, and a forecast that has actually
been scored against that same fact.  Only samples with both immutable receipts
are eligible for fresh online optimization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re

from .feedback import ReplayPlanner, _finite_metrics


def _aware(value, label):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be timezone-aware ISO 8601.") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone.")
    return result.astimezone(timezone.utc)


def admit_registered_fact(store, *, sample_id, target_sha256, available_at, qc):
    if not isinstance(sample_id, str) or not re.fullmatch(r"[a-f0-9]{64}", sample_id):
        raise ValueError("Invalid sample id.")
    if not isinstance(target_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", target_sha256):
        raise ValueError("Invalid target sha256.")
    available = _aware(available_at, "available_at")
    if not isinstance(qc, dict) or not qc:
        raise ValueError("Physical/QC receipt is required.")
    safe_qc = {}
    for key, value in qc.items():
        if type(value) in (bool, str, int) or (type(value) is float and math.isfinite(value)):
            safe_qc[str(key)] = value
        else:
            raise ValueError("Unsupported QC value.")
    with store.transaction(write=True) as db:
        sample = db.execute('SELECT issue_time FROM samples WHERE id=?', (sample_id,)).fetchone()
        if sample is None:
            raise ValueError("Fact refers to an unregistered sample.")
        rows = db.execute("""SELECT a.descriptor_json FROM sample_assets sa
            JOIN assets a ON a.id=sa.asset_id
            WHERE sa.sample_id=? AND sa.kind='targets'""", (sample_id,)).fetchall()
        registered = {json.loads(row[0]).get("sha256") for row in rows}
        if target_sha256 not in registered:
            raise ValueError("Fact sha256 is not a registered target version for this sample.")
        payload = {"sample_id": sample_id, "target_sha256": target_sha256,
                   "available_at": available.isoformat(), "qc": safe_qc}
        store._event(db, "fact_admitted_verified", payload)
    return payload


def verify_sample_forecast(store, *, sample_id, target_sha256, forecast_generation,
                           issue_time, valid_time, lead_hours, metrics,
                           forecast_sha256=None):
    if not isinstance(sample_id, str) or not re.fullmatch(r"[a-f0-9]{64}", sample_id):
        raise ValueError("Invalid sample id.")
    if not isinstance(target_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", target_sha256):
        raise ValueError("Invalid target sha256.")
    if not isinstance(forecast_generation, str) or not re.fullmatch(r"[a-f0-9]{32}", forecast_generation):
        raise ValueError("Invalid forecast generation.")
    if forecast_sha256 is not None and (not isinstance(forecast_sha256, str)
                                        or not re.fullmatch(r"[a-f0-9]{64}", forecast_sha256)):
        raise ValueError("Invalid forecast sha256.")
    if type(lead_hours) is not int or not 0 < lead_hours <= 72 or lead_hours % 3:
        raise ValueError("Lead must be 3..72 hours in 3-hour steps.")
    issue, valid = _aware(issue_time, "issue_time"), _aware(valid_time, "valid_time")
    if int((valid - issue).total_seconds()) != lead_hours * 3600:
        raise ValueError("valid_time does not match issue+lead.")
    with store.transaction(write=True) as db:
        if db.execute('SELECT 1 FROM checkpoint_generations WHERE id=?',
                      (forecast_generation,)).fetchone() is None:
            raise ValueError("Unknown forecast generation.")
        sample = db.execute('SELECT issue_time FROM samples WHERE id=?', (sample_id,)).fetchone()
        if sample is None or _aware(sample[0], "sample issue time") != issue:
            raise ValueError("Sample does not match the forecast issue.")
        admitted = False
        for row in db.execute("SELECT payload_json FROM events WHERE kind='fact_admitted_verified'"):
            receipt = json.loads(row[0])
            if receipt.get("sample_id") == sample_id and receipt.get("target_sha256") == target_sha256:
                admitted = True
                break
        if not admitted:
            raise ValueError("Forecast cannot be scored before matching factual admission.")
        payload = {"sample_id": sample_id, "target_sha256": target_sha256,
                   "forecast_generation": forecast_generation,
                   "forecast_sha256": forecast_sha256,
                   "issue_time": issue.isoformat(), "valid_time": valid.isoformat(),
                   "lead_hours": lead_hours, "metrics": _finite_metrics(metrics)}
        store._event(db, "forecast_verified_sample", payload)
    return payload


def eligible_sample_ids(events):
    """Pure helper used by tests and the planner."""
    admitted, verified = set(), set()
    for kind, payload in events:
        if kind not in ("fact_admitted_verified", "forecast_verified_sample"):
            continue
        sample = payload.get("sample_id") if isinstance(payload, dict) else None
        if not isinstance(sample, str) or not re.fullmatch(r"[a-f0-9]{64}", sample):
            continue
        (admitted if kind == "fact_admitted_verified" else verified).add(sample)
    return admitted & verified


class VerifiedReplayPlanner(ReplayPlanner):
    """Fresh planner requiring both physical admission and forecast scoring."""

    def _eligible(self):
        with self.store.transaction() as db:
            rows = db.execute("""SELECT kind,payload_json FROM events
                WHERE kind IN ('fact_admitted_verified','forecast_verified_sample')
                ORDER BY seq""").fetchall()
        events = []
        for row in rows:
            try:
                events.append((row[0], json.loads(row[1])))
            except (TypeError, json.JSONDecodeError):
                continue
        return eligible_sample_ids(events)

    def _fresh(self):
        eligible = self._eligible()
        if not eligible:
            return []
        selected = []
        with self.store.transaction() as db:
            rows = db.execute("""SELECT u.id AS use_id,u.sample_id,s.issue_time
                FROM usage_plan u JOIN samples s ON s.id=u.sample_id
                JOIN assignments a USING(issue_time)
                LEFT JOIN training_events e ON e.use_id=u.id
                WHERE a.role='train' AND u.pass_number=0 AND e.use_id IS NULL
                ORDER BY s.issue_time,u.seq""")
            for row in rows:
                if row['sample_id'] in eligible:
                    selected.append(dict(row))
                    if len(selected) == self.policy.max_new_samples:
                        break
        return selected
