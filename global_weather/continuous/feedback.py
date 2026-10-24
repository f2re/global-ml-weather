"""Guarded factual feedback for the coupled weather model.

Fresh facts never mutate published weights directly.  They enter the immutable
CampaignStore ledger, are mixed with historical replay, and are applied through
StepTrainer only when a candidate improves a fixed guard evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import uuid

import torch

from .checkpointing import StepTrainer
from .contracts import canonical, default_contract, fingerprint, SPLIT_VERSION
from .store import CampaignStore
from ..coupled import ARCHITECTURE as COUPLED_ARCHITECTURE


def coupled_campaign_contract(*, mesh_level=8, analysis_hidden=32, forecast_hidden=4,
                              latent_slots=8, horizon_hours=3,
                              issue_hours=(0, 6, 12, 18)):
    """Return an immutable identity for a coupled campaign.

    +3 h is the default because it matches the currently qualified Mesh-8
    rollout. Longer horizons must be explicit and independently qualified.
    """
    if not 0 <= mesh_level <= 8 or min(analysis_hidden, forecast_hidden, latent_slots) < 1:
        raise ValueError("Invalid coupled architecture dimensions.")
    if type(horizon_hours) is not int or not 3 <= horizon_hours <= 72 or horizon_hours % 3:
        raise ValueError("Coupled horizon must be 3..72 hours in 3-hour steps.")
    contract = default_contract()
    contract.update({
        "architecture": COUPLED_ARCHITECTURE,
        "mesh_level": mesh_level,
        "hidden": forecast_hidden,
        "latent_slots": latent_slots,
        "horizon_hours": horizon_hours,
        "issue_hours": list(issue_hours),
        "coupled": {
            "analysis_hidden": analysis_hidden,
            "forecast_hidden": forecast_hidden,
            "bridge": "latent-to-pressure-cross-attention-v1",
            "feedback": "guarded-online-feedback-v1",
        },
    })
    return json.loads(canonical(contract))


class CoupledCampaignStore(CampaignStore):
    """CampaignStore whose workspace is pinned to one coupled contract."""

    def __init__(self, workspace, contract, *, clock=None):
        if not isinstance(contract, dict) or contract.get("architecture") != COUPLED_ARCHITECTURE:
            raise ValueError("Need an explicit coupled campaign contract.")
        if contract.get("split", {}).get("version") != SPLIT_VERSION:
            raise ValueError("Unsupported split version.")
        self.coupled_contract = json.loads(canonical(contract))
        super().__init__(workspace, clock=clock)

    def _campaign(self, db, *, create=False):
        current = super()._campaign(db, create=False)
        wanted = fingerprint(self.coupled_contract)
        if current is not None:
            if current["contract_hash"] != wanted:
                raise ValueError("Workspace belongs to another model contract.")
            return current
        if not create:
            return None
        identity = uuid.uuid4().hex
        db.execute('INSERT INTO campaign VALUES (1,?,?,?,?)',
                   (identity, canonical(self.coupled_contract), wanted,
                    self._now().isoformat()))
        self._event(db, "campaign_created", {
            "campaign_id": identity, "architecture": COUPLED_ARCHITECTURE})
        return super()._campaign(db, create=False)


def _aware(value, label):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be timezone-aware ISO 8601.") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone.")
    return result.astimezone(timezone.utc)


def _finite_metrics(metrics):
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Metrics must be a nonempty dictionary.")
    result = {}
    for key, value in metrics.items():
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError("Metrics must contain finite scalars.")
        result[str(key)] = float(value)
    return result


def record_fact_admission(store, *, sample_id, target_sha256, available_at, qc):
    """Mark an already registered target as physically/QC admitted."""
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
    payload = {"sample_id": sample_id, "target_sha256": target_sha256,
               "available_at": available.isoformat(), "qc": safe_qc}
    with store.transaction(write=True) as db:
        if db.execute('SELECT 1 FROM samples WHERE id=?', (sample_id,)).fetchone() is None:
            raise ValueError("Fact refers to an unregistered sample.")
        store._event(db, "fact_admitted", payload)
    return payload


def record_forecast_verification(store, *, forecast_generation, issue_time, valid_time,
                                 lead_hours, metrics, forecast_sha256=None,
                                 target_sha256=None):
    """Append an immutable error receipt after factual targets arrive."""
    if not isinstance(forecast_generation, str) or not re.fullmatch(r"[a-f0-9]{32}", forecast_generation):
        raise ValueError("Invalid forecast checkpoint generation.")
    if type(lead_hours) is not int or not 0 < lead_hours <= 72 or lead_hours % 3:
        raise ValueError("Lead must be 3..72 hours in 3-hour steps.")
    for value in (forecast_sha256, target_sha256):
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)):
            raise ValueError("Invalid forecast/target sha256.")
    issue, valid = _aware(issue_time, "issue_time"), _aware(valid_time, "valid_time")
    if int((valid - issue).total_seconds()) != lead_hours * 3600:
        raise ValueError("valid_time does not match issue+lead.")
    payload = {
        "forecast_generation": forecast_generation, "issue_time": issue.isoformat(),
        "valid_time": valid.isoformat(), "lead_hours": lead_hours,
        "metrics": _finite_metrics(metrics), "forecast_sha256": forecast_sha256,
        "target_sha256": target_sha256,
    }
    with store.transaction(write=True) as db:
        if db.execute('SELECT 1 FROM checkpoint_generations WHERE id=?',
                      (forecast_generation,)).fetchone() is None:
            raise ValueError("Forecast generation is not registered in this campaign.")
        store._event(db, "forecast_verified", payload)
    return payload


@dataclass(frozen=True)
class OnlineLearningPolicy:
    min_new_samples: int = 32
    max_new_samples: int = 128
    replay_per_new: int = 3
    replay_pool_size: int = 4096
    gradient_clip: float = 0.5
    primary_metric: str = "rmse"
    min_relative_improvement: float = 0.001
    guardrail_metrics: tuple[str, ...] = ("rmse", "mae", "abs_bias")
    max_relative_guardrail_degradation: float = 0.005
    absolute_guardrail_tolerance: float = 1e-8

    def __post_init__(self):
        if not 1 <= self.min_new_samples <= self.max_new_samples <= 4096:
            raise ValueError("Invalid online batch bounds.")
        if not 0 <= self.replay_per_new <= 32 or self.replay_pool_size < self.max_new_samples:
            raise ValueError("Invalid replay policy.")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("Invalid gradient clip.")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("Invalid improvement threshold.")
        if not 0 <= self.max_relative_guardrail_degradation < 1:
            raise ValueError("Invalid guardrail tolerance.")


def masked_error_metrics(prediction, target, mask, scale=None):
    """RMSE/MAE/bias over factual, admitted values only."""
    if prediction.shape != target.shape or prediction.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Prediction, target and mask shapes differ.")
    if not bool(mask.any()):
        raise ValueError("At least one factual target is required.")
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("Verified values must be finite.")
    error = prediction - target
    if scale is not None:
        if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
            raise ValueError("Scale must be finite and positive.")
        error = error / scale
    values = error[mask]
    bias = values.mean()
    return {"count": int(values.numel()),
            "rmse": float(values.square().mean().sqrt().detach().cpu()),
            "mae": float(values.abs().mean().detach().cpu()),
            "bias": float(bias.detach().cpu()),
            "abs_bias": float(bias.abs().detach().cpu())}


def candidate_decision(before, after, policy):
    """Accept improvement only when every lower-is-better guardrail survives."""
    before, after = _finite_metrics(before), _finite_metrics(after)
    metric = policy.primary_metric
    if metric not in before or metric not in after:
        raise ValueError("Primary metric missing from evaluation.")
    base = before[metric]
    limit = (policy.absolute_guardrail_tolerance if base == 0
             else base * (1 - policy.min_relative_improvement))
    failures = [] if after[metric] <= limit else [f"{metric}: {after[metric]:.6g} > {limit:.6g}"]
    for name in policy.guardrail_metrics:
        if name not in before or name not in after:
            raise ValueError(f"Guardrail metric missing: {name}")
        allowed = (before[name] * (1 + policy.max_relative_guardrail_degradation)
                   + policy.absolute_guardrail_tolerance)
        if after[name] > allowed:
            failures.append(f"{name}: {after[name]:.6g} > {allowed:.6g}")
    return {"accepted": not failures,
            "reason": "accepted" if not failures else "; ".join(failures),
            "before": before, "after": after}


class ReplayPlanner:
    """Mix newly admitted train facts with month-stratified historical replay."""

    def __init__(self, store, policy):
        self.store, self.policy = store, policy

    def _admitted(self):
        with self.store.transaction() as db:
            rows = db.execute("SELECT payload_json FROM events WHERE kind='fact_admitted' ORDER BY seq").fetchall()
        result = set()
        for row in rows:
            try:
                sample = json.loads(row[0]).get("sample_id")
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(sample, str) and re.fullmatch(r"[a-f0-9]{64}", sample):
                result.add(sample)
        return result

    def _fresh(self):
        admitted = self._admitted()
        if not admitted:
            return []
        with self.store.transaction() as db:
            rows = db.execute('''SELECT u.id AS use_id,u.sample_id,s.issue_time
                FROM usage_plan u JOIN samples s ON s.id=u.sample_id
                JOIN assignments a USING(issue_time)
                LEFT JOIN training_events e ON e.use_id=u.id
                WHERE a.role='train' AND u.pass_number=0 AND e.use_id IS NULL
                ORDER BY s.issue_time,u.seq LIMIT ?''',
                (self.policy.max_new_samples * 8,)).fetchall()
        return [dict(r) for r in rows if r['sample_id'] in admitted][:self.policy.max_new_samples]

    def _replay_pool(self, exclude):
        with self.store.transaction() as db:
            rows = db.execute('''SELECT s.id AS sample_id,s.issue_time,max(u.pass_number) AS max_pass
                FROM samples s JOIN assignments a USING(issue_time)
                JOIN usage_plan u ON u.sample_id=s.id
                JOIN training_events e ON e.use_id=u.id WHERE a.role='train'
                GROUP BY s.id,s.issue_time ORDER BY s.issue_time DESC LIMIT ?''',
                (self.policy.replay_pool_size,)).fetchall()
        return [dict(r) for r in rows if r['sample_id'] not in exclude]

    @staticmethod
    def _pick(rows, count, salt):
        buckets = {m: [] for m in range(1, 13)}
        for row in rows:
            month = int(row["issue_time"][5:7])
            rank = hashlib.sha256((salt + row["sample_id"]).encode()).hexdigest()
            buckets[month].append((rank, row))
        for bucket in buckets.values():
            bucket.sort(key=lambda x: x[0])
        selected = []
        while len(selected) < count:
            moved = False
            for month in range(1, 13):
                if buckets[month]:
                    selected.append(buckets[month].pop(0)[1]); moved = True
                    if len(selected) == count:
                        break
            if not moved:
                break
        return selected

    def plan(self):
        fresh = self._fresh()
        if len(fresh) < self.policy.min_new_samples:
            return {"status": "waiting_for_facts", "new": len(fresh), "use_ids": []}
        exclude = {r["sample_id"] for r in fresh}
        with self.store.transaction() as db:
            head = db.execute('SELECT generation_id FROM checkpoint_head WHERE singleton=1').fetchone()
        salt = "initial" if head is None else head[0]
        count = min(len(fresh) * self.policy.replay_per_new, 4096 - len(fresh))
        replay = self._pick(self._replay_pool(exclude), count, salt)
        replay_ids = [self.store.plan_use(r["sample_id"],
                      pass_number=int(r["max_pass"]) + 1)["id"] for r in replay]
        return {"status": "ready", "new": len(fresh), "replay": len(replay_ids),
                "use_ids": [r["use_id"] for r in fresh] + replay_ids}


class CandidateRejected(RuntimeError):
    def __init__(self, report):
        super().__init__(report["reason"]); self.report = report


class GuardedOnlineLearner:
    """Run exactly one candidate step; publish it only after guard evaluation."""

    def __init__(self, store, model, optimizer, *, identity, batch_loss, evaluate,
                 policy=None, scheduler=None, scaler=None):
        self.store, self.model, self.optimizer = store, model, optimizer
        self.identity, self.batch_loss, self.evaluate = identity, batch_loss, evaluate
        self.policy = policy or OnlineLearningPolicy()
        self.scheduler, self.scaler = scheduler, scaler

    def _trainer(self):
        return StepTrainer(self.store, self.model, self.optimizer, identity=self.identity,
                           scheduler=self.scheduler, scaler=self.scaler)

    def _restore(self):
        with self._trainer():
            pass

    def attempt(self, use_ids, *, cursor):
        if not use_ids:
            return {"status": "waiting_for_facts"}
        rejected = None
        try:
            with self._trainer() as trainer:
                self.model.eval(); before = _finite_metrics(self.evaluate(self.model))

                def operation():
                    self.model.train(); self.optimizer.zero_grad(set_to_none=True)
                    built = self.batch_loss(self.model, use_ids)
                    loss, batch = built if isinstance(built, tuple) else (built, {})
                    if not isinstance(loss, torch.Tensor) or loss.ndim or not torch.isfinite(loss):
                        raise ValueError("Online loss must be a finite scalar tensor.")
                    loss.backward()
                    params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                    if not params:
                        raise ValueError("No trainable gradient in online candidate.")
                    norm = torch.nn.utils.clip_grad_norm_(params, self.policy.gradient_clip)
                    if not torch.isfinite(torch.as_tensor(norm)):
                        raise FloatingPointError("Nonfinite online gradient norm.")
                    self.optimizer.step(); self.model.eval()
                    decision = candidate_decision(before, self.evaluate(self.model), self.policy)
                    report = {**decision, "loss": float(loss.detach().cpu()),
                              "gradient_norm": float(torch.as_tensor(norm).detach().cpu()),
                              "batch": _finite_metrics(batch) if batch else {}}
                    if not decision["accepted"]:
                        raise CandidateRejected(report)
                    return report

                result = trainer.step(use_ids, operation, cursor=cursor)
            with self.store.transaction(write=True) as db:
                self.store._event(db, "online_candidate_committed", {
                    "generation_id": result.get("generation_id"), "use_count": len(use_ids),
                    "metrics": result.get("metrics", {})})
            return {"status": "accepted", **result}
        except CandidateRejected as exc:
            rejected = exc.report
        except BaseException:
            self._restore(); raise
        self._restore()
        with self.store.transaction(write=True) as db:
            self.store._event(db, "online_candidate_rejected", {
                "use_count": len(use_ids), "report": rejected})
        return {"status": "rejected", "report": rejected}


class AutomaticFeedbackCycle:
    """One polling iteration after preparation has registered/admitted facts."""

    def __init__(self, planner, learner):
        self.planner, self.learner = planner, learner

    def run_once(self, cycle_id):
        if not isinstance(cycle_id, str) or not cycle_id or len(cycle_id) > 128:
            raise ValueError("cycle_id must be a short nonempty string.")
        planned = self.planner.plan()
        if planned["status"] != "ready":
            return planned
        return self.learner.attempt(planned["use_ids"], cursor={
            "feedback_cycle": cycle_id, "new": planned["new"],
            "replay": planned["replay"]})
