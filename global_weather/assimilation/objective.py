"""Observation-space objective: robust departures normalised by observation error.

For every target i with prediction H_i(x), measurement y_i and error sigma_i,
the departure r_i = (H_i(x) - y_i) / sigma_i is dimensionless, so variables
of different units are combined only after division by their own error.

* One member: Huber norm rho(r) with delta = 3, the robust VarQC-style norm
  used in variational assimilation to limit the influence of gross errors.
* Several members: the fair CRPS of the ensemble departures,
  mean_m |r_m| - sum_{m<n} |r_m - r_n| / (M (M - 1)), an unbiased proper score.
Weights are the superobbing density weights of the plan (one unit per
occupied latent cell, variable, layer and lead), the analysis weight and quality.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .operator import Plan
from .tokens import VARIABLES


def huber(residual: Tensor, delta: float) -> Tensor:
    absolute = residual.abs()
    clipped = absolute.clamp_max(delta)
    return 0.5 * clipped ** 2 + delta * (absolute - clipped)


def fair_crps(residuals: Tensor) -> Tensor:
    """[M, P] ensemble departures -> [P] fair CRPS (M >= 2)."""
    members = residuals.shape[0]
    skill = residuals.abs().mean(0)
    spread = (residuals[:, None, :] - residuals[None, :, :]).abs().sum((0, 1)) / (2 * members * (members - 1))
    return skill - spread


def observation_objective(predictions: Tensor, plan: Plan, *, huber_delta: float = 3.0) -> tuple[Tensor, dict]:
    """predictions [M, P] in physical units -> (scalar loss, report)."""
    if predictions.dim() != 2 or predictions.shape[1] != plan.count:
        raise ValueError('Predictions must have shape [members, targets].')
    if plan.count == 0:
        raise ValueError('No supported targets in this window.')
    if not bool(torch.isfinite(predictions).all()):
        raise FloatingPointError('Nonfinite observation-space prediction.')
    residual = (predictions - plan.value) / plan.sigma
    score = huber(residual[0], huber_delta) if predictions.shape[0] == 1 else fair_crps(residual)
    weight = plan.weight.to(score.dtype)
    loss = (score * weight).sum() / weight.sum()
    report = {'targets': plan.count, 'rejected': dict(plan.rejected), 'members': int(predictions.shape[0])}
    with torch.no_grad():
        mean = predictions.mean(0)
        error = (mean - plan.value).double()
        per_variable = {}
        for code in torch.unique(plan.variable).tolist():
            for phase, selector in (('analysis', plan.is_analysis), ('forecast', ~plan.is_analysis)):
                rows = (plan.variable == code) & selector
                if bool(rows.any()):
                    per_variable[f'{VARIABLES[code]}/{phase}'] = {
                        'count': int(rows.sum()), 'rmse': float(error[rows].square().mean().sqrt()),
                        'bias': float(error[rows].mean()),
                        'normalised_loss': float((score[rows] * weight[rows]).sum() / weight[rows].sum())}
        report['variables'] = per_variable
        report['loss'] = float(loss)
    return loss, report


def persistence_skill(plan: Plan, predictions: Tensor, baseline: np.ndarray) -> dict:
    """RMSE of the model and of a baseline on their common support, per variable."""
    out = {}
    finite = np.isfinite(baseline)
    mean = predictions.mean(0).detach().double().cpu().numpy()
    value = plan.value.double().cpu().numpy()
    variable = plan.variable.cpu().numpy()
    forecast = ~plan.is_analysis.cpu().numpy()
    for code in np.unique(variable):
        rows = (variable == code) & finite & forecast
        if rows.sum() < 2:
            continue
        model_rmse = float(np.sqrt(np.mean((mean[rows] - value[rows]) ** 2)))
        base_rmse = float(np.sqrt(np.mean((baseline[rows] - value[rows]) ** 2)))
        out[VARIABLES[code]] = {'count': int(rows.sum()), 'model_rmse': model_rmse, 'baseline_rmse': base_rmse,
                                'rmse_skill': None if base_rmse == 0 else 1 - model_rmse / base_rmse}
    return out
