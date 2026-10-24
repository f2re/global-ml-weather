"""Missing-target-safe, area-weighted objectives; no target is never 'zero rain'."""
import torch
import torch.nn as nn
from torch.nn import functional as F
from .vertical import (PROFILE_SCALES, SURFACE_SCALES, PROFILE_VARIABLES,
                       PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS)


def masked_huber(prediction,target,mask,weights=None,delta=1.):
    if prediction.shape != target.shape or mask.shape != target.shape or delta <= 0:
        raise ValueError('Matching shapes and positive delta are required.')
    valid = mask.bool() & torch.isfinite(target)
    if (valid & ~torch.isfinite(prediction)).any():
        raise ValueError('Nonfinite prediction at a valid target.')
    weight = torch.ones_like(target) if weights is None else torch.broadcast_to(weights,target.shape)
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('Weights must be finite and nonnegative.')
    p = torch.where(valid,prediction,torch.zeros_like(prediction))
    t = torch.where(valid,target,torch.zeros_like(target))
    w = torch.where(valid,weight,torch.zeros_like(weight))
    loss = (F.huber_loss(p,t,reduction='none',delta=delta)*w).sum()/w.sum().clamp_min(1e-12)
    return loss,int((valid & (w > 0)).sum())


def forecast_loss(frame,profile_target,profile_mask,surface_target,surface_mask,area,
                  *,normalization=None,pressure_pa=None,step_hours=3):
    """Equal per-variable tasks; terrain mask must come from TARGET pressure.

    Never use a predicted below-ground mask to hide errors during training.
    Pressure-level losses are not dp mass-integral losses. Area is normalized
    inside each task, then nonempty tasks are averaged.
    """
    tasks = []
    for k,scale in enumerate(PROFILE_SCALES):
        if normalization is not None:
            if pressure_pa is None: raise ValueError('Pressure coordinates required for z-scaled loss.')
            _,std=normalization.get(PROFILE_VARIABLES[k],PROFILE_UNITS[k]).at(pressure_pa.detach().cpu().numpy())
            scale=torch.as_tensor(std,device=frame.profiles.device,dtype=frame.profiles.dtype)
        loss,count = masked_huber(frame.profiles[...,k]/scale,profile_target[...,k]/scale,
                                   profile_mask[...,k],area[:,None])
        if count: tasks.append(loss)
    for k,scale in enumerate(SURFACE_SCALES):
        if normalization is not None:
            interval=step_hours if SURFACE_VARIABLES[k]=='precipitation_step' else None
            _,scale=normalization.get(SURFACE_VARIABLES[k],SURFACE_UNITS[k]).at(interval_hours=interval)
        loss,count = masked_huber(frame.surface[...,k]/scale,surface_target[...,k]/scale,
                                   surface_mask[...,k],area)
        if count: tasks.append(loss)
    if not tasks:
        raise ValueError('No valid supervised targets for this forecast frame.')
    return torch.stack(tasks).mean()


class SatelliteAssimilationLoss(nn.Module):
    """
    Вычисляет взвешенный функционал невязки в пространстве наблюдений с:
    1) автоматическим контролем грубых ошибок фона (Background Quality Control, |O - B| > threshold * sigma_obs);
    2) робастной функцией Хьюбера (Huber loss) для подавления негауссовых хвостов;
    3) тихоновской регуляризацией дрейфа коэффициентов VarBC.
    """

    def __init__(
        self,
        huber_delta: float = 1.5,
        gross_error_sigma: float = 3.5,
        varbc_reg_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.gross_error_sigma = gross_error_sigma
        self.varbc_reg_weight = varbc_reg_weight

    def compute_robust_channel_loss(
        self,
        normalized_innovation: torch.Tensor,  # [N_obs, C] уже деленная на sigma_obs(C_sym)
        admissible_mask: torch.Tensor,        # [N_obs, C]
    ) -> torch.Tensor:
        with torch.no_grad():
            gross_qc_mask = (normalized_innovation.abs() <= self.gross_error_sigma).float()
            total_mask = admissible_mask * gross_qc_mask

        elementwise_huber = F.huber_loss(
            normalized_innovation * total_mask,
            torch.zeros_like(normalized_innovation),
            reduction="none",
            delta=self.huber_delta,
        )
        denom = total_mask.sum().clamp(min=1.0)
        return (elementwise_huber * total_mask).sum() / denom

    def forward(
        self,
        mtvza_out: dict[str, torch.Tensor],
        ikfs2_out: dict[str, torch.Tensor],
        amv_uv_pred: torch.Tensor,            # [N_amv, 2]
        amv_uv_obs: torch.Tensor,             # [N_amv, 2]
        amv_sigma: torch.Tensor,              # [N_amv, 1]
        amv_qi: torch.Tensor,                 # [N_amv, 1] Quality Indicator in [0, 1]
        varbc_params: list[torch.Tensor],
        varbc_priors: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        loss_mtvza = self.compute_robust_channel_loss(
            mtvza_out["normalized_innovation"],
            mtvza_out["admissible_mask"],
        )
        loss_ikfs2 = self.compute_robust_channel_loss(
            ikfs2_out["masked_channel_innovation"],
            ikfs2_out["channel_mask"],
        )

        # Потери по векторам атмосферного движения (AMV из Электро-Л и Арктика-М, фильтр QI >= 0.80)
        amv_norm_innov = (amv_uv_obs - amv_uv_pred) / amv_sigma.clamp(min=1.5)
        amv_mask = (amv_qi >= 0.80).float() * amv_qi
        loss_amv = self.compute_robust_channel_loss(amv_norm_innov, amv_mask.expand_as(amv_norm_innov))

        # Регуляризация параметров VarBC относительно априорного состояния
        loss_varbc = torch.tensor(0.0, device=loss_mtvza.device)
        for p, p0 in zip(varbc_params, varbc_priors):
            loss_varbc = loss_varbc + F.mse_loss(p, p0)

        total_sat_loss = loss_mtvza + loss_ikfs2 + 0.5 * loss_amv + self.varbc_reg_weight * loss_varbc
        return {
            "loss_sat_total": total_sat_loss,
            "loss_mtvza": loss_mtvza,
            "loss_ikfs2": loss_ikfs2,
            "loss_amv": loss_amv,
            "loss_varbc_reg": loss_varbc,
        }


def masked_rmse_with_bias_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
    bias_weight: float = 0.25,
    min_effective_count: int = 30,
) -> torch.Tensor:
    """Вычисляет нормированный Huber/MSE лосс + явный штраф за систематическое смещение (bias).

    Для разреженных высотных профилей (TEMP) плавно зануляет вес на сроках,
    где число наблюдений меньше порога репрезентативности min_effective_count.
    """
    valid = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
    count = valid.sum()
    if count == 0:
        return pred.sum() * 0.0

    diff = torch.where(valid, (pred - target) / scale, torch.zeros_like(pred))

    # Среднеквадратичная компонента (с мягким сглаживанием выбросов Huber beta=3.0)
    huber = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=3.0)
    mse_term = (huber * valid).sum() / count.clamp_min(1)

    # Явный штраф за систематический сдвиг (убирает Warm Bias +2.5K и провал давления)
    mean_bias = diff.sum() / count.clamp_min(1)
    bias_term = mean_bias.pow(2)

    # Защита от градиентного взрыва на промежуточных сроках с 1-5 радиозондами
    sample_confidence = torch.clamp(count.float() / float(min_effective_count), min=0.0, max=1.0)

    return sample_confidence * (mse_term + bias_weight * bias_term)

