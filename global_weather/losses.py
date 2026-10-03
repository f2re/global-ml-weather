"""Missing-target-safe, area-weighted objectives; no target is never 'zero rain'."""
import torch
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
