"""A supervised step for prepared real or synthetic samples.

Data retrieval, conservative regridding, splits and multi-epoch orchestration
remain external. This module does not infer missing targets from observations.
"""
from dataclasses import dataclass
from datetime import timedelta
import torch
import numpy as np
from .physics import hydrostatic_penalty
from .losses import forecast_loss
from .observations import utc


@dataclass
class Targets:
    lead_hours: tuple[int, ...]
    profiles: torch.Tensor      # [K,N,L,6], physical units
    profile_mask: torch.Tensor  # same shape, from target QC and target terrain
    surface: torch.Tensor       # [K,N,8]
    surface_mask: torch.Tensor  # same shape
    grid_fingerprint: str
    pressure_pa: torch.Tensor

    def to(self, device):
        from dataclasses import fields
        return type(self)(**{f.name: (getattr(self, f.name).to(device)
                           if isinstance(getattr(self, f.name), torch.Tensor) else getattr(self, f.name))
                           for f in fields(self)})

    def validate(self,model):
        if self.grid_fingerprint != model.grid_fingerprint or not torch.equal(self.pressure_pa,model.pressure_pa):
            raise ValueError('Target grid/pressure axis mismatch.')
        leads = self.lead_hours
        if not leads or tuple(sorted(set(leads))) != leads or leads[0] < 0 or leads[-1] > 72 or any(x%model.step_hours for x in leads):
            raise ValueError('Targets need unique sorted forecast leads compatible with model step.')
        k,n,l = len(leads),len(model.xyz),len(model.pressure_pa)
        if self.profiles.shape != (k,n,l,6) or self.surface.shape != (k,n,8):
            raise ValueError('Target shape mismatch.')
        if self.profile_mask.shape != self.profiles.shape or self.surface_mask.shape != self.surface.shape:
            raise ValueError('Every target variable requires a mask.')
        if self.profile_mask.dtype != torch.bool or self.surface_mask.dtype != torch.bool:
            raise ValueError('Target masks must be Boolean.')
        if (self.profile_mask & ~torch.isfinite(self.profiles)).any() or (self.surface_mask & ~torch.isfinite(self.surface)).any():
            raise ValueError('Nonfinite target marked as observed.')
        if 0 in leads and self.surface_mask[leads.index(0),:,6].any():
            raise ValueError('Lead-zero precipitation has no forecast accumulation interval.')
        # Known target surface pressure is authoritative, never predicted ps.
        known_ps=self.surface_mask[:,:,4]
        ps=self.surface[:,:,4]
        below=model.pressure_pa[None,None,:] > ps[:,:,None]
        if (self.profile_mask.any(-1) & below & known_ps[:,:,None]).any():
            raise ValueError('Targets below known terrain must be masked.')


def assert_time_separation(train_issue_times,validation_issue_times,*,horizon_hours=72,history_hours=12):
    """Strict chronological split: no shared input/target window at the boundary."""
    if not train_issue_times or not validation_issue_times:
        raise ValueError('Both train and validation time sets are required.')
    if horizon_hours <= 0 or history_hours <= 0:
        raise ValueError('Positive windows required.')
    last_target=max(map(utc,train_issue_times))+timedelta(hours=horizon_hours)
    first_input=min(map(utc,validation_issue_times))-timedelta(hours=history_hours)
    if last_target >= first_input:
        raise ValueError('Train/validation observation or target windows overlap.')


def train_step(model,optimizer,observations,elevation_m,land_fraction,targets: Targets,*,grad_clip=1.,physics_weight=0.):
    if physics_weight < 0 or not np.isfinite(physics_weight):
        raise ValueError("Physics weight must be finite and nonnegative.")
    if grad_clip <= 0:
        raise ValueError('Positive gradient clipping threshold required.')
    targets.validate(model)
    if not observations.accepted_records:
        raise ValueError('No usable observations; do not silently train a climatology-only sample.')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lookup={lead:i for i,lead in enumerate(targets.lead_hours)}
    losses=[]
    area=model.processor.graphs[0].area
    for frame in model(observations,elevation_m,land_fraction,horizon_hours=targets.lead_hours[-1]):
        if frame.lead_hours not in lookup:
            continue
        i=lookup[frame.lead_hours]
        losses.append(forecast_loss(frame,targets.profiles[i],targets.profile_mask[i],
                                   targets.surface[i],targets.surface_mask[i],area,
                                   normalization=getattr(model,'normalization',None),
                                   pressure_pa=model.pressure_pa,step_hours=model.step_hours))
        if physics_weight:
            losses[-1]=losses[-1]+physics_weight*hydrostatic_penalty(
                frame.profiles,model.pressure_pa,targets.profile_mask[i])
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss.')
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError('Missing/nonfinite gradients; optimizer was not stepped.')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip,error_if_nonfinite=True)
    optimizer.step()
    return {'loss':float(loss.detach()),'gradient_norm':float(norm),'supervised_leads':len(losses)}
