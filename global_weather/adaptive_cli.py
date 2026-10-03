"""Explicitly synthetic CPU check of the v0.2 adaptive architecture."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from .grid import build_pyramid
from .cli import synthetic_records
from .observations import pack_observations
from .vertical import PRESSURE_HPA
from .adaptive import AdaptiveWeatherModel
from .training import Targets, train_step
from .checkpoints import save_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh-level', type=int, default=1)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--latent-slots', type=int, choices=(4, 8, 16, 38), default=8)
    p.add_argument('--optimizer-steps', type=int, default=2)
    p.add_argument('--horizon-hours', type=int, default=72)
    p.add_argument('--output', type=Path, default=Path('outputs/adaptive'))
    a = p.parse_args(argv)
    if a.optimizer_steps < 0: p.error('Optimizer steps must be nonnegative.')
    torch.manual_seed(17)
    torch.set_num_threads(1)
    grids = build_pyramid(a.mesh_level)
    issue = datetime(2026, 10, 3, tzinfo=timezone.utc)
    obs = pack_observations(synthetic_records(grids[0], issue), grids[0],
                            np.array(PRESSURE_HPA)*100, issue)
    model = AdaptiveWeatherModel(grids, obs.vocabulary, observation_schema=obs.schema_fingerprint,
                                  hidden=a.hidden, latent_slots=a.latent_slots,
                                  allow_unscaled_synthetic=True)
    static = torch.zeros(grids[0].n_cells)
    with torch.no_grad():
        teacher = list(model(obs, static, static, horizon_hours=3))[-1]
    target = teacher.profiles[None].clone()
    target[..., 0] += 1.
    targets = Targets((3,), target, teacher.profile_mask[None, ..., None].expand_as(target).clone(),
                      teacher.surface[None].clone(), teacher.surface_mask[None].clone(),
                      model.grid_fingerprint, model.pressure_pa)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    history = [train_step(model, optimizer, obs, static, static, targets)
               for _ in range(a.optimizer_steps)]
    model.eval()
    a.output.mkdir(parents=True, exist_ok=True)
    leads = []
    with torch.inference_mode():
        for frame in model(obs, static, static, horizon_hours=a.horizon_hours):
            if not torch.isfinite(frame.profiles).all(): raise FloatingPointError('Invalid rollout.')
            leads.append(frame.lead_hours)
        state = model.analysis_state(obs, static, static)
    save_checkpoint(a.output/'synthetic_weights.pt', model)
    report = dict(status='synthetic_adaptive_smoke', scientifically_validated=False,
                  normalization='synthetic-unscaled; NOT imported ERA5 statistics',
                  cells=grids[0].n_cells, latent_shape=list(state.latent.shape),
                  output_pressure_levels=37, forecast_leads=leads,
                  parameters=sum(p.numel() for p in model.parameters()), training=history)
    (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__': main()
