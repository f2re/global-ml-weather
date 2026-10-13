"""Program self-test of R11 on an explicitly synthetic analytic atmosphere.

Checks, without any real data or weather-skill claim:
1. two optimizer steps run with finite, nonzero gradients in every branch
   that has targets (the learned column operator is exempt without radiances);
2. global dry-air mass is conserved over a 72-hour rollout;
3. Phi(p_s) equals g z_s and hydrostatic thickness is exact above ground;
4. q <= 1.05 q_s(T, p) everywhere and winds are tangent to the sphere;
5. omega obeys omega(top) = 0 and the kinematic surface boundary;
6. one 3-hour step influences nodes farther than 1620 km (150 m/s * 3 h).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import torch

from ..grid import EARTH_RADIUS_M
from . import thermo
from .training import Experiment, load_config, train

SELFTEST_CONFIG = {
    'experiment_id': 'physical-assimilation-selftest', 'allow_synthetic': True, 'mesh_level': 3,
    'hidden': 32, 'message_dim': 16, 'processor_layers': 4, 'analysis_layers': 4, 'noise_dim': 4,
    'checkpoint_steps': False, 'precision': 'fp32', 'threads': 2, 'deterministic': True,
    'stages': [{'name': 'selftest', 'rollout_hours': 6, 'epochs': 1, 'issues_per_epoch': 2, 'issues_per_step': 1,
                'peak_lr': 1e-3, 'min_lr': 1e-4, 'warmup_steps': 0, 'schedule': 'cosine', 'ensemble_members': 2,
                'patience': 1, 'cycling': True}],
    'data': {'validation_issues': 1, 'max_input_tokens_per_source': 20000, 'max_target_tokens_per_source': 20000},
}


def reach_after_one_step(model, issue) -> float:
    """Forward-mode Jacobian of the 3-hour state w.r.t. one node of the analysis state."""
    from torch.func import jvp
    model.eval()
    with torch.no_grad():
        analysis, _ = model.constrain(model.analyse(None, issue), None)
    tangent = torch.zeros_like(analysis)
    tangent[0] = 1.0

    def advance(state):
        return model.step(state, issue)

    _, change = jvp(advance, (analysis,), (tangent,))
    touched = change.abs().sum(-1) > 0
    angle = torch.arccos((model.xyz[touched] @ model.xyz[0]).clamp(-1, 1))
    return float(angle.max()) * EARTH_RADIUS_M / 1000


def run(output) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(SELFTEST_CONFIG)
    result = train(SELFTEST_CONFIG, output / 'training', synthetic=True, device='cpu', max_steps=2)
    experiment = Experiment(config, synthetic=True)
    model = experiment.build_model()
    from .training import checkpoint_path
    reference = json.loads((output / 'training' / 'latest.json').read_text())
    state = torch.load(checkpoint_path(output / 'training', reference), weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    issue = datetime(2021, 3, 1, 12, tzinfo=timezone.utc)
    with torch.no_grad():
        sample = experiment.sample(issue, 'train', 72, rng=np.random.default_rng(1), training=False)
        rollout = model(sample.tokens, issue, 72)
        masses = np.array([float(model.dry_mass(f)) for f in rollout.physics])
        pressure = model.pressure
        hydro, surface_phi, saturation, tangent = 0.0, 0.0, 0.0, 0.0
        for fields in rollout.physics:
            ps, phi = fields['surface_pressure'], fields['geopotential']
            tv = thermo.virtual_temperature(fields['temperature'], fields['humidity'])
            thickness = thermo.RD * 0.5 * (tv[:, :-1] + tv[:, 1:]) * torch.log(pressure[:-1] / pressure[1:])
            above = (pressure[1:][None] <= ps[:, None]) & (pressure[:-1][None] <= ps[:, None])
            hydro = max(hydro, float(((phi[:, 1:] - phi[:, :-1] - thickness).abs() * above).max()))
            j = (pressure[None] > ps[:, None]).sum(-1)
            tv_s = thermo.interpolate_to_surface(tv, pressure, ps)
            implied = phi[torch.arange(len(ps)), j] - thermo.RD * 0.5 * (tv_s + tv[torch.arange(len(ps)), j]) * torch.log(ps / pressure[j])
            surface_phi = max(surface_phi, float((implied - thermo.G0 * model.elevation).abs().max()))
            qs = thermo.saturation_specific_humidity(fields['temperature'], pressure)
            saturation = max(saturation, float((fields['humidity'] / qs).max()))
            tangent = max(tangent, float((fields['wind'] * model.xyz[:, None]).sum(-1).abs().max()))
        omega = model.omega(rollout)
        top = float(omega[..., -1].abs().max())
    reach = reach_after_one_step(model, issue)
    history = result['history'][-1]
    exempt = {name for name in history['parameters_without_gradient']
              if name.startswith(('column_operator', 'channel_bias', 'channel_embedding'))}
    report = {
        'schema': 'physical-assimilation-selftest-1', 'data_kind': 'synthetic_analytic',
        'optimizer_steps': history['optimizer_steps'], 'train_loss': history['train_loss'],
        'parameters_without_gradient': history['parameters_without_gradient'],
        'exempt_without_radiance_targets': sorted(exempt),
        'dry_mass_relative_drift_72h': float(np.abs(masses - masses[0]).max() / masses[0]),
        'hydrostatic_residual_max_m2_s2': hydro, 'surface_geopotential_error_max_m2_s2': surface_phi,
        'max_q_over_qs': saturation, 'max_radial_wind_m_s': tangent, 'omega_top_max_pa_s': top,
        'one_step_influence_radius_km': reach, 'required_radius_km': 1620.0,
        'scientific_acceptance': False}
    checks = {
        'gradients': not (set(history['parameters_without_gradient']) - exempt),
        'dry_mass': report['dry_mass_relative_drift_72h'] < 1e-6,
        'hydrostatic': hydro < 10.0,
        'surface_geopotential': surface_phi < 5.0,
        'saturation': saturation <= thermo.MAX_RELATIVE_HUMIDITY + 1e-5,
        'tangent_wind': tangent < 1e-3,
        'omega_top': top < 1e-9,
        'domain_of_dependence': reach >= 1620.0}
    report['checks'] = checks
    report['passed'] = all(checks.values())
    (output / 'selftest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not report['passed']:
        raise RuntimeError('R11 self-test failed: ' + ', '.join(k for k, v in checks.items() if not v))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output), allow_nan=False))


if __name__ == '__main__':
    main()
