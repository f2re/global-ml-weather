"""R9 physical checks and an explicit observation-space objective policy.

Climatological normalization, observation uncertainty and QC are DIFFERENT
objects. This module never changes the fixed mean/std or an archived value.
The saturation threshold is a configurable gross-error screen, not a claim
that every radiosonde is unreliable above a universal pressure surface.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch

from .observations import utc
from .vertical import PRESSURE_HPA

RD = 287.05
EPSILON = 0.622


@dataclass(frozen=True)
class PhysicalPolicy:
    schema: str = 'profile-physical-policy-1'
    architecture: str = 'legacy'  # isolate the loss/QC experiment first
    huber_delta: float = 2.5
    humidity_qc: bool = True
    maximum_water_relative_humidity: float = 2.0
    humidity_min_pressure_pa: float = 0.0  # optional sensitivity test, NOT default
    variable_weights: tuple = (1., 1., 1., 1., 1.)
    bias_weight: float = 0.0
    memory_budget_mib: int = 8192
    bias_mode: str = 'column'  # 'column' (legacy) or 'level_charbonnier'
    bias_gate_min_count: int = 3
    quadrature_mode: str = 'mass'  # 'mass' (legacy) or 'hybrid'
    coupled_wind_loss: bool = False

    def __post_init__(self):
        if self.schema != 'profile-physical-policy-1' or self.architecture not in ('legacy', 'hydrostatic_flow'):
            raise ValueError('Unknown physical profile policy or architecture.')
        if type(self.humidity_qc) is not bool:
            raise ValueError('humidity_qc must be Boolean.')
        for name, lo, hi in (('huber_delta', .1, 100.),
                             ('maximum_water_relative_humidity', 1., 5.),
                             ('humidity_min_pressure_pa', 0., 100000.),
                             ('bias_weight', 0., 100.)):
            x = getattr(self, name)
            if type(x) not in (float, int) or not math.isfinite(x) or not lo <= x <= hi:
                raise ValueError('Invalid physical policy parameter: ' + name)
        if (len(self.variable_weights) != 5 or any(type(w) not in (float, int)
                or not math.isfinite(w) or w <= 0 for w in self.variable_weights)):
            raise ValueError('Five finite positive variable weights are required.')
        if type(self.memory_budget_mib) is not int or not 128 <= self.memory_budget_mib <= 1048576:
            raise ValueError('Invalid physical activation memory budget.')
        if self.bias_mode not in ('column', 'level_charbonnier'):
            raise ValueError('bias_mode must be column or level_charbonnier.')
        if type(self.bias_gate_min_count) is not int or self.bias_gate_min_count < 1:
            raise ValueError('bias_gate_min_count must be a positive integer.')
        if self.quadrature_mode not in ('mass', 'hybrid'):
            raise ValueError('quadrature_mode must be mass or hybrid.')
        if type(self.coupled_wind_loss) is not bool:
            raise ValueError('coupled_wind_loss must be Boolean.')

    def payload(self):
        value = asdict(self)
        value['variable_weights'] = list(value['variable_weights'])
        return value


def parse_policy(value):
    if not isinstance(value, dict) or set(value) - set(PhysicalPolicy.__dataclass_fields__):
        raise ValueError('Unknown physical policy fields.')
    return PhysicalPolicy(**value)


def pressure_layer_thickness(pressure_pa):
    """Quadrature widths over the DECLARED pressure interval, not surface mass.

    Geometric interior interfaces; endpoints are the supplied extreme levels.
    A radiosonde is not a global area sample. These widths weight pressure bins
    AFTER averaging profiles, not every reported level of a dense sounding.
    """
    p = np.asarray(pressure_pa, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or (p <= 0).any() or not (np.diff(p) < 0).all():
        raise ValueError('Strictly decreasing finite positive pressure required.')
    interfaces = np.r_[p[0], np.sqrt(p[:-1] * p[1:]), p[-1]]
    return interfaces[:-1] - interfaces[1:]


def compute_vertical_quadrature_weights(pressure_pa, quadrature_mode='mass'):
    """Compute per-variable vertical layer quadrature weights across pressure levels.

    Returns array [5, N_levels] summing to 1.0 along the level axis.
    """
    p = np.asarray(pressure_pa, dtype=np.float64)
    dp = pressure_layer_thickness(p)
    w_mass = dp / dp.sum()
    if quadrature_mode == 'mass':
        return np.tile(w_mass, (5, 1))

    # Logarithmic thickness interfaces: ln(interface_k / interface_{k+1})
    ln_p = np.log(p)
    interfaces_ln = np.r_[ln_p[0], 0.5 * (ln_p[:-1] + ln_p[1:]), ln_p[-1]]
    dlnp = interfaces_ln[:-1] - interfaces_ln[1:]

    # Burst damping above 20 hPa (p < 2000 Pa)
    burst_damping = np.where(p < 2000.0, 0.5, 1.0)
    dlnp_damped = dlnp * burst_damping
    w_log = dlnp_damped / dlnp_damped.sum()

    # 5 variables: 0: temperature, 1: specific_humidity, 2: u, 3: v, 4: geopotential
    # T and Phi get 0.5 mass + 0.5 log
    # q, u, v get 0.85 mass + 0.15 log
    w = np.zeros((5, len(p)), dtype=np.float64)
    w[0] = 0.50 * w_mass + 0.50 * w_log
    w[4] = 0.50 * w_mass + 0.50 * w_log
    w[1] = 0.85 * w_mass + 0.15 * w_log
    w[2] = 0.85 * w_mass + 0.15 * w_log
    w[3] = 0.85 * w_mass + 0.15 * w_log
    return w


def saturation_pressure_water(temperature_k):
    """Murphy & Koop (2005), Eq. 10, Pa; explicit 123 < T < 332 K domain.

    DOI: 10.1256/qj.04.94. Used only for a permissive observation QC screen.
    Water reference deliberately allows ordinary ice supersaturation.
    """
    t = float(temperature_k)
    if not math.isfinite(t) or not 123 < t < 332:
        raise ValueError('Outside Murphy-Koop water formula domain.')
    return math.exp(54.842763 - 6763.22/t - 4.210*math.log(t) + .000367*t
                    + math.tanh(.0415*(t-218.8))
                    * (53.878 - 1331.22/t - 9.44523*math.log(t) + .014025*t))


def _collocation(record):
    # Never borrow a neighbouring pressure/time or another launch's temperature.
    group = record.get('profile_id') or record.get('provider_message_id')
    if not group:
        return None
    return (record.get('provider'), group, utc(record['observed_at']),
            float(record['pressure_pa']), float(record['latitude']),
            float(record['longitude']))


def screen_humidity(records, policy: PhysicalPolicy):
    """Return retained ORIGINAL records and auditable counts; never clip truth.

    Run before bounded sampling. Missing paired T is reported as unchecked,
    not as bad humidity. No temperature from reanalysis/predictions is used.
    """
    temperatures = defaultdict(set)
    for r in records:
        if r.get('variable') == 'temperature' and r.get('valid', True) is True:
            key = _collocation(r)
            if key is not None and math.isfinite(float(r['value'])):
                temperatures[key].add(float(r['value']))
    retained, reasons, examples = [], Counter(), []
    for r in records:
        if r.get('variable') != 'specific_humidity' or not policy.humidity_qc:
            retained.append(r)
            continue
        q, p = float(r['value']), float(r['pressure_pa'])
        reason = None
        if not math.isfinite(q) or not 0 <= q < 1 or not math.isfinite(p) or p <= 0:
            reason = 'humidity_invalid_physical_range'
        elif p < policy.humidity_min_pressure_pa:
            reason = 'humidity_explicit_pressure_sensitivity_exclusion'
        else:
            values = temperatures.get(_collocation(r), set())
            if len(values) != 1:
                reasons['humidity_unchecked_missing_or_conflicting_paired_temperature'] += 1
            else:
                try:
                    es = saturation_pressure_water(next(iter(values)))
                    e = p * q / (EPSILON + (1-EPSILON)*q)
                    if e > policy.maximum_water_relative_humidity * es:
                        reason = 'humidity_paired_thermodynamic_gross_error'
                    else:
                        reasons['humidity_paired_check_passed'] += 1
                except ValueError:
                    reasons['humidity_unchecked_temperature_domain'] += 1
        if reason is None:
            retained.append(r)
        else:
            reasons[reason] += 1
            if len(examples) < 20:
                examples.append({'observation_id': r.get('observation_id'),
                                 'pressure_pa': p, 'value': q, 'reason': reason})
    return retained, {'input_records': len(records), 'retained_records': len(retained),
                      'excluded_records': len(records)-len(retained),
                      'reasons': dict(reasons), 'examples': examples,
                      'interpretation': 'gross_error_screen_not_sensor_failure_attribution'}


def hydrostatic_projection(temperature, humidity, raw_geopotential, pressure_pa, weights=None,
                           surface_geopotential=None):
    """Hydrostatic least-squares projection with a freely learned column anchor or fixed terrain anchor.

    Phi[k+1]-Phi[k] = Rd * mean(Tv[k:k+2]) * log(p[k]/p[k+1]).
    When surface_geopotential is provided (e.g. g * z_mesh), the column base is
    pinned to physical terrain. Condensate loading is not represented.
    Supported full columns are required by caller.
    """
    if temperature.shape != humidity.shape or temperature.shape != raw_geopotential.shape:
        raise ValueError('T, q and geopotential shapes must match.')
    if (pressure_pa.ndim != 1 or temperature.shape[-1] != len(pressure_pa)
            or len(pressure_pa) < 2 or not bool(torch.isfinite(pressure_pa).all())
            or not bool((pressure_pa > 0).all()) or not bool((pressure_pa[:-1] > pressure_pa[1:]).all())):
        raise ValueError('Decreasing physical pressure axis required.')
    if not all(bool(torch.isfinite(x).all()) for x in (temperature, humidity, raw_geopotential)):
        raise ValueError('Cannot project a missing/nonfinite thermodynamic column.')
    if not bool((temperature > 0).all()) or not bool(((humidity >= 0) & (humidity < 1)).all()):
        raise ValueError('Invalid virtual-temperature inputs.')
    tv = temperature * (1 + (1/EPSILON - 1)*humidity)
    thickness = RD * .5*(tv[..., :-1]+tv[..., 1:]) * torch.log(pressure_pa[:-1]/pressure_pa[1:])
    g = torch.cat((torch.zeros_like(temperature[..., :1]), thickness.cumsum(-1)), -1)
    if surface_geopotential is not None:
        anchor = surface_geopotential.unsqueeze(-1) if surface_geopotential.dim() == g.dim() - 1 else surface_geopotential
    else:
        w = torch.ones_like(pressure_pa) if weights is None else weights
        if w.shape != pressure_pa.shape or not bool(torch.isfinite(w).all()) or not bool((w > 0).all()):
            raise ValueError('Finite positive projection metric required.')
        anchor = ((raw_geopotential-g)*w).sum(-1, keepdim=True)/w.sum()
    return anchor + g


def integrate_hydrostatic_geopotential(
    t_profile: torch.Tensor,
    q_profile: torch.Tensor,
    pressure_levels_pa: torch.Tensor,
    p_surface_pa: torch.Tensor,
    t_surface_k: torch.Tensor,
    z_surface_m: torch.Tensor,
) -> torch.Tensor:
    """Вычисляет геопотенциал Phi [B, N_cells, 37] (м^2/с^2) строго по барометрической формуле.

    Снижает риск расхождения знаков смещения между T и Phi и связывает
    градиенты радиозондовых измерений геопотенциала напрямую с температурой всех слоёв ниже.
    """
    g_std = 9.80665
    r_dry = 287.058
    q_safe = torch.clamp(q_profile, min=0.0, max=0.05)
    tv_profile = t_profile * (1.0 + (1.0 / EPSILON - 1.0) * q_safe)

    p_lev = pressure_levels_pa.view(*([1] * (t_profile.dim() - 1)), -1).to(t_profile.device)
    phi_sfc = g_std * z_surface_m.unsqueeze(-1)

    # 1. Геопотенциал первого уровня сетки относительно поверхности
    tv_sfc = t_surface_k.unsqueeze(-1) * (1.0 + (1.0 / EPSILON - 1.0) * q_safe[..., :1])
    tv_mean_0 = 0.5 * (tv_sfc + tv_profile[..., :1])
    dlnp_0 = torch.log(p_surface_pa.unsqueeze(-1).clamp_min(100.0) / p_lev[..., :1])
    phi_0 = phi_sfc + r_dry * tv_mean_0 * dlnp_0

    # 2. Послойное интегрирование между 37 изобарическими уровнями вверх
    tv_layer = 0.5 * (tv_profile[..., :-1] + tv_profile[..., 1:])
    dlnp_layers = torch.log(p_lev[..., :-1] / p_lev[..., 1:])
    dphi_layers = r_dry * tv_layer * dlnp_layers

    phi_upper = phi_0 + torch.cumsum(dphi_layers, dim=-1)
    return torch.cat([phi_0, phi_upper], dim=-1)



def physical_objective(model, frames, targets, policy: PhysicalPolicy):
    """Huber -> profile/bin -> pressure quadrature -> lead -> variable mean.

    Records with explicit observation_error use sqrt(sigma_climate^2+sigma_obs^2)
    as a robust residual scale, NOT a claimed likelihood. Without error metadata
    the fixed sigma is unchanged. QC must already have run before sampling.
    """
    from .profile_training import VARIABLES
    from .analysis.observation_operator import _prediction
    bins, counts, physical_errors = defaultdict(list), [0]*5, defaultdict(list)
    residuals_by_var = defaultdict(list)
    residuals_by_var_level = defaultdict(list)
    linear_counts = [0]*5
    p = np.array(PRESSURE_HPA, dtype=float)*100
    w_levels = compute_vertical_quadrature_weights(p, policy.quadrature_mode)
    origin = utc(frames[0].valid_time)

    # If coupled_wind_loss, pair u and v by (lead, level, provider, group)
    wind_pairs = defaultdict(dict) if policy.coupled_wind_loss else None

    for r in targets:
        v = VARIABLES.index(r['variable'])
        _, std, supported = model.normalization.at(v, r['pressure_pa'])
        if not supported:
            continue
        result = _prediction(frames, model.grid, r, model.pressure_pa, 3)
        if result is None:
            continue
        predicted = result[0]
        if not bool(torch.isfinite(predicted)):
            raise FloatingPointError('Nonfinite supported physical prediction.')
        truth = float(r['value'])
        if not math.isfinite(truth):
            raise ValueError('Nonfinite observed target.')
        error = r.get('observation_error', 0.)
        quality = r.get('quality', 1.)
        if (type(error) not in (int, float) or not math.isfinite(error) or error < 0
                or type(quality) not in (int, float) or not math.isfinite(quality) or not 0 < quality <= 1):
            raise ValueError('Invalid uncertainty or quality metadata.')
        if model.normalization.humidity_transform != 'identity':
            raise ValueError('R9 requires unchanged physical GraphCast normalization.')
        residual = (predicted-truth)/math.hypot(float(std), float(error))
        residuals_by_var[v].append(residual)
        absolute = residual.abs()
        delta = policy.huber_delta
        level = int(np.abs(np.log(p)-math.log(r['pressure_pa'])).argmin())
        residuals_by_var_level[(v, level)].append(residual)
        lead = max(0, math.ceil((utc(r['observed_at'])-origin).total_seconds()/10800))
        group = r.get('profile_id') or r.get('provider_message_id') or r.get('observation_id')
        if not group:
            raise ValueError('Observation/launch identity required; station catalog ID is not required.')

        if policy.coupled_wind_loss and v in (2, 3):
            wind_key = (lead, level, str(r.get('provider')), group)
            wind_pairs[wind_key][v] = (residual, quality, absolute)
            counts[v] += 1
            physical_errors[v].append(predicted.detach() - truth)
        else:
            clipped = absolute.clamp_max(delta)
            term = .5*clipped.square() + delta*(absolute-clipped)
            bins[(v, lead, level, str(r.get('provider')), group)].append((term, float(quality)))
            counts[v] += 1
            linear_counts[v] += int(bool(absolute.detach() > delta))
            physical_errors[v].append(predicted.detach()-truth)

    if policy.coupled_wind_loss and wind_pairs:
        delta = policy.huber_delta
        for (lead, level, provider, group), pair_data in wind_pairs.items():
            if 2 in pair_data and 3 in pair_data:
                r_u, q_u, abs_u = pair_data[2]
                r_v, q_v, abs_v = pair_data[3]
                r_vec = torch.sqrt(r_u.pow(2) + r_v.pow(2) + 1e-12)
                clipped_vec = r_vec.clamp_max(delta)
                term_vec = .5 * clipped_vec.square() + delta * (r_vec - clipped_vec)
                half_term = 0.5 * term_vec
                qual = 0.5 * (float(q_u) + float(q_v))
                bins[(2, lead, level, provider, group)].append((half_term, qual))
                bins[(3, lead, level, provider, group)].append((half_term, qual))
                is_lin = int(bool(r_vec.detach() > delta))
                linear_counts[2] += is_lin
                linear_counts[3] += is_lin
            else:
                for v_wind, (r_w, q_w, abs_w) in pair_data.items():
                    clipped_w = abs_w.clamp_max(delta)
                    term_w = .5 * clipped_w.square() + delta * (abs_w - clipped_w)
                    bins[(v_wind, lead, level, provider, group)].append((term_w, float(q_w)))
                    linear_counts[v_wind] += int(bool(abs_w.detach() > delta))

    if not bins:
        raise ValueError('No supported observed targets for R9 objective.')

    strata = defaultdict(list)
    for (v, lead, level, provider, group), rows in bins.items():
        weights = rows[0][0].new_tensor([r[1] for r in rows])
        strata[(v, lead, level)].append((torch.stack([r[0] for r in rows])*weights).sum()/weights.sum())

    leads = defaultdict(list)
    for (v, lead, level), values in strata.items():
        leads[(v, lead)].append((torch.stack(values).mean(), float(w_levels[v, level])))

    variables = defaultdict(list)
    for (v, lead), values in leads.items():
        denominator = sum(w for _, w in values)
        variables[v].append(sum(x*w for x, w in values)/denominator)

    terms = {}
    for v, x in variables.items():
        base_term = torch.stack(x).mean()
        if policy.bias_weight > 0.0:
            if policy.bias_mode == 'level_charbonnier':
                level_penalties = []
                level_weights_list = []
                for k in range(len(p)):
                    res_k = residuals_by_var_level.get((v, k), [])
                    if len(res_k) >= policy.bias_gate_min_count:
                        mean_b_k = torch.stack(res_k).mean()
                        delta_b = 1.0
                        charb = delta_b * delta_b * (torch.sqrt(1.0 + (mean_b_k / delta_b).pow(2)) - 1.0)
                        level_penalties.append(charb * float(w_levels[v, k]))
                        level_weights_list.append(float(w_levels[v, k]))
                if level_penalties:
                    bias_penalty = torch.stack(level_penalties).sum() / sum(level_weights_list)
                    terms[v] = base_term + policy.bias_weight * bias_penalty
                else:
                    terms[v] = base_term
            elif v in residuals_by_var and residuals_by_var[v]:
                mean_bias = torch.stack(residuals_by_var[v]).mean()
                terms[v] = base_term + policy.bias_weight * mean_bias.pow(2)
            else:
                terms[v] = base_term
        else:
            terms[v] = base_term

    loss = sum(policy.variable_weights[v]*x for v, x in terms.items())/sum(policy.variable_weights[v] for v in terms)

    unpenalized_loss = sum(policy.variable_weights[v] * torch.stack(variables[v]).mean()
                           for v in variables) / sum(policy.variable_weights[v] for v in variables)

    report = {'schema': 'profile-physical-objective-report-1', 'accepted': counts,
              'profile_pressure_lead_groups': len(bins),
              'variable_loss': {VARIABLES[v]: float(x.detach()) for v, x in terms.items()},
              'physical_rmse': {VARIABLES[v]: float(torch.stack(x).square().mean().sqrt())
                                for v, x in physical_errors.items()},
              'huber_linear_fraction': {VARIABLES[v]: linear_counts[v]/max(counts[v], 1) for v in terms},
              'unpenalized_loss': float(unpenalized_loss.detach()),
              'normalization_changed': False}
    return loss, counts, report
