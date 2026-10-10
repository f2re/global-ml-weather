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
    memory_budget_mib: int = 8192

    def __post_init__(self):
        if self.schema != 'profile-physical-policy-1' or self.architecture not in ('legacy', 'hydrostatic_flow'):
            raise ValueError('Unknown physical profile policy or architecture.')
        if type(self.humidity_qc) is not bool:
            raise ValueError('humidity_qc must be Boolean.')
        for name, lo, hi in (('huber_delta', .1, 100.),
                             ('maximum_water_relative_humidity', 1., 5.),
                             ('humidity_min_pressure_pa', 0., 100000.)):
            x = getattr(self, name)
            if type(x) not in (float, int) or not math.isfinite(x) or not lo <= x <= hi:
                raise ValueError('Invalid physical policy parameter: ' + name)
        if (len(self.variable_weights) != 5 or any(type(w) not in (float, int)
                or not math.isfinite(w) or w <= 0 for w in self.variable_weights)):
            raise ValueError('Five finite positive variable weights are required.')
        if type(self.memory_budget_mib) is not int or not 128 <= self.memory_budget_mib <= 1048576:
            raise ValueError('Invalid physical activation memory budget.')

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


def hydrostatic_projection(temperature, humidity, raw_geopotential, pressure_pa, weights=None):
    """Hydrostatic least-squares projection with a freely learned column anchor.

    Phi[k+1]-Phi[k] = Rd * mean(Tv[k:k+2]) * log(p[k]/p[k+1]).
    The additive anchor minimizes weighted squared distance to raw Phi. It is
    NOT pinned to zero, ISA, climatology or a guessed surface height. Condensate
    loading is not represented. Supported full columns are required by caller.
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
    w = torch.ones_like(pressure_pa) if weights is None else weights
    if w.shape != pressure_pa.shape or not bool(torch.isfinite(w).all()) or not bool((w > 0).all()):
        raise ValueError('Finite positive projection metric required.')
    anchor = ((raw_geopotential-g)*w).sum(-1, keepdim=True)/w.sum()
    return anchor + g


def physical_objective(model, frames, targets, policy: PhysicalPolicy):
    """Huber -> profile/bin -> pressure quadrature -> lead -> variable mean.

    Records with explicit observation_error use sqrt(sigma_climate^2+sigma_obs^2)
    as a robust residual scale, NOT a claimed likelihood. Without error metadata
    the fixed sigma is unchanged. QC must already have run before sampling.
    """
    from .profile_training import VARIABLES
    from .analysis.observation_operator import _prediction
    bins, counts, physical_errors = defaultdict(list), [0]*5, defaultdict(list)
    linear_counts = [0]*5
    p = np.array(PRESSURE_HPA, dtype=float)*100
    dp = pressure_layer_thickness(p)
    origin = utc(frames[0].valid_time)
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
        absolute = residual.abs()
        delta = policy.huber_delta
        clipped = absolute.clamp_max(delta)
        term = .5*clipped.square() + delta*(absolute-clipped)
        level = int(np.abs(np.log(p)-math.log(r['pressure_pa'])).argmin())
        lead = max(0, math.ceil((utc(r['observed_at'])-origin).total_seconds()/10800))
        group = r.get('profile_id') or r.get('provider_message_id') or r.get('observation_id')
        if not group:
            raise ValueError('Observation/launch identity required; station catalog ID is not required.')
        bins[(v, lead, level, str(r.get('provider')), group)].append((term, float(quality)))
        counts[v] += 1
        linear_counts[v] += int(bool(absolute.detach() > delta))
        physical_errors[v].append(predicted.detach()-truth)
    if not bins:
        raise ValueError('No supported observed targets for R9 objective.')
    strata = defaultdict(list)
    for (v, lead, level, provider, group), rows in bins.items():
        weights = rows[0][0].new_tensor([r[1] for r in rows])
        strata[(v, lead, level)].append((torch.stack([r[0] for r in rows])*weights).sum()/weights.sum())
    leads = defaultdict(list)
    for (v, lead, level), values in strata.items():
        leads[(v, lead)].append((torch.stack(values).mean(), float(dp[level])))
    variables = defaultdict(list)
    for (v, lead), values in leads.items():
        denominator = sum(w for _, w in values)
        variables[v].append(sum(x*w for x, w in values)/denominator)
    terms = {v: torch.stack(x).mean() for v, x in variables.items()}
    loss = sum(policy.variable_weights[v]*x for v, x in terms.items())/sum(policy.variable_weights[v] for v in terms)
    report = {'schema': 'profile-physical-objective-report-1', 'accepted': counts,
              'profile_pressure_lead_groups': len(bins),
              'variable_loss': {VARIABLES[v]: float(x.detach()) for v, x in terms.items()},
              'physical_rmse': {VARIABLES[v]: float(torch.stack(x).square().mean().sqrt())
                                for v, x in physical_errors.items()},
              'huber_linear_fraction': {VARIABLES[v]: linear_counts[v]/counts[v] for v in terms},
              'normalization_changed': False}
    return loss, counts, report
