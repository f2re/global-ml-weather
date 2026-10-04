"""Conditional tau-omega inversion with an externally justified dielectric LUT.

No SMAP constants are silently transferred to MTVZA frequencies. The supplied
LUT must represent this platform, frequency, incidence, soil and roughness.
"""
from dataclasses import dataclass
import numpy as np
from .core import Field, QC, aligned, condition, result, require_hash, canonical
from .algorithms import _identity, _flags

SOIL_CHECKS = {'unfrozen', 'snow_free', 'no_precipitation', 'open_water_excluded', 'rfi_screened'}


@dataclass(frozen=True)
class SoilEmissivityLUT:
    moisture: np.ndarray
    emissivity_hv: np.ndarray
    metadata: dict
    source_sha256: str

    def __post_init__(self):
        x, e = np.asarray(self.moisture, float), np.asarray(self.emissivity_hv, float)
        if x.ndim != 1 or not 3 <= len(x) <= 4096 or e.shape != (len(x), 2):
            raise ValueError('Таблица должна содержать 3–4096 значений влажности и две поляризации.')
        if not np.isfinite(x).all() or not np.isfinite(e).all() or not ((0 <= x) & (x <= 1)).all() or not (np.diff(x) > 0).all():
            raise ValueError('Неверная ось объёмной влажности.')
        if not ((0 < e) & (e <= 1)).all() or not (np.diff(e, axis=0) < 0).all():
            raise ValueError('Нужна монотонная зависимость излучательной способности от влажности без плоских участков.')
        m = self.metadata
        for key in ('source', 'platform', 'instrument', 'condition_id', 'dielectric_model', 'soil_texture', 'roughness', 'license', 'reference'):
            if not isinstance(m.get(key), str) or not m[key].strip():
                raise ValueError(f'Не задано происхождение таблицы: {key}')
        if m.get('data_kind') not in ('real', 'synthetic') or m.get('units') != 'm3 m-3':
            raise ValueError('Неверное происхождение или единицы таблицы.')
        if not 1 <= m.get('frequency_ghz', 0) <= 40 or not 0 <= m.get('incidence_deg', -1) < 80:
            raise ValueError('Неверная частота или геометрия таблицы.')
        if not 0 < m.get('depth_bottom_m', 0) <= .1 or m.get('depth_top_m') != 0:
            raise ValueError('Это поверхностная оценка; корневая зона не поддерживается.')
        if len(m.get('channel_ids', [])) != 2 or len(set(m['channel_ids'])) != 2:
            raise ValueError('Укажите физические H/V каналы.')
        require_hash(self.source_sha256); canonical(m)
        object.__setattr__(self, 'moisture', x.copy())
        object.__setattr__(self, 'emissivity_hv', e.copy())


def tau_omega_forward(emissivity, soil_temperature, vegetation_temperature, tau_nadir, omega, incidence_deg):
    """TB=Ts*e*gamma + Tv*(1-omega)*(1-gamma)*(1+(1-e)*gamma), K.

    Atmospheric/downwelling sky effects must be corrected separately. Scattering
    is represented by the single albedo parameter, no canopy multiple scattering.
    """
    gamma = np.exp(-np.asarray(tau_nadir)/np.cos(np.deg2rad(incidence_deg)))
    vegetation = np.asarray(vegetation_temperature)*(1-np.asarray(omega))*(1-gamma)
    return np.asarray(soil_temperature)*emissivity*gamma + vegetation*(1+(1-emissivity)*gamma)


def soil_moisture(tb_h: Field, tb_v: Field, soil_temperature: Field, vegetation_temperature: Field,
                  tau_nadir: Field, omega: Field, incidence: Field, eligible: Field, lut: SoilEmissivityLUT,
                  *, available_at, data_kind, max_chi2=9., max_conditional_sigma=.10):
    """Weighted two-channel inversion along the piecewise-linear emissivity curve.

    Ancillaries and LUT condition are fixed, not jointly retrieved. Returned
    uncertainty includes TB noise only; it is NOT total retrieval uncertainty.
    """
    for f in (tb_h, tb_v):
        f.require('surface_brightness_temperature', 'K')
        if f.metadata.get('atmosphere_corrected') is not True or f.uncertainty is None:
            raise ValueError('Нужны атмосферно скорректированная TB и её погрешность, не цифровые отсчёты.')
    soil_temperature.require('effective_soil_temperature', 'K')
    vegetation_temperature.require('vegetation_temperature', 'K')
    tau_nadir.require('vegetation_optical_depth_nadir', '1'); omega.require('vegetation_scattering_albedo', '1')
    incidence.require('view_zenith_angle', 'degree')
    if not SOIL_CHECKS.issubset(set(eligible.metadata.get('checks', []))):
        raise ValueError('Нужны проверки снега, промерзания, осадков, открытой воды и радиопомех.')
    if not np.isfinite([max_chi2, max_conditional_sigma]).all() or min(max_chi2, max_conditional_sigma) <= 0:
        raise ValueError('Некорректные пороги проверки инверсии.')
    if lut.metadata['data_kind'] != data_kind:
        raise ValueError('Синтетическая таблица не допускается как реальная калибровка.')
    for k, f in enumerate((tb_h, tb_v)):
        for key in ('source', 'platform', 'instrument', 'frequency_ghz', 'condition_id'):
            if f.metadata.get(key) != lut.metadata[key]:
                raise ValueError('Таблица не соответствует частоте, платформе или физическим условиям.')
        if f.metadata.get('polarization') != ('H', 'V')[k] or f.metadata.get('channel_id') != lut.metadata['channel_ids'][k]:
            raise ValueError('Несовместимая поляризация или физический канал.')
        if f.metadata.get('footprint_id') is None or f.metadata.get('footprint_id') != tb_h.metadata.get('footprint_id'):
            raise ValueError('H/V измерения должны иметь согласованную область чувствительности.')
    fields = [tb_h, tb_v, soil_temperature, vegetation_temperature, tau_nadir, omega, incidence, eligible]
    valid = aligned(tb_h, tb_v, max_skew_seconds=60.) & aligned(*fields, max_skew_seconds=3600.)
    qc = _flags(valid)
    ok = condition(eligible, tb_h)
    qc[~ok] |= int(QC.CONDITIONS)
    geometry = np.abs(incidence.values-lut.metadata['incidence_deg']) <= .05
    physical = ((tb_h.values > 0) & (tb_v.values > 0) & (soil_temperature.values > 0) & (vegetation_temperature.values > 0)
                & (tau_nadir.values >= 0) & (omega.values >= 0) & (omega.values <= 1) & geometry)
    noise = np.stack([tb_h.uncertainty, tb_v.uncertainty], -1)
    physical &= np.isfinite(noise).all(-1) & (noise > 0).all(-1)
    qc[~physical] |= int(QC.DOMAIN)
    valid &= ok & physical
    output = np.full(tb_h.values.shape, np.nan); uncertainty = output.copy()
    for ind in np.ndindex(output.shape):
        if not valid[ind]: continue
        predicted = tau_omega_forward(lut.emissivity_hv, soil_temperature.values[ind],
                        vegetation_temperature.values[ind], tau_nadir.values[ind], omega.values[ind], incidence.values[ind])
        observed = np.array([tb_h.values[ind], tb_v.values[ind]])
        w = 1/noise[ind]**2
        delta = np.diff(predicted, axis=0)
        slopes = delta/np.diff(lut.moisture)[:, None]
        denom = (w*delta**2).sum(1)
        f = np.clip((w*delta*(observed-predicted[:-1])).sum(1)/np.maximum(denom, 1e-30), 0., 1.)
        chi2 = (w*(observed-(predicted[:-1]+f[:, None]*delta))**2).sum(1)
        j = int(np.argmin(chi2))
        theta = lut.moisture[j]+f[j]*(lut.moisture[j+1]-lut.moisture[j])
        sensitivity = np.sqrt((w*slopes[j]**2).sum())
        sigma = 1/sensitivity if sensitivity > 0 else np.inf
        if chi2[j] > max_chi2:
            qc[ind] |= int(QC.NO_SOLUTION); valid[ind] = False
        if sigma > max_conditional_sigma:
            qc[ind] |= int(QC.LOW_SENSITIVITY); valid[ind] = False
        if theta <= lut.moisture[0]+1e-10 or theta >= lut.moisture[-1]-1e-10:
            qc[ind] |= int(QC.BOUNDARY); valid[ind] = False
        output[ind], uncertainty[ind] = theta, sigma
    # LUT provenance is a fixed parameter artifact, not a fictitious timed observation.
    return result('soil_moisture_surface', 'tau-omega-lut-v1', output, valid, qc,
                  dependencies=fields, uncertainty=uncertainty,
                  assumptions=['fixed_dielectric_roughness_and_vegetation_parameters', 'surface_layer_not_root_zone',
                               'tau_omega_no_multiple_scattering', 'atmospheric_and_sky_correction_external',
                               'conditional_tb_noise_not_total_error'],
                  attributes={'depth_top_m': 0, 'depth_bottom_m': lut.metadata['depth_bottom_m'],
                              'lut_sha256': lut.source_sha256, 'lut_provenance': lut.metadata,
                              'max_chi2': max_chi2, 'max_conditional_sigma': max_conditional_sigma},
                  **_identity(tb_h, available_at, data_kind))
