"""Restricted physical retrievals; no empirical weather classes disguised as units."""
import numpy as np
from .core import Field, QC, aligned, condition, result, utc


def _flags(valid):
    return np.where(valid, 0, int(QC.MISSING)).astype(np.uint16)


def _identity(primary, available_at, data_kind):
    return dict(primary=primary, available_at=available_at,
                source=primary.metadata['source'], platform=primary.metadata['platform'], data_kind=data_kind)


def spectral_index(a: Field, b: Field, eligible: Field, solar_zenith: Field, *,
                   product, available_at, data_kind, max_solar_zenith=80.):
    """(a-b)/(a+b) on aligned reflectances. Conditions must exclude clouds/snow as appropriate."""
    roles = {'ndvi': ('nir', 'red'), 'ndvi_toa': ('nir', 'red'),
             'ndmi': ('nir', 'swir_1p6'), 'ndsi': ('green', 'swir_1p6')}
    if product not in roles or not 0 < max_solar_zenith < 90:
        raise ValueError('Неизвестный индекс или предел солнечного угла.')
    quantity = 'toa_reflectance' if product == 'ndvi_toa' else 'surface_reflectance'
    a.require(quantity, '1'); b.require(quantity, '1'); solar_zenith.require('solar_zenith_angle', 'degree')
    if (a.metadata.get('spectral_role'), b.metadata.get('spectral_role')) != roles[product]:
        raise ValueError('Спектральные полосы не соответствуют формуле. Номера каналов не угадываются.')
    if not all(a.metadata.get(k) for k in ('source','platform','calibration_family')):
        raise ValueError('Неизвестная платформа или калибровка отражения.')
    if any(a.metadata.get(k) != b.metadata.get(k) for k in ('source', 'platform', 'calibration_family')):
        raise ValueError('Каналы разных платформ или несовместимой калибровки.')
    valid = aligned(a, b, max_skew_seconds=60.) & aligned(a, solar_zenith)
    qc = _flags(valid)
    ok = condition(eligible, a) & (solar_zenith.values >= 0) & (solar_zenith.values <= max_solar_zenith)
    qc[~ok] |= int(QC.CONDITIONS)
    denominator = a.values+b.values
    physical = (a.values >= 0) & (a.values <= 1) & (b.values >= 0) & (b.values <= 1) & (denominator > 1e-8)
    qc[~physical] |= int(QC.DOMAIN)
    with np.errstate(divide='ignore', invalid='ignore'):
        value = (a.values-b.values)/denominator
        uncertainty = None
        if a.uncertainty is not None and b.uncertainty is not None:
            uncertainty = np.sqrt((2*b.values*a.uncertainty)**2 + (2*a.values*b.uncertainty)**2)/denominator**2
    method = {'ndvi': 'ndvi-surface-v1', 'ndvi_toa': 'ndvi-toa-v1',
              'ndmi': 'ndmi-surface-v1', 'ndsi': 'ndsi-surface-v1'}[product]
    return result(product, method, value, valid & ok & physical, qc,
                  dependencies=[a, b, eligible, solar_zenith], uncertainty=uncertainty,
                  assumptions=['aligned_spectral_bands', 'independent_band_errors_if_reported',
                               'index_not_soil_moisture_or_snow_fraction'],
                  attributes={'max_solar_zenith_deg': max_solar_zenith,
                              'spectral_roles': list(roles[product]), 'reflectance_kind': quantity},
                  **_identity(a, available_at, data_kind))


def cloud_brightness_temperature(bt: Field, cloudy: Field, *, available_at, data_kind):
    """Cloud-mask-selected window brightness temperature, NOT thermodynamic cloud-top T."""
    bt.require('brightness_temperature', 'K')
    if bt.metadata.get('spectral_role') != 'thermal_window':
        raise ValueError('Нужен проверенный тепловой канал окна прозрачности.')
    valid = bt.valid & condition(cloudy, bt)
    qc = _flags(bt.valid); qc[~condition(cloudy, bt)] |= int(QC.CONDITIONS)
    return result('cloud_top_brightness_temperature', 'cloud-ir-bt-v1', bt.values, valid, qc,
                  dependencies=[bt, cloudy], uncertainty=bt.uncertainty,
                  assumptions=['cloud_mask_external', 'brightness_temperature_not_true_cloud_top_temperature'],
                  **_identity(bt, available_at, data_kind))


def cloud_height(cloud_temperature: Field, temperature_profile: Field, height_profile: Field,
                 opaque_single_layer: Field, *, available_at, data_kind):
    """Unique piecewise-linear T(z) crossing. Ambiguous inversions are masked, not guessed.

    Heights and their datum are provided explicitly. No fixed lapse rate, ISA,
    tropopause guess or continuation of a missing part of a profile is used.
    """
    ct = cloud_temperature.require('cloud_top_temperature', 'K')
    tp = temperature_profile.require('air_temperature', 'K')
    zp = height_profile.require('height_above_mean_sea_level', 'm')
    if ct.metadata.get('atmospheric_correction_verified') is not True:
        raise ValueError('ИК яркостная температура не подставляется вместо скорректированной температуры облака.')
    if tp.values.shape != zp.values.shape or tp.values.shape[:-1] != ct.values.shape or tp.values.shape[-1] < 2:
        raise ValueError('Нужен согласованный вертикальный профиль для каждого пикселя.')
    if tp.grid_id != ct.grid_id or zp.grid_id != ct.grid_id:
        raise ValueError('Профили и облако относятся к разным горизонтальным сеткам.')
    if (np.diff(zp.values, axis=-1)[zp.valid[..., 1:] & zp.valid[..., :-1]] <= 0).any():
        raise ValueError('Высоты профиля должны строго возрастать.')
    if tp.observed_at != zp.observed_at:
        raise ValueError('Температура и высота профиля должны иметь один срок.')
    if abs((utc(tp.observed_at)-utc(ct.observed_at)).total_seconds()) > 3*3600:
        raise ValueError('Профиль слишком далёк по времени; предел данного метода — 3 часа.')
    if (tp.values[tp.valid] <= 0).any() or (ct.values[ct.valid] <= 0).any():
        raise ValueError('Температура в K должна быть положительной.')
    valid = ct.valid & condition(opaque_single_layer, ct)
    qc = _flags(ct.valid); qc[~condition(opaque_single_layer, ct)] |= int(QC.CONDITIONS)
    output = np.full(ct.values.shape, np.nan); sigma = output.copy()
    for ind in np.ndindex(ct.values.shape):
        if not valid[ind]: continue
        t, z = tp.values[ind], zp.values[ind]
        mask = tp.valid[ind] & zp.valid[ind]
        target = ct.values[ind]
        roots = []
        flat = False
        for j in range(len(t)-1):
            if not (mask[j] and mask[j+1]): continue
            dt = t[j+1]-t[j]
            if abs(dt) < 1e-10:
                if abs(target-t[j]) < 1e-8: flat = True
                continue
            f = (target-t[j])/dt
            if 0 <= f <= 1:
                zz = z[j]+f*(z[j+1]-z[j])
                if not any(abs(zz-r[0]) < 1e-5 for r in roots):
                    roots.append((zz, abs(dt/(z[j+1]-z[j])), j, f))
        if flat or len(roots) > 1:
            qc[ind] |= int(QC.AMBIGUOUS); valid[ind] = False
        elif not roots:
            qc[ind] |= int(QC.NO_SOLUTION); valid[ind] = False
        else:
            zz, slope, j, f = roots[0]; output[ind] = zz
            if slope < 1e-5:
                qc[ind] |= int(QC.LOW_SENSITIVITY); valid[ind] = False
            elif ct.uncertainty is not None and tp.uncertainty is not None:
                st = tp.uncertainty[ind]
                sigma[ind] = np.sqrt(ct.uncertainty[ind]**2 + ((1-f)*st[j])**2 + (f*st[j+1])**2)/slope
    return result('cloud_top_height', 'opaque-profile-height-v1', output, valid, qc,
                  dependencies=[ct, tp, zp, opaque_single_layer], uncertainty=sigma,
                  assumptions=['opaque_single_layer', 'temperature_height_profile_is_causal',
                               'no_extrapolation', 'ambiguous_crossings_rejected', 'height_above_mean_sea_level_not_cloud_base'],
                  attributes={'profile_assisted': True}, **_identity(ct, available_at, data_kind))


def liquid_water_path(optical_depth: Field, effective_radius: Field, liquid_single_layer: Field,
                      *, available_at, data_kind, covariance: Field | None = None):
    """LWP=(2/3)*rho_water*tau*r_eff for a vertically homogeneous liquid layer.

    tau and r_eff must already be retrieved with a sensor-specific radiative
    transfer algorithm. This does NOT infer them from two infrared temperatures.
    """
    tau = optical_depth.require('cloud_optical_depth', '1')
    re = effective_radius.require('cloud_droplet_effective_radius', 'm')
    valid = aligned(tau, re, max_skew_seconds=60.)
    if any(f.metadata.get('retrieval_verified') is not True for f in (tau, re)):
        raise ValueError('Нужны проверенные входные оценки оптической толщины и радиуса.')
    qc = _flags(valid)
    ok = condition(liquid_single_layer, tau)
    qc[~ok] |= int(QC.CONDITIONS)
    physical = (tau.values > 0) & (tau.values <= 200) & (re.values > 0) & (re.values <= 100e-6)
    qc[~physical] |= int(QC.DOMAIN)
    factor = 2/3 * 1000.
    value = factor*tau.values*re.values
    uncertainty = None
    dependencies = [tau, re, liquid_single_layer]
    method = 'homogeneous-liquid-lwp-v1'
    error_assumption = 'conditional_independent_input_errors'
    if covariance is not None:
        covariance.require('covariance_optical_depth_effective_radius', 'm')
        valid &= aligned(tau, covariance, max_skew_seconds=60.)
        qc[~valid] |= int(QC.MISSING)
        if tau.uncertainty is None or re.uncertainty is None:
            raise ValueError('Для ковариации нужны обе стандартные неопределённости.')
        # Cov(tau, r) has units m. It is not a dimensionless correlation.
        bound = tau.uncertainty*re.uncertainty
        known = np.isfinite(bound) & np.isfinite(covariance.values)
        admissible = known & (np.abs(covariance.values) <= bound + 1e-12*np.maximum(bound, 1e-30))
        physical &= admissible
        qc[valid & ~admissible] |= int(QC.DOMAIN)
        dependencies.append(covariance)
        method = 'homogeneous-liquid-lwp-cov-v1'
        error_assumption = 'conditional_first_order_covariance_not_total_error'
    if tau.uncertainty is not None and re.uncertainty is not None:
        variance = (re.values*tau.uncertainty)**2+(tau.values*re.uncertainty)**2
        if covariance is not None:
            variance = variance + 2*tau.values*re.values*covariance.values
        # Only roundoff can make a PSD quadratic form slightly negative.
        uncertainty = factor*np.sqrt(np.maximum(variance, 0.))
    return result('cloud_liquid_water_path', method, value,
                  valid & ok & physical, qc, dependencies=dependencies, uncertainty=uncertainty,
                  assumptions=['liquid_only_not_ice_or_vapour', 'vertically_homogeneous_effective_radius',
                               error_assumption, 'optical_retrieval_required'],
                  **_identity(tau, available_at, data_kind))


def planck_radiance(temperature, wavelength_um):
    """Spectral radiance W m-2 sr-1 um-1, monochromatic Planck function."""
    c1, c2 = 1.191042972e8, 1.438776877e4
    return c1 / (wavelength_um**5 * np.expm1(c2/(wavelength_um*np.asarray(temperature))))


def surface_temperature(radiance: Field, emissivity: Field, transmittance: Field,
                        upwelling: Field, downwelling: Field, clear_land: Field,
                        *, available_at, data_kind):
    """Invert Ltoa=tau*(eps*B(Ts)+(1-eps)*Ldown)+Lup, monochromatic approximation."""
    unit = 'W m-2 sr-1 um-1'
    radiance.require('spectral_radiance', unit)
    emissivity.require('surface_emissivity', '1'); transmittance.require('atmospheric_transmittance', '1')
    upwelling.require('upwelling_spectral_radiance', unit); downwelling.require('downwelling_spectral_radiance', unit)
    wavelength = radiance.metadata.get('wavelength_um')
    if radiance.metadata.get('spectral_model') != 'monochromatic' or type(wavelength) not in (float, int) or not 8 <= wavelength <= 14:
        raise ValueError('Нужны явная монохроматическая модель и длина волны 8–14 мкм; интегральный канал требует своего оператора.')
    for field in (upwelling, downwelling):
        if field.metadata.get('wavelength_um') != wavelength:
            raise ValueError('Спектральные радиансы относятся к разным длинам волн.')
    valid = aligned(radiance, emissivity, transmittance, upwelling, downwelling)
    qc = _flags(valid); ok = condition(clear_land, radiance); qc[~ok] |= int(QC.CONDITIONS)
    e, t = emissivity.values, transmittance.values
    physical = (e > 0) & (e <= 1) & (t > 0) & (t <= 1) & (upwelling.values >= 0) & (downwelling.values >= 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        blackbody = ((radiance.values-upwelling.values)/t-(1-e)*downwelling.values)/e
        value = 1.438776877e4/(wavelength*np.log1p(1.191042972e8/(wavelength**5*blackbody)))
    physical &= blackbody > 0
    qc[~physical] |= int(QC.DOMAIN)
    return result('land_surface_temperature', 'monochromatic-surface-temperature-v1', value,
                  valid & ok & physical, qc, dependencies=[radiance, emissivity, transmittance, upwelling, downwelling, clear_land],
                  assumptions=['clear_land_only', 'known_emissivity_and_atmospheric_terms',
                               'monochromatic_approximation_not_band_response_retrieval', 'surface_not_2m_temperature'],
                  attributes={'wavelength_um': wavelength}, **_identity(radiance, available_at, data_kind))


def collocate_msumr_ikfs2(
    msumr_cloud_mask: np.ndarray,
    msumr_cth_pressure_hpa: np.ndarray,
    msumr_lat: np.ndarray,
    msumr_lon: np.ndarray,
    ikfs2_lat: np.ndarray,
    ikfs2_lon: np.ndarray,
    *,
    footprint_km: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Связывает 1-км пиксели МСУ-МР с 35-км пятнами ИКФС-2.

    Возвращает (cloud_fraction [N_obs], cth_pressure_hpa [N_obs]).
    """
    n_obs = len(ikfs2_lat)
    cloud_fraction = np.zeros(n_obs, dtype=np.float32)
    cth_pressure = np.full(n_obs, 1013.25, dtype=np.float32)

    deg_radius = (footprint_km / 2.0) / 111.0
    flat_lat = np.asarray(msumr_lat).ravel()
    flat_lon = np.asarray(msumr_lon).ravel()
    flat_mask = np.asarray(msumr_cloud_mask, dtype=bool).ravel()
    flat_cth = np.asarray(msumr_cth_pressure_hpa, dtype=np.float32).ravel()

    for i in range(n_obs):
        dlat = flat_lat - ikfs2_lat[i]
        dlon = (flat_lon - ikfs2_lon[i] + 180.0) % 360.0 - 180.0
        dlon *= np.cos(np.deg2rad(ikfs2_lat[i]))
        dist_sq = dlat**2 + dlon**2
        in_fov = dist_sq <= deg_radius**2
        if in_fov.any():
            fov_mask = flat_mask[in_fov]
            cloud_fraction[i] = float(np.mean(fov_mask))
            if fov_mask.any():
                cth_pressure[i] = float(np.median(flat_cth[in_fov & flat_mask]))
            else:
                cth_pressure[i] = 1013.25
        else:
            cloud_fraction[i] = 0.0
            cth_pressure[i] = 1013.25

    return cloud_fraction, cth_pressure


def polar_amv_winds(
    triplet_ir_bt: list[np.ndarray],
    triplet_times_sec: list[float],
    pixel_scale_km: float = 4.0,
    search_radius_px: int = 8,
    template_size_px: int = 16,
) -> dict[str, np.ndarray]:
    """Генерация полярных AMV-ветров по триплетам МСУ-ГС КА «Арктика-М».

    Вычисляет компоненты скорости u, v, смещение и индикатор качества QI.
    """
    if len(triplet_ir_bt) != 3 or len(triplet_times_sec) != 3:
        raise ValueError("Нужен триплет из ровно трех последовательных кадров.")
    dt1 = triplet_times_sec[1] - triplet_times_sec[0]
    dt2 = triplet_times_sec[2] - triplet_times_sec[1]
    if dt1 <= 0 or dt2 <= 0:
        raise ValueError("Временные отметки триплета должны строго возрастать.")

    f0, f1, f2 = triplet_ir_bt
    h, w = f1.shape
    step = template_size_px
    y_coords, x_coords = np.mgrid[template_size_px:h-template_size_px:step, template_size_px:w-template_size_px:step]
    pts_y = y_coords.ravel()
    pts_x = x_coords.ravel()
    n_pts = len(pts_y)

    u_vec = np.zeros(n_pts, dtype=np.float32)
    v_vec = np.zeros(n_pts, dtype=np.float32)
    qi_vec = np.zeros(n_pts, dtype=np.float32)

    half = template_size_px // 2
    for i in range(n_pts):
        cy, cx = pts_y[i], pts_x[i]
        t1 = f1[cy-half:cy+half, cx-half:cx+half]
        if not np.isfinite(t1).all():
            continue

        best_diff0 = 1e9
        best_dy0, best_dx0 = 0, 0
        for dy in range(-search_radius_px, search_radius_px + 1):
            for dx in range(-search_radius_px, search_radius_px + 1):
                y0, x0 = cy + dy, cx + dx
                if 0 <= y0 - half and y0 + half <= h and 0 <= x0 - half and x0 + half <= w:
                    patch0 = f0[y0-half:y0+half, x0-half:x0+half]
                    diff = np.mean(np.abs(patch0 - t1))
                    if diff < best_diff0:
                        best_diff0 = diff
                        best_dy0, best_dx0 = dy, dx

        best_diff2 = 1e9
        best_dy2, best_dx2 = 0, 0
        for dy in range(-search_radius_px, search_radius_px + 1):
            for dx in range(-search_radius_px, search_radius_px + 1):
                y2, x2 = cy + dy, cx + dx
                if 0 <= y2 - half and y2 + half <= h and 0 <= x2 - half and x2 + half <= w:
                    patch2 = f2[y2-half:y2+half, x2-half:x2+half]
                    diff = np.mean(np.abs(patch2 - t1))
                    if diff < best_diff2:
                        best_diff2 = diff
                        best_dy2, best_dx2 = dy, dx

        vx1 = (-best_dx0 * pixel_scale_km * 1000.0) / dt1
        vy1 = (-best_dy0 * pixel_scale_km * 1000.0) / dt1
        vx2 = (best_dx2 * pixel_scale_km * 1000.0) / dt2
        vy2 = (best_dy2 * pixel_scale_km * 1000.0) / dt2

        u_vec[i] = 0.5 * (vx1 + vx2)
        v_vec[i] = 0.5 * (vy1 + vy2)

        speed_diff = np.hypot(vx1 - vx2, vy1 - vy2)
        mean_speed = np.hypot(u_vec[i], v_vec[i]) + 1e-3
        qi_vec[i] = float(np.clip(1.0 - speed_diff / (0.5 * mean_speed + 5.0), 0.0, 1.0))

    return {
        "u": u_vec,
        "v": v_vec,
        "qi": qi_vec,
        "pts_y": pts_y,
        "pts_x": pts_x,
    }
