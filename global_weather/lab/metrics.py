"""Physical diagnostics are distinct from skill against independent observations."""
import numpy as np


def paired_scores(prediction, target, mask, weights, baseline=None):
    p, y, m = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if p.shape != y.shape or m.shape != y.shape or m.dtype != bool:
        raise ValueError('Нужны совпадающие размеры и логическая маска.')
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), p.shape)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Некорректные веса площади.')
    use = m & (w > 0)
    if (use & (~np.isfinite(p) | ~np.isfinite(y))).any():
        raise ValueError('Неконечное значение на проверяемой цели.')
    if not use.any():
        return {'count': 0, 'mae': None, 'rmse': None, 'bias': None, 'baseline_rmse': None, 'skill_rmse': None}
    ww = w[use]; error = p[use].astype(np.float64) - y[use]
    rmse = float(np.sqrt(np.average(error**2, weights=ww)))
    out = dict(count=int(use.sum()), mae=float(np.average(abs(error), weights=ww)),
               rmse=rmse, bias=float(np.average(error, weights=ww)), baseline_rmse=None, skill_rmse=None)
    if baseline is not None:
        b = np.asarray(baseline)
        if b.shape != p.shape or not np.isfinite(b[use]).all():
            raise ValueError('Контрольный прогноз должен покрывать те же цели.')
        ref = float(np.sqrt(np.average((b[use] - y[use])**2, weights=ww)))
        out.update(baseline_rmse=ref, skill_rmse=1-rmse/ref if ref > 0 else None)
    return out


def physical_diagnostics(profiles, surface, pressure_pa, area_m2):
    """Raw-output bounds plus masked hydrostatic residual. Not conservation proof."""
    p, s, pressure, area = map(np.asarray, (profiles, surface, pressure_pa, area_m2))
    if p.ndim != 3 or p.shape[-1] != 6 or s.shape != (len(p), 8) or p.shape[1] != len(pressure):
        raise ValueError('Неверные размеры физических полей.')
    if area.shape != (len(p),) or not np.isfinite(area).all() or (area <= 0).any():
        raise ValueError('Нужны положительные площади ячеек.')
    if not np.isfinite(pressure).all() or (pressure <= 0).any() or not (np.diff(pressure) < 0).all():
        raise ValueError('Давления должны убывать.')
    finite = bool(np.isfinite(p).all() and np.isfinite(s).all())
    result = dict(finite=finite, negative_humidity=int((p[..., 1] < 0).sum()),
                  negative_precipitation=int((s[:, 6] < 0).sum()),
                  dewpoint_above_temperature=int((s[:, 1] > s[:, 0]+1e-5).sum()),
                  cloud_out_of_range=int(((s[:, 7] < 0) | (s[:, 7] > 1)).sum()),
                  nonpositive_pressure=int((s[:, 4:6] <= 0).sum()),
                  nonpositive_temperature=int((p[..., 0] <= 0).sum() + (s[:, :2] <= 0).sum()),
                  hydrostatic_rmse_m2_s2=None, hydrostatic_pairs=0,
                  water_budget='not_implemented', energy_budget='not_implemented',
                  meteorological_skill='not_measured')
    if finite:
        above = pressure[None, :] <= s[:, 4, None]
        valid = above[:, 1:] & above[:, :-1]
        tv = p[..., 0]*(1+.608*p[..., 1])
        residual = np.diff(p[..., 4], axis=1) - 287.05*.5*(tv[:, 1:]+tv[:, :-1])*np.log(pressure[:-1]/pressure[1:])
        weights = np.broadcast_to(area[:, None], valid.shape)[valid]
        result['hydrostatic_pairs'] = int(valid.sum())
        if valid.any():
            result['hydrostatic_rmse_m2_s2'] = float(np.sqrt(np.average(residual[valid]**2, weights=weights)))
    return result
