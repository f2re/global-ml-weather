"""Admission of complete supervised horizons. Does not certify source truth."""
import numpy as np
from ..vertical import PRESSURE_HPA, PROFILE_VARIABLES, SURFACE_VARIABLES


def check_coverage(data, minimum=0.99):
    if type(minimum) not in (int, float) or not 0 < minimum <= 1:
        raise ValueError("Порог покрытия должен быть в (0, 1].")
    profiles, pm, surface, sm = (data[k] for k in ("profiles", "profile_mask", "surface", "surface_mask"))
    ps = surface[:, :, 4]
    above = np.array(PRESSURE_HPA)[None, None, :] * 100 <= ps[:, :, None]
    rows, missing = [], []
    for i, lead in enumerate(data["lead_hours"]):
        # Missing ps is not permission to shrink the atmospheric domain.
        for k, name in enumerate(SURFACE_VARIABLES):
            if i == 0 and k == 6:
                continue
            fraction = float((sm[i, :, k] & np.isfinite(surface[i, :, k])).mean())
            rows.append(dict(lead_hours=int(lead), variable=name, fraction=fraction))
            if fraction < minimum:
                missing.append(f"+{lead}ч {name}: {fraction:.1%}")
        for k, name in enumerate(PROFILE_VARIABLES):
            for j, level in enumerate(PRESSURE_HPA):
                eligible = above[i, :, j] & sm[i, :, 4]
                count = int(eligible.sum())
                fraction = float((pm[i, eligible, j, k] & np.isfinite(profiles[i, eligible, j, k])).mean()) if count else None
                rows.append(dict(lead_hours=int(lead), variable=name, pressure_hpa=level,
                                 eligible_cells=count, fraction=fraction))
                if fraction is not None and fraction < minimum:
                    missing.append(f"+{lead}ч {name}/{level}гПа: {fraction:.1%}")
    if missing:
        raise ValueError("Неполные обязательные цели: " + "; ".join(missing[:12]))
    return rows
