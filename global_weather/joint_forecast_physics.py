"""Memory-bounded physical diagnostics for dense joint forecasts.

Pressure velocity is diagnosed from pressure-coordinate continuity rather than
learned as an unobserved target. Meshes with exact Voronoi polygons use their
true dual edges. Mesh8 has no materialised Voronoi polygons, so it uses the
same vertex-centred barycentric control-volume family as its stored cell areas:
each shared boundary joins the two adjacent spherical-triangle centroids via
the primal-edge midpoint. Fluxes are shared with exactly opposite signs.

The imposed upper boundary is omega(100 Pa)=0. No lower mass correction is
invented because global surface-pressure tendency is not available in this
experiment. Budget quantities reported here are diagnostics on common support,
not hard conservation claims.
"""
from __future__ import annotations

import numpy as np
import torch

from .grid import EARTH_RADIUS_M, latlon
from .profile_physics import EPSILON, RD, pressure_layer_thickness
from .vertical import PRESSURE_HPA

G0 = 9.80665
CP_DRY = 1004.0
DEFAULT_EDGE_CHUNK = 32768
DEFAULT_LEVEL_CHUNK = 4


def _transport(vector, origin, destination):
    """Shortest-geodesic parallel transport of tangent Cartesian vectors."""
    return vector - (np.sum(vector * destination, axis=-1) /
                     (1 + np.sum(origin * destination, axis=-1)))[..., None] * (origin + destination)


def _triangle_centres(grid):
    centre = grid.xyz[grid.faces].sum(axis=1)
    centre /= np.linalg.norm(centre, axis=1, keepdims=True)
    return centre


def _barycentric_edge_geometry(grid):
    """Dual boundary geometry consistent with mesh8 triangle control areas."""
    faces = np.asarray(grid.faces, dtype=np.int64)
    face_count = len(faces)
    raw = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    raw.sort(axis=1)
    face_ids = np.tile(np.arange(face_count, dtype=np.int64), 3)
    key = raw[:, 0] * np.int64(grid.n_cells) + raw[:, 1]
    order = np.argsort(key, kind='stable')
    ordered_key = key[order]
    if len(ordered_key) % 2 or not np.array_equal(ordered_key[0::2], ordered_key[1::2]):
        raise ValueError('Closed triangular sphere must provide two faces per primal edge.')
    first, second = order[0::2], order[1::2]
    pairs = raw[first]
    centres = _triangle_centres(grid)
    left_face, right_face = centres[face_ids[first]], centres[face_ids[second]]
    midpoint = grid.xyz[pairs[:, 0]] + grid.xyz[pairs[:, 1]]
    midpoint /= np.linalg.norm(midpoint, axis=1, keepdims=True)
    normal = grid.xyz[pairs[:, 1]] - grid.xyz[pairs[:, 0]]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    length = EARTH_RADIUS_M * (
        np.arccos(np.clip(np.einsum('ij,ij->i', left_face, midpoint), -1, 1))
        + np.arccos(np.clip(np.einsum('ij,ij->i', midpoint, right_face), -1, 1)))
    if (len(pairs) * 2 != grid.edges.shape[1] or not np.isfinite(length).all()
            or np.any(length <= 0)):
        raise ValueError('Invalid barycentric dual-edge geometry.')
    return (pairs, midpoint.astype(np.float32), normal.astype(np.float32),
            length.astype(np.float32), 'spherical_barycentric_triangle_control_volume')


def _edge_geometry(grid):
    cached = getattr(grid, '_joint_flux_geometry', None)
    if cached is not None:
        return cached
    if not grid.has_polygon_geometry:
        cached = _barycentric_edge_geometry(grid)
        grid._joint_flux_geometry = cached
        return cached
    pairs = grid.edges[:, grid.edges[0] < grid.edges[1]].T
    midpoints, normals, lengths = [], [], []
    for left, right in pairs:
        shared = sorted(set(grid.regions[left]) & set(grid.regions[right]))
        if len(shared) != 2:
            raise ValueError('A closed Voronoi edge requires exactly two shared vertices.')
        va, vb = grid.vertices[shared]
        midpoint = va + vb
        midpoint /= np.linalg.norm(midpoint)
        normal = grid.xyz[right] - grid.xyz[left]
        normal /= np.linalg.norm(normal)
        midpoints.append(midpoint)
        normals.append(normal)
        lengths.append(EARTH_RADIUS_M * np.arccos(np.clip(np.dot(va, vb), -1, 1)))
    cached = (pairs, np.asarray(midpoints), np.asarray(normals), np.asarray(lengths),
              'spherical_voronoi')
    grid._joint_flux_geometry = cached
    return cached


def spherical_wind_divergence(u, v, mask, grid, *, edge_chunk=DEFAULT_EDGE_CHUNK,
                              level_chunk=DEFAULT_LEVEL_CHUNK):
    """Return finite-volume divergence [s^-1] without mesh8 edge explosion.

    Arrays are ``(..., cell, pressure)``. Fluxes are processed by edge and
    pressure chunks, so mesh8 never materialises a
    ``[1_966_080, 37, 3]`` velocity tensor. A cell-level divergence is supported
    only if all of its surrounding edge winds are supported. Support is derived
    from the original wind mask only; an invalid cell may invalidate its direct
    neighbours but cannot propagate farther because of edge iteration order.
    """
    raw_u, raw_v = np.asarray(u), np.asarray(v)
    output_dtype = np.float64 if raw_u.dtype.itemsize >= 8 or raw_v.dtype.itemsize >= 8 else np.float32
    u, v, mask = np.asarray(raw_u, dtype=output_dtype), np.asarray(raw_v, dtype=output_dtype), np.asarray(mask, bool)
    if (u.shape != v.shape or mask.shape != u.shape or u.ndim < 2
            or u.shape[-2] != grid.n_cells or type(edge_chunk) is not int or edge_chunk < 1
            or type(level_chunk) is not int or level_chunk < 1):
        raise ValueError('Wind arrays require matching (...,cell,pressure) shapes and positive chunks.')
    valid = mask & np.isfinite(u) & np.isfinite(v)
    pressure_count = u.shape[-1]
    prefix = u.shape[:-2]
    batch = int(np.prod(prefix)) if prefix else 1
    uf = u.reshape(batch, grid.n_cells, pressure_count)
    vf = v.reshape(batch, grid.n_cells, pressure_count)
    validf = valid.reshape(batch, grid.n_cells, pressure_count)
    divergence = np.full((batch, grid.n_cells, pressure_count), np.nan, dtype=output_dtype)
    supported = np.zeros((batch, grid.n_cells, pressure_count), dtype=bool)

    ll = np.deg2rad(latlon(grid.xyz))
    lat, lon = ll[:, 0], ll[:, 1]
    basis_dtype = np.float64 if output_dtype == np.float64 else np.float32
    xyz = np.asarray(grid.xyz, dtype=basis_dtype)
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1).astype(basis_dtype)
    north = np.stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)), axis=-1).astype(basis_dtype)
    pairs, midpoint, normal, length, _ = _edge_geometry(grid)
    midpoint, normal, length = midpoint.astype(basis_dtype, copy=False), normal.astype(basis_dtype, copy=False), length.astype(basis_dtype, copy=False)
    areas = np.asarray(grid.areas_m2, dtype=np.float64)

    for sample in range(batch):
        for level_start in range(0, pressure_count, level_chunk):
            level_stop = min(level_start + level_chunk, pressure_count)
            section = slice(level_start, level_stop)
            sums = np.zeros((grid.n_cells, level_stop-level_start), dtype=np.float64)
            base_support = validf[sample, :, section]
            local_support = base_support.copy()
            for edge_start in range(0, len(pairs), edge_chunk):
                edge_stop = min(edge_start + edge_chunk, len(pairs))
                left = pairs[edge_start:edge_stop, 0]
                right = pairs[edge_start:edge_stop, 1]
                edge_valid = base_support[left] & base_support[right]
                ul = np.where(edge_valid, uf[sample, left, section], 0)
                vl = np.where(edge_valid, vf[sample, left, section], 0)
                ur = np.where(edge_valid, uf[sample, right, section], 0)
                vr = np.where(edge_valid, vf[sample, right, section], 0)
                left_vector = ul[..., None]*east[left, None, :] + vl[..., None]*north[left, None, :]
                right_vector = ur[..., None]*east[right, None, :] + vr[..., None]*north[right, None, :]
                mid = midpoint[edge_start:edge_stop]
                left_vector = _transport(left_vector, xyz[left, None, :], mid[:, None, :])
                right_vector = _transport(right_vector, xyz[right, None, :], mid[:, None, :])
                flux = np.sum(.5*(left_vector+right_vector)*normal[edge_start:edge_stop, None, :], axis=-1)
                flux *= length[edge_start:edge_stop, None]
                flux = np.where(edge_valid, flux, 0).astype(np.float64, copy=False)
                np.add.at(sums, left, flux)
                np.add.at(sums, right, -flux)
                np.logical_and.at(local_support, left, edge_valid)
                np.logical_and.at(local_support, right, edge_valid)
            values = sums/areas[:, None]
            divergence[sample, :, section] = np.where(local_support, values, np.nan).astype(output_dtype)
            supported[sample, :, section] = local_support
    return divergence.reshape((*prefix, grid.n_cells, pressure_count)), supported.reshape((*prefix, grid.n_cells, pressure_count))


def integrate_pressure_velocity(divergence, mask, pressure_pa):
    """Integrate d(omega)/dp=-div(V) downward from omega(top)=0."""
    raw = np.asarray(divergence)
    dtype = np.float64 if raw.dtype.itemsize >= 8 else np.float32
    div, mask, pressure = np.asarray(raw, dtype=dtype), np.asarray(mask, bool), np.asarray(pressure_pa, np.float64)
    if (div.shape != mask.shape or pressure.ndim != 1 or div.shape[-1] != len(pressure)
            or len(pressure) < 2 or not np.isfinite(pressure).all() or np.any(pressure <= 0)
            or np.any(np.diff(pressure) >= 0)):
        raise ValueError('Need matching divergence/mask and descending positive pressures.')
    support = mask & np.isfinite(div)
    omega = np.full(div.shape, np.nan, dtype=dtype)
    output_mask = np.zeros_like(mask)
    output_mask[..., -1] = support[..., -1]
    omega[..., -1] = np.where(output_mask[..., -1], 0, np.nan)
    for level in range(len(pressure)-2, -1, -1):
        output_mask[..., level] = output_mask[..., level+1] & support[..., level]
        value = omega[..., level+1] - .5*(div[..., level+1]+div[..., level]) * (pressure[level]-pressure[level+1])
        omega[..., level] = np.where(output_mask[..., level], value, np.nan)
    return omega, output_mask


def _hydrostatic_diagnostics(profiles, mask, pressure):
    count, squared, maximum = 0, 0.0, 0.0
    factor = 1/EPSILON - 1
    for level in range(len(pressure)-1):
        valid = (mask[..., level, 0] & mask[..., level+1, 0]
                 & mask[..., level, 1] & mask[..., level+1, 1]
                 & mask[..., level, 4] & mask[..., level+1, 4])
        if not valid.any():
            continue
        t0, t1 = profiles[..., level, 0], profiles[..., level+1, 0]
        q0, q1 = profiles[..., level, 1], profiles[..., level+1, 1]
        tv0, tv1 = t0*(1+factor*q0), t1*(1+factor*q1)
        expected = RD*.5*(tv0+tv1)*np.log(pressure[level]/pressure[level+1])
        residual = profiles[..., level+1, 4]-profiles[..., level, 4]-expected
        values = residual[valid].astype(np.float64)
        count += values.size
        squared += float(np.dot(values, values))
        maximum = max(maximum, float(np.max(np.abs(values))))
    return {'count': count, 'rms_m2_s2': float(np.sqrt(squared/count)) if count else None,
            'max_abs_m2_s2': maximum if count else None}


def _continuity_diagnostics(divergence, divergence_mask, omega, omega_mask, pressure):
    count, squared, maximum = 0, 0.0, 0.0
    for level in range(len(pressure)-1):
        valid = (divergence_mask[..., level] & divergence_mask[..., level+1]
                 & omega_mask[..., level] & omega_mask[..., level+1])
        if not valid.any():
            continue
        residual = ((omega[..., level]-omega[..., level+1])/(pressure[level]-pressure[level+1])
                    + .5*(divergence[..., level]+divergence[..., level+1]))
        values = residual[valid].astype(np.float64)
        count += values.size
        squared += float(np.dot(values, values))
        maximum = max(maximum, float(np.max(np.abs(values))))
    return {'count': count, 'rms_s-1': float(np.sqrt(squared/count)) if count else None,
            'max_abs_s-1': maximum if count else None}


def _global_flux_closure(divergence, divergence_mask, areas):
    flat_d = divergence.reshape((-1, divergence.shape[-2], divergence.shape[-1]))
    flat_m = divergence_mask.reshape(flat_d.shape)
    values = []
    for sample in range(len(flat_d)):
        for level in range(flat_d.shape[-1]):
            if flat_m[sample, :, level].all():
                values.append(float(np.sum(flat_d[sample, :, level].astype(np.float64)*areas)))
    return {'fully_supported_slices': len(values),
            'max_abs_area_integral_m2_s-1': max(map(abs, values)) if values else None}


def _common_support_budget(profiles, mask, grid, pressure):
    """Global pressure-coordinate budget proxies on identical lead support."""
    flat = profiles.reshape((-1, grid.n_cells, len(pressure), 6))
    flat_mask = mask.reshape(flat.shape)
    if len(flat) < 1:
        return {}
    dp = pressure_layer_thickness(pressure)
    area = np.asarray(grid.areas_m2, np.float64)

    def series(required, field):
        common = np.all(np.all(flat_mask[..., required], axis=-1), axis=0)
        totals = []
        for lead in range(len(flat)):
            total = 0.0
            for level in range(len(pressure)):
                cells = common[:, level]
                if cells.any():
                    values = field(flat[lead, cells, level]).astype(np.float64)
                    total += float(np.sum(values*area[cells])) * float(dp[level]/G0)
            totals.append(total)
        reference = totals[0]
        drift = ([float((value-reference)/reference) for value in totals]
                 if np.isfinite(reference) and abs(reference) > np.finfo(float).tiny else None)
        return {'values': totals, 'relative_to_initial': drift,
                'common_support_fraction': float(common.mean())}

    water = series([1], lambda row: row[:, 1])
    dry_static = series([0, 4], lambda row: CP_DRY*row[:, 0]+row[:, 4])
    kinetic = series([2, 3], lambda row: .5*(row[:, 2]**2+row[:, 3]**2))
    return {'water_mass_kg': water, 'dry_static_energy_j': dry_static,
            'kinetic_energy_j': kinetic,
            'interpretation': 'pressure-coordinate integrals on common issued support; diagnostics, not enforced closed-system budgets'}


def apply_forecast_physics(profiles, profile_mask, grid, elevation_m, *, copy=None,
                           edge_chunk=DEFAULT_EDGE_CHUNK, level_chunk=DEFAULT_LEVEL_CHUNK):
    """Apply issuance physics and return auditable diagnostics.

    Small-grid callers retain value semantics by default. Mesh8 defaults to
    in-place processing because inference has just created disposable stacked
    arrays and duplicating the profile tensor costs more than a gigabyte.
    Explicit ``copy=True/False`` always overrides this policy.
    """
    profiles = np.asarray(profiles, dtype=np.float32)
    mask = np.asarray(profile_mask, dtype=bool)
    if copy is None:
        copy = grid.level != 8
    if type(copy) is not bool:
        raise ValueError('copy must be bool or None.')
    if copy:
        profiles, mask = profiles.copy(), mask.copy()
    elevation = np.asarray(elevation_m, np.float32)
    if (profiles.shape != mask.shape or profiles.shape[-3:] != (grid.n_cells, 37, 6)
            or elevation.shape != (grid.n_cells,) or not np.isfinite(elevation).all()):
        raise ValueError('Need profiles (...,cell,37,6), masks and finite terrain metres.')
    mask &= np.isfinite(profiles)
    pressure = np.asarray(PRESSURE_HPA, np.float64)*100
    # Product geometry mask only; training target masks remain untouched.
    above_terrain = mask[..., 4] & (profiles[..., 4] >= G0*elevation[:, None])
    wind_support = mask[..., 2] & mask[..., 3] & above_terrain
    divergence, divergence_mask = spherical_wind_divergence(
        profiles[..., 2], profiles[..., 3], wind_support, grid,
        edge_chunk=edge_chunk, level_chunk=level_chunk)
    omega, omega_mask = integrate_pressure_velocity(divergence, divergence_mask, pressure)
    profiles[..., 5] = omega
    mask[..., 5] = omega_mask
    mask &= above_terrain[..., None]
    hydrostatic = _hydrostatic_diagnostics(profiles, mask, pressure)
    continuity = _continuity_diagnostics(divergence, divergence_mask, omega, omega_mask, pressure)
    finite_q = profiles[..., 1][mask[..., 1]]
    humidity = {'count': int(finite_q.size), 'minimum': float(finite_q.min()) if finite_q.size else None,
                'maximum': float(finite_q.max()) if finite_q.size else None,
                'physical_domain_violations': int(np.count_nonzero((finite_q < 0) | (finite_q >= 1)))}
    budgets = _common_support_budget(profiles, mask, grid, pressure)
    closure = _global_flux_closure(divergence, divergence_mask, np.asarray(grid.areas_m2, np.float64))
    profiles[~mask] = np.nan
    _, _, _, _, flux_geometry = _edge_geometry(grid)
    metadata = {
        'omega_origin': 'diagnostic_pressure_coordinate_continuity',
        'omega_units': 'Pa s-1', 'omega_supervised': False,
        'omega_equation': 'div_h(V) + d(omega)/dp = 0',
        'omega_boundary': 'omega(p=100 Pa)=0 imposed; no lower boundary or mass correction',
        'omega_discretization': flux_geometry+' shared-edge flux; parallel-transported pair winds; vertical trapezoid',
        'omega_support': 'both wind components and geometric support at every adjacent cell and top-to-level path',
        'edge_chunk': edge_chunk, 'pressure_level_chunk': level_chunk,
        'issuance_mask': 'predicted geopotential >= g0 * independent land-aware point DEM height',
        'target_mask_modified': False,
        'hydrostatic_residual': hydrostatic,
        'continuity_residual': continuity,
        'specific_humidity': humidity,
        'global_flux_closure': closure,
        'budget_diagnostics': budgets,
        'limitations': ['diagnostic omega is not direct observation supervision',
                        'finite-volume interpolation has truncation error',
                        'mesh8 uses barycentric rather than exact Voronoi control boundaries',
                        'imposed top boundary and absent lower mass correction may bias omega',
                        'budget integrals use common pressure-level support and do not establish exact conservation',
                        'geopotential/geometric DEM datum approximation'],
    }
    return profiles, mask, metadata


def enforce_global_mass_and_tendency_bounds(
    surface_next: dict[str, torch.Tensor],
    surface_prev: dict[str, torch.Tensor],
    surface_init: dict[str, torch.Tensor],
    cell_area_weights: torch.Tensor | None = None,
    max_dp_3h_pa: float = 1800.0,   # макс. физическая тенденция давления за 3ч (18 гПа)
    max_dt_3h_k: float = 12.0,      # макс. скачок температуры за 3ч (12 К)
) -> dict[str, torch.Tensor]:
    """Стабилизирует шаг авторегрессии:

    1) Ограничивает экстремальные нефизичные скачки за один 3-часовой шаг.
    2) Проецирует поле P_sfc и MSLP на многообразие постоянной глобальной массы атмосферы.
    3) Гарантирует физическое условие T_d <= T.
    """
    out = dict(surface_next)

    # Веса площади ячеек икосаэдрической сетки (на Mesh-8 площади почти равны, но точный вес лучше)
    if cell_area_weights is None:
        w = torch.ones_like(out["station_pressure"])
    else:
        w = cell_area_weights.to(out["station_pressure"].device)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # 1. Ограничение 3-часовой барической и термической тенденции
    dp = torch.clamp(
        out["station_pressure"] - surface_prev["station_pressure"],
        min=-max_dp_3h_pa,
        max=max_dp_3h_pa,
    )
    p_bounded = surface_prev["station_pressure"] + dp

    # 2. Сохранение глобальной средней массы атмосферы относительно t=0
    p_init_mean = (surface_init["station_pressure"] * w).sum(dim=-1, keepdim=True)
    p_curr_mean = (p_bounded * w).sum(dim=-1, keepdim=True)
    p_conserved = p_bounded - (p_curr_mean - p_init_mean)
    out["station_pressure"] = torch.clamp(p_conserved, min=45000.0, max=108500.0)

    if "station_mean_sea_level_pressure" in out and "station_mean_sea_level_pressure" in surface_init:
        dmslp = torch.clamp(
            out["station_mean_sea_level_pressure"] - surface_prev["station_mean_sea_level_pressure"],
            min=-max_dp_3h_pa,
            max=max_dp_3h_pa,
        )
        mslp_bounded = surface_prev["station_mean_sea_level_pressure"] + dmslp
        mslp_init_mean = (surface_init["station_mean_sea_level_pressure"] * w).sum(dim=-1, keepdim=True)
        mslp_curr_mean = (mslp_bounded * w).sum(dim=-1, keepdim=True)
        out["station_mean_sea_level_pressure"] = torch.clamp(
            mslp_bounded - (mslp_curr_mean - mslp_init_mean),
            min=87000.0,
            max=108500.0,
        )

    # 3. Ограничение скачка температуры и защита дефицита точки росы (Td <= T)
    dt = torch.clamp(
        out["station_temperature"] - surface_prev["station_temperature"],
        min=-max_dt_3h_k,
        max=max_dt_3h_k,
    )
    t_next = surface_prev["station_temperature"] + dt
    out["station_temperature"] = t_next
    out["station_dew_point"] = torch.minimum(out["station_dew_point"], t_next)

    return out

