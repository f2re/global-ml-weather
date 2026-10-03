"""Physical diagnostics and an explicitly conservative EXCHANGE operator.

Exchange alone is not a complete moisture budget or a positivity-preserving
transport solver. No claim of conservation by the whole neural model is made.
"""
import torch
from .vertical import hydrostatic_residual


def cartesian_wind(u, v, xyz):
    lon = torch.atan2(xyz[:, 1], xyz[:, 0])
    lat = torch.asin(xyz[:, 2].clamp(-1, 1))
    east = torch.stack((-lon.sin(), lon.cos(), torch.zeros_like(lon)), -1)
    north = torch.stack((-lat.sin()*lon.cos(), -lat.sin()*lon.sin(), lat.cos()), -1)
    return u[:, None]*east + v[:, None]*north


def physical_context(frame, pressure_pa, xyz, elevation, land):
    """Dimensionless state indicators plus Cartesian wind, derived without targets.

    Eight indicators: T2, q at lowest valid isobar, log(ps), potential-temperature
    difference, its validity, Coriolis/2Omega, elevation, land fraction.
    The theta difference is NOT Brunt-Vaisala N^2 or a convection diagnosis.
    """
    p = pressure_pa
    profiles, surface = frame.profiles, frame.surface
    above = p[None, :] <= surface[:, 4, None]
    lowest = above.to(torch.int64).argmax(-1)
    q = profiles[torch.arange(len(xyz), device=xyz.device), lowest, 1]
    q = torch.where(above.any(-1), q, torch.zeros_like(q))
    lo, hi = int((p-85000.).abs().argmin()), int((p-70000.).abs().argmin())
    valid = above[:, lo] & above[:, hi] & (lo != hi)
    theta = profiles[:, :, 0]*(100000./p)**(287.05/1004.)
    stability = torch.where(valid, (theta[:, hi]-theta[:, lo])/50., 0.)
    use_upper = above[:, lo]
    u = torch.where(use_upper, profiles[:, lo, 2], surface[:, 2])
    v = torch.where(use_upper, profiles[:, lo, 3], surface[:, 3])
    context = torch.stack(((surface[:, 0]-280.)/30., q/.01,
                           torch.log(surface[:, 4]/100000.), stability,
                           valid.to(surface.dtype), xyz[:, 2], elevation/5000., land), -1)
    if not torch.isfinite(context).all():
        raise FloatingPointError('Nonfinite physical context.')
    return context, cartesian_wind(u, v, xyz)


def conservative_exchange(edge_flux, edge_pairs, area_m2):
    """Positive flux travels i->j, kg/s; return tendency in kg/m2/s.

    Pass each undirected edge ONCE. Shared flux contributes -F and +F, so the
    area-weighted global sum is zero up to roundoff. Sources/sinks are separate.
    """
    if area_m2.ndim != 1 or not torch.isfinite(area_m2).all() or (area_m2 <= 0).any():
        raise ValueError('Finite positive cell areas required.')
    if edge_pairs.ndim != 2 or edge_pairs.shape[0] != 2 or edge_pairs.dtype != torch.long:
        raise ValueError('edge_pairs must be int64[2,E].')
    if edge_flux.ndim < 1 or len(edge_flux) != edge_pairs.shape[1] or not torch.isfinite(edge_flux).all():
        raise ValueError('Finite flux for every edge required.')
    if edge_pairs.numel() and ((edge_pairs < 0).any() or (edge_pairs >= len(area_m2)).any()
                              or (edge_pairs[0] == edge_pairs[1]).any()):
        raise ValueError('Invalid edge indices.')
    pairs = torch.sort(edge_pairs, dim=0).values.T
    if len(torch.unique(pairs, dim=0)) != len(pairs):
        raise ValueError('Duplicate undirected edges would count the flux twice.')
    rate = edge_flux.new_zeros((len(area_m2), *edge_flux.shape[1:]))
    rate = rate.index_add(0, edge_pairs[0], -edge_flux).index_add(0, edge_pairs[1], edge_flux)
    return rate / area_m2.reshape((-1,) + (1,)*(edge_flux.ndim-1))


def hydrostatic_penalty(profiles, pressure_pa, target_mask):
    """Masked relative hydrostatic residual, not a penalty on zero divergence.

    The caller supplies TARGET QC/terrain masks; forecast ps cannot hide errors.
    """
    if target_mask.shape != profiles.shape or target_mask.dtype != torch.bool:
        raise ValueError('A Boolean target mask for each profile variable is required.')
    valid = target_mask[..., [0, 1, 4]].all(-1)
    pairs = valid[..., 1:] & valid[..., :-1]
    safe = torch.where(target_mask, profiles, torch.zeros_like(profiles))
    if not torch.isfinite(safe).all():
        raise ValueError('Nonfinite forecast at a valid physical target.')
    residual = hydrostatic_residual(safe, pressure_pa)
    tv = safe[..., 0]*(1 + .608*safe[..., 1])
    scale = (287.05*.5*(tv[..., 1:]+tv[..., :-1]) *
             torch.log(pressure_pa[:-1]/pressure_pa[1:])).abs().clamp_min(1.)
    error = torch.where(pairs, (residual/scale).square(), torch.zeros_like(residual))
    return error.sum()/pairs.sum().clamp_min(1)
