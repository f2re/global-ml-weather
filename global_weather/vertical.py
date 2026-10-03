"""Pressure coordinates and diagnostic constraints. No ISA profile substitution."""
import numpy as np
import torch
from torch import Tensor

# Descending pressure, 37 ERA5 pressure-output levels, NOT 137 native model levels.
PRESSURE_HPA = (1000,975,950,925,900,875,850,825,800,775,750,700,650,600,550,
                500,450,400,350,300,250,225,200,175,150,125,100,70,50,30,20,
                10,7,5,3,2,1)
PROFILE_VARIABLES = ('temperature', 'specific_humidity', 'u', 'v', 'geopotential', 'omega')
PROFILE_UNITS = ('K', 'kg kg-1', 'm s-1', 'm s-1', 'm2 s-2', 'Pa s-1')
SURFACE_VARIABLES = ('t2m', 'td2m', 'u10', 'v10', 'surface_pressure', 'mslp',
                     'precipitation_step', 'total_cloud_fraction')
SURFACE_UNITS = ('K','K','m s-1','m s-1','Pa','Pa','kg m-2','1')
PROFILE_SCALES = (30., .005, 20., 20., 100_000., .5)
SURFACE_SCALES = (30.,30.,20.,20.,10_000.,10_000.,10.,1.)


def validate_levels(levels):
    p = np.asarray(levels, dtype=float)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or not (p > 0).all() or not (np.diff(p) < 0).all():
        raise ValueError('Pressure levels must be finite, positive and strictly descending.')
    return p


def above_ground(pressure_pa: Tensor, surface_pressure_pa: Tensor):
    if not torch.isfinite(surface_pressure_pa).all() or (surface_pressure_pa <= 0).any():
        raise ValueError('Invalid surface pressure.')
    return pressure_pa <= surface_pressure_pa[..., None]


def hydrostatic_residual(profiles: Tensor, pressure_pa: Tensor):
    """Phi(top)-Phi(bottom)-Rd*mean(Tv)*log(pbottom/ptop), in m2/s2.

    A diagnostic/soft loss on adjacent isobars; it is not exact mass conservation.
    Caller must mask pairs below terrain and missing T/q/Phi targets.
    """
    t, q, phi = profiles[..., 0], profiles[..., 1], profiles[..., 4]
    tv = t * (1 + .608*q)
    return phi[..., 1:] - phi[..., :-1] - 287.05 * .5*(tv[..., 1:]+tv[..., :-1]) * torch.log(
        pressure_pa[:-1]/pressure_pa[1:])


def tangent_components(cartesian: Tensor, xyz: Tensor):
    """Convert a global Cartesian vector to local east/north; remove radial part."""
    radial = (cartesian * xyz).sum(-1, keepdim=True)
    tangent = cartesian - radial * xyz
    lon = torch.atan2(xyz[..., 1], xyz[..., 0])
    lat = torch.asin(xyz[..., 2].clamp(-1, 1))
    east = torch.stack((-lon.sin(), lon.cos(), torch.zeros_like(lon)), -1)
    north = torch.stack((-lat.sin()*lon.cos(), -lat.sin()*lon.sin(), lat.cos()), -1)
    return (tangent*east).sum(-1), (tangent*north).sum(-1)
