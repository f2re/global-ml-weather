"""R11 thermodynamics, hydrostatics, mass and omega on analytic cases."""
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from global_weather.assimilation import thermo
from global_weather.vertical import PRESSURE_HPA

P = torch.tensor(PRESSURE_HPA, dtype=torch.float64) * 100.


def test_isothermal_column_is_exact_and_anchored_to_terrain():
    temperature = torch.full((3, 37), 250., dtype=torch.float64)
    surface_pressure = torch.tensor([101000., 85000., 60000.], dtype=torch.float64)
    surface_phi = thermo.G0 * torch.tensor([0., 1500., 4000.], dtype=torch.float64)
    phi = thermo.geopotential_from_surface(temperature, torch.zeros_like(temperature), P, surface_pressure, surface_phi)
    exact = surface_phi[:, None] + thermo.RD * 250. * torch.log(surface_pressure[:, None] / P)
    torch.testing.assert_close(phi, exact, rtol=0, atol=1e-8)


def test_geopotential_is_continuous_when_surface_crosses_a_level():
    temperature = torch.linspace(295., 200., 37, dtype=torch.float64)[None]
    humidity = torch.full_like(temperature, 1e-3)
    values = [thermo.geopotential_from_surface(temperature, humidity, P, torch.tensor([s], dtype=torch.float64),
                                               torch.zeros(1, dtype=torch.float64))[0, 20]
              for s in (97500. - 1e-4, 97500., 97500. + 1e-4)]
    assert abs(float(values[0] - values[1])) < 1e-3 and abs(float(values[2] - values[1])) < 1e-3


def test_saturation_bounds_and_reference_value():
    qs = thermo.saturation_specific_humidity(torch.tensor(300., dtype=torch.float64), torch.tensor(1e5, dtype=torch.float64))
    assert float(qs) == pytest.approx(0.02230, rel=2e-3)
    stratopause = thermo.saturation_specific_humidity(torch.tensor(270., dtype=torch.float64),
                                                      torch.tensor(100., dtype=torch.float64))
    assert 0 < float(stratopause) < 0.39
    logit = torch.linspace(-30, 30, 101, dtype=torch.float64)
    q = thermo.specific_humidity_from_logit(logit, torch.full_like(logit, 280.), torch.full_like(logit, 8e4))
    assert float((q / thermo.saturation_specific_humidity(torch.tensor(280.), torch.tensor(8e4))).max()) <= 1.05 + 1e-9


def test_dew_point_inverts_saturation_within_tenth_kelvin():
    for t in (243.15, 273.15, 303.15):
        temperature = torch.tensor(t, dtype=torch.float64)
        pressure = torch.tensor(9e4, dtype=torch.float64)
        q = thermo.saturation_specific_humidity(temperature, pressure)
        assert float(thermo.dew_point_from_humidity(q, pressure)) == pytest.approx(t, abs=0.15)


def test_reductions_and_station_pressure_follow_hypsometry():
    ps, tv, z = torch.tensor(90000.), torch.tensor(280.), torch.tensor(1000.)
    assert float(thermo.mean_sea_level_pressure(ps, tv, torch.tensor(0.))) == pytest.approx(90000.)
    mslp = float(thermo.mean_sea_level_pressure(ps, tv, z))
    assert mslp == pytest.approx(90000. * np.exp(9.80665 * 1000. / (287.05 * (280. + 3.25))), rel=1e-6)
    same = thermo.pressure_at_height(ps, tv, z, z, torch.tensor(0.0065))
    assert float(same) == pytest.approx(90000.)
    higher = thermo.pressure_at_height(ps, tv, z, z + 100., torch.tensor(0.))
    assert float(higher) == pytest.approx(90000. * np.exp(-9.80665 * 100. / (287.05 * 280.)), rel=1e-6)


def test_dry_mass_corrector_restores_target_exactly():
    rng = np.random.default_rng(0)
    area = torch.tensor(rng.uniform(1e12, 2e12, 5))
    humidity = torch.tensor(rng.uniform(0, 1e-3, (5, 37)))
    ps = torch.tensor(rng.uniform(6e4, 1.02e5, 5))
    target = thermo.global_dry_air_mass(ps, humidity, P, area)
    moved = ps + torch.tensor(rng.normal(0, 80, 5))
    corrected = thermo.correct_dry_air_mass(moved, humidity, P, area, target)
    assert abs(float(thermo.global_dry_air_mass(corrected, humidity, P, area) / target - 1)) < 1e-12
    # Uniform offset: horizontal gradients of p_s are unchanged.
    torch.testing.assert_close(corrected - moved, (corrected - moved)[:1].expand(5))


def test_omega_satisfies_both_boundaries_and_constant_divergence_case():
    divergence = torch.full((2, 37), 1e-5, dtype=torch.float64)
    ps = torch.tensor([100000., 70000.], dtype=torch.float64)
    surface = torch.tensor([-0.3, 0.2], dtype=torch.float64)
    omega, raw = thermo.omega_with_lower_boundary(divergence, P, ps, surface)
    torch.testing.assert_close(raw[0, 0], torch.tensor(-1e-5 * (100000. - 100.), dtype=torch.float64))
    assert torch.all(omega[:, -1] == 0)
    torch.testing.assert_close(omega[0, 0], surface[0])
    j = int((P > ps[1]).sum())
    interpolated = thermo.interpolate_to_surface(omega, P, ps)
    torch.testing.assert_close(interpolated[1], surface[1], atol=1e-3, rtol=0)
    assert j > 0


def test_toa_insolation_at_solstice():
    xyz = torch.tensor([[0., 0., 1.], [0., 0., -1.], [1., 0., 0.]], dtype=torch.float64)
    flux = thermo.toa_insolation(xyz, datetime(2021, 6, 21, 12, tzinfo=timezone.utc))
    assert float(flux[0]) == pytest.approx(1361 * 0.968 * np.sin(np.radians(23.44)), rel=0.02)
    assert float(flux[1]) == 0
    assert float(flux[2]) == pytest.approx(1361 * 0.968 * np.cos(np.radians(23.44)), rel=0.02)
