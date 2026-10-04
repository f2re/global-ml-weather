"""Explicit ERA5 CF-NetCDF to spherical query points. Not conservative remapping.

No downloads or radiometric retrievals. Hourly precipitation semantics must be
confirmed by the operator. Every unsupported variable stays masked, never zero.
"""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import numpy as np
from .dataset import utc, integer
from .io import atomic_json, sha256, write_arrays, MAX_ARRAY_BYTES
from ..vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS
from ..grid import build_grid, latlon

PROFILE = [('temperature', 't'), ('specific_humidity', 'q'), ('u_component_of_wind', 'u'),
           ('v_component_of_wind', 'v'), ('geopotential', 'z'), ('vertical_velocity', 'w')]
SURFACE = [('2m_temperature', 't2m'), ('2m_dewpoint_temperature', 'd2m'),
           ('10m_u_component_of_wind', 'u10'), ('10m_v_component_of_wind', 'v10'),
           ('surface_pressure', 'sp'), ('mean_sea_level_pressure', 'msl'),
           ('total_precipitation', 'tp'), ('total_cloud_cover', 'tcc')]
UNITS = {'K': {'K'}, 'kg kg-1': {'kg kg-1', 'kg kg**-1', 'kg/kg', '1'},
         'm s-1': {'m s-1', 'm s**-1', 'm/s'}, 'm2 s-2': {'m2 s-2', 'm**2 s**-2', 'm^2 s^-2'},
         'Pa s-1': {'Pa s-1', 'Pa s**-1', 'Pa/s'}, 'Pa': {'Pa'}, '1': {'1', '(0 - 1)', '(0-1)'}}


def _coordinate(ds, names):
    found = [n for n in names if n in ds.coords]
    if len(found) != 1 or ds[found[0]].ndim != 1:
        raise ValueError(f'Неоднозначная координата: {names}')
    return found[0]


class CFField:
    def __init__(self, path):
        import xarray as xr
        self.path = Path(path)
        self.ds = xr.open_dataset(path)  # CF scale/offset and missing values applied exactly once by xarray.
        try:
            lat = _coordinate(self.ds, ('latitude', 'lat')); lon = _coordinate(self.ds, ('longitude', 'lon'))
            if self.ds[lat].attrs.get('units') not in ('degrees_north', 'degree_north') or self.ds[lon].attrs.get('units') not in ('degrees_east', 'degree_east'):
                raise ValueError('Требуются географические координаты в градусах.')
            self.ds = self.ds.rename({lat: 'latitude', lon: 'longitude'})
            lats = self.ds.latitude.values; lons = self.ds.longitude.values
            if len(lats) < 2 or len(lons) < 2 or not np.isfinite(lats).all() or not np.isfinite(lons).all():
                raise ValueError('Требуется конечная прямоугольная географическая сетка.')
            if np.max(np.abs(lats)) > 90 or len(np.unique(lats)) != len(lats):
                raise ValueError('Некорректные широты.')
            longitude = np.mod(lons, 360.)
            if len(np.unique(longitude)) != len(longitude):
                raise ValueError('Повтор меридиана 0/360 требует отдельного согласования.')
            self.ds = self.ds.assign_coords(longitude=longitude).sortby('latitude').sortby('longitude')
            self.cyclic = False
            x = self.ds.longitude.values
            gaps = np.diff(x)
            if np.allclose(gaps, gaps[0], rtol=1e-5, atol=1e-7) and np.isclose(x[-1]-x[0]+gaps[0], 360.):
                self.cyclic = True
            if len(lats)*len(lons) > 2_500_000:
                raise ValueError('Исходная сетка превышает предел этого адаптера.')
        except Exception:
            self.ds.close(); raise

    def close(self):
        self.ds.close()

    def _selected(self, aliases, units, *, when=None, pressure_hpa=None, precipitation=False):
        import xarray as xr
        names = [n for n in aliases if n in self.ds.data_vars]
        if not names:
            return None
        if len(names) != 1:
            raise ValueError(f'Неоднозначная переменная: {aliases}')
        value = self.ds[names[0]]
        native = value.attrs.get('units')
        if precipitation:
            if native not in ('m', 'kg m-2', 'kg m**-2'):
                raise ValueError('Неизвестная единица количества осадков.')
        elif native not in UNITS[units]:
            raise ValueError(f'Несовместимые единицы {names[0]}: {native}')
        t = [n for n in ('time', 'valid_time') if n in value.dims]
        if when is not None:
            if len(t) != 1:
                raise ValueError('У временного поля отсутствует однозначное время.')
            stamp = np.datetime64(when.replace(tzinfo=None), 'ns')
            times = value[t[0]].values.astype('datetime64[ns]')
            if np.count_nonzero(times == stamp) > 1:
                raise ValueError('Повторный срок ERA5.')
            if stamp not in times:
                return None
            value = value.sel({t[0]: stamp})
        elif t:
            if value.sizes[t[0]] != 1:
                raise ValueError('Статическое поле содержит несколько сроков.')
            value = value.isel({t[0]: 0})
        if pressure_hpa is not None:
            axes = [n for n in ('level', 'pressure_level', 'isobaricInhPa') if n in value.dims]
            if len(axes) != 1:
                raise ValueError('У профиля нет однозначной оси давления.')
            axis = axes[0]; pressure_units = value[axis].attrs.get('units')
            if pressure_units not in ('hPa', 'millibars', 'Pa'):
                raise ValueError('У оси давления нет подтверждённых единиц.')
            query = pressure_hpa*100 if pressure_units == 'Pa' else pressure_hpa
            levels = value[axis].values
            if np.count_nonzero(levels == query) > 1:
                raise ValueError('Повторный уровень давления.')
            if query not in levels:
                return None
            value = value.sel({axis: query})
        if set(value.dims) != {'latitude', 'longitude'}:
            raise ValueError('Лишние оси: expver, ensemble или уровень требуют явного выбора.')
        return value

    def _interpolate(self, value, xyz):
        import xarray as xr
        if self.cyclic:
            lo = value.isel(longitude=[-1]).assign_coords(longitude=value.longitude.values[-1:]-360)
            hi = value.isel(longitude=[0]).assign_coords(longitude=value.longitude.values[:1]+360)
            value = xr.concat((lo, value, hi), dim='longitude')
        ll = latlon(xyz)
        result = value.interp(latitude=xr.DataArray(ll[:, 0], dims='point'),
                              longitude=xr.DataArray(np.mod(ll[:, 1], 360), dims='point'), method='linear').values
        # A regional domain crossing 0 degrees must not fill its large unobserved internal gap.
        if not self.cyclic:
            x = self.ds.longitude.values; gaps = np.diff(x)
            step = float(np.min(gaps))
            query = np.mod(ll[:, 1], 360.)
            for j in np.flatnonzero(gaps > 1.5*step):
                result[(query > x[j]) & (query < x[j+1])] = np.nan
        return np.asarray(result, np.float32)

    def values(self, aliases, units, xyz, *, when=None, pressure_hpa=None, precipitation=False):
        value = self._selected(aliases, units, when=when, pressure_hpa=pressure_hpa, precipitation=precipitation)
        if value is None:
            return np.full(len(xyz), np.nan, np.float32)
        result = self._interpolate(value, xyz)
        if precipitation and value.attrs['units'] == 'm':
            result = result*1000.
        return result

    def wind(self, u_aliases, v_aliases, xyz, *, when=None, pressure_hpa=None):
        """Interpolate Cartesian tangent vectors, then return local east/north components.

        This avoids treating changing local tangent bases as one flat coordinate
        frame. It is still a point interpolant, not conservative momentum transport.
        """
        u = self._selected(u_aliases, 'm s-1', when=when, pressure_hpa=pressure_hpa)
        v = self._selected(v_aliases, 'm s-1', when=when, pressure_hpa=pressure_hpa)
        if u is None or v is None:
            missing = np.full(len(xyz), np.nan, np.float32)
            return missing, missing.copy()
        valid = np.isfinite(u) & np.isfinite(v)
        u, v = u.where(valid), v.where(valid)
        lat, lon = np.deg2rad(self.ds.latitude), np.deg2rad(self.ds.longitude)
        cart = (-u*np.sin(lon)-v*np.sin(lat)*np.cos(lon),
                u*np.cos(lon)-v*np.sin(lat)*np.sin(lon), v*np.cos(lat))
        x, y, z = [self._interpolate(a, xyz) for a in cart]
        ll = np.deg2rad(latlon(xyz)); lat, lon = ll[:, 0], ll[:, 1]
        east = -x*np.sin(lon)+y*np.cos(lon)
        north = -x*np.sin(lat)*np.cos(lon)-y*np.sin(lat)*np.sin(lon)+z*np.cos(lat)
        return east.astype(np.float32), north.astype(np.float32)


def prepare_targets(pressure_path, surface_path, output, *, issue_time, mesh_level, horizon_hours=72,
                    step_hours=3, confirm_utc=False, precipitation_kind=None):
    if not confirm_utc:
        raise ValueError('Подтвердите, что временная ось исходных ERA5 задана в UTC.')
    if precipitation_kind not in (None, 'hourly_increment'):
        raise ValueError('Неподдерживаемое накопление осадков.')
    integer(mesh_level, 0, 6); integer(horizon_hours, 1, 72)
    if step_hours not in (1, 3, 6) or horizon_hours % step_hours:
        raise ValueError('Несовместимые шаг и горизонт.')
    grid = build_grid(mesh_level); issue = utc(issue_time)
    leads = list(range(0, horizon_hours+1, step_hours)); shape = (len(leads), grid.n_cells)
    expected = len(leads)*grid.n_cells*(37*6+8)*5
    if expected > MAX_ARRAY_BYTES:
        raise ValueError('Целевой массив превышает предел. Используйте меньшую сетку для этого адаптера.')
    profiles = np.full((*shape, 37, 6), np.nan, np.float32)
    surface = np.full((*shape, 8), np.nan, np.float32)
    paths = (Path(pressure_path), Path(surface_path))
    hashes = {str(p): sha256(p) for p in paths}
    pf = CFField(pressure_path)
    try:
        sf = CFField(surface_path)
        try:
            for i, lead in enumerate(leads):
                when = issue+timedelta(hours=lead)
                for j, pressure in enumerate(PRESSURE_HPA):
                    for k, aliases in enumerate(PROFILE):
                        if k not in (2, 3):
                            profiles[i, :, j, k] = pf.values(aliases, PROFILE_UNITS[k], grid.xyz, when=when, pressure_hpa=pressure)
                    profiles[i, :, j, 2], profiles[i, :, j, 3] = pf.wind(
                        PROFILE[2], PROFILE[3], grid.xyz, when=when, pressure_hpa=pressure)
                for k, aliases in enumerate(SURFACE):
                    if k in (2, 3):
                        continue
                    if k != 6:
                        surface[i, :, k] = sf.values(aliases, SURFACE_UNITS[k], grid.xyz, when=when)
                    elif i and precipitation_kind == 'hourly_increment':
                        values = [sf.values(aliases, 'kg m-2', grid.xyz,
                                            when=when-timedelta(hours=h), precipitation=True) for h in range(step_hours)]
                        # Missing an hour invalidates the interval; nansum would invent absent precipitation.
                        surface[i, :, k] = np.sum(values, axis=0)
                surface[i, :, 2], surface[i, :, 3] = sf.wind(SURFACE[2], SURFACE[3], grid.xyz, when=when)
            pm = np.isfinite(profiles); sm = np.isfinite(surface)
            pm &= (np.array(PRESSURE_HPA)[None, None, :, None]*100 <= surface[:, :, 4, None, None])
            profiles[~pm] = np.nan
        finally:
            sf.close()
    finally:
        pf.close()
    if hashes != {str(p): sha256(p) for p in paths}:
        raise ValueError('Источник изменился во время подготовки.')
    output = Path(output)
    write_arrays(output, profiles=profiles, profile_mask=pm, surface=surface, surface_mask=sm,
                 pressure_hpa=np.array(PRESSURE_HPA), lead_hours=np.array(leads), issue_time=issue.isoformat(),
                 grid_fingerprint=grid.fingerprint, profile_variables=np.array(PROFILE_VARIABLES),
                 profile_units=np.array(PROFILE_UNITS), surface_variables=np.array(SURFACE_VARIABLES),
                 surface_units=np.array(SURFACE_UNITS))
    report = {'schema': 'era5-target-provenance-1', 'artifact_sha256': sha256(output), 'source_sha256': hashes,
              'operator': 'bilinear_at_spherical_cell_centres; NOT conservative_cell_average',
              'wind_operator': 'Cartesian tangent interpolation and ENU projection; not conservative momentum transport',
              'time_axis': 'UTC_confirmed_by_operator', 'precipitation_kind': precipitation_kind,
              'step_hours': step_hours, 'issue_time': issue.isoformat(), 'horizon_hours': horizon_hours,
              'extrapolation': False, 'scale_applied': 'xarray_CF_once', 'source_truth_verified': False}
    atomic_json(output.with_suffix('.provenance.json'), report)
    return report


def prepare_static(source, output, *, mesh_level):
    grid = build_grid(integer(mesh_level, 0, 6)); field = CFField(source)
    original = sha256(Path(source))
    try:
        height = field.values(('geopotential', 'z'), 'm2 s-2', grid.xyz)/9.80665
        land = field.values(('land_sea_mask', 'lsm'), '1', grid.xyz)
    finally:
        field.close()
    if not np.isfinite(height).all() or not np.isfinite(land).all() or ((land < 0) | (land > 1)).any():
        raise ValueError('Для статических полей требуется полное глобальное покрытие.')
    if sha256(Path(source)) != original:
        raise ValueError('Источник изменён при чтении.')
    write_arrays(output, elevation_m=height, land_fraction=land, surface_units=np.array(['m', '1']),
                 grid_fingerprint=grid.fingerprint)
    report = {'grid_fingerprint': grid.fingerprint, 'mesh_level': mesh_level, 'source_sha256': original, 'operator': 'bilinear_at_cell_centres_not_area_average',
              'terrain': 'surface_geopotential_divided_by_g0', 'g0_m_s2': 9.80665,
              'dynamic_surface_state_included': False, 'artifact_sha256': sha256(Path(output))}
    atomic_json(Path(output).with_suffix('.provenance.json'), report)
    return report
