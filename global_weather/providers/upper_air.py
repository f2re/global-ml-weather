"""Normalize incomplete, mobile and off-synoptic upper-air profiles.

This module is a physical-value adapter, not a TEMP/BUFR decoder.  Decoders may
supply a registered station, a temporary platform identifier, or coordinates
only.  Every level retains its own time and position whenever available.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

from ..observation_identity import observation_identity, profile_identity

UTC = timezone.utc
G0 = 9.80665


def _utc(value, label):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and len(value) <= 64:
        try:
            result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(f'Неверное время: {label}.') from exc
    else:
        raise ValueError(f'Не указано время: {label}.')
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f'Время {label} должно иметь часовой пояс.')
    return result.astimezone(UTC)


def _number(value, label, lower, upper):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f'Неверное значение: {label}.')
    return float(value)


def saturation_vapour_pressure_pa(temperature_k):
    """Bolton-type saturation pressure over water, used only for conversion."""
    t = _number(temperature_k, 'температура', 150, 350)
    c = t - 273.15
    return 611.2 * math.exp(17.67 * c / (c + 243.5))


def specific_humidity(pressure_pa, *, temperature_k=None, dewpoint_k=None,
                      relative_humidity=None):
    pressure = _number(pressure_pa, 'давление', 1, 120000)
    if dewpoint_k is not None:
        vapour = saturation_vapour_pressure_pa(dewpoint_k)
    elif relative_humidity is not None and temperature_k is not None:
        humidity = _number(relative_humidity, 'относительная влажность', 0, 1)
        vapour = humidity * saturation_vapour_pressure_pa(temperature_k)
    else:
        raise ValueError('Для влажности нужны точка росы либо T и RH.')
    if not 0 <= vapour < pressure:
        raise ValueError('Парциальное давление пара несовместимо с давлением воздуха.')
    return .622 * vapour / (pressure - .378 * vapour)


def wind_components(speed_ms, direction_deg):
    speed = _number(speed_ms, 'скорость ветра', 0, 200)
    direction = _number(direction_deg, 'направление ветра', 0, 360)
    angle = math.radians(direction)
    return -speed * math.sin(angle), -speed * math.cos(angle)


def _availability(level, profile, observed, acquired_at, mode, latency_minutes):
    value = level.get('available_at', profile.get('available_at'))
    if value is not None:
        result = _utc(value, 'готовность наблюдения')
        basis = 'reported'
    elif mode == 'assumed_latency':
        if type(latency_minutes) is not int or not 0 <= latency_minutes <= 24 * 60:
            raise ValueError('Для предполагаемой задержки нужны целые минуты 0–1440.')
        result = observed + timedelta(minutes=latency_minutes)
        basis = 'assumed_latency_not_historical_receipt'
    elif mode == 'archive_acquisition':
        result = _utc(acquired_at, 'загрузка архива')
        basis = 'archive_acquisition_not_historical_receipt'
    else:
        raise ValueError('Нет времени готовности; выберите reported, assumed_latency или archive_acquisition.')
    if result < observed:
        raise ValueError('Готовность наблюдения раньше измерения.')
    return result, basis


def normalize_profile(profile, *, acquired_at, availability_mode='reported', latency_minutes=None):
    """Convert one arbitrary upper-air ascent to scalar observation records.

    Missing variables and missing standard levels are expected.  No vertical or
    temporal interpolation is performed here.  The model receives only actual
    measurements with explicit masks created downstream.
    """
    if not isinstance(profile, dict):
        raise ValueError('Профиль должен быть объектом.')
    provider = profile.get('provider')
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError('Нужен поставщик аэрологических данных.')
    if availability_mode not in ('reported', 'assumed_latency', 'archive_acquisition'):
        raise ValueError('Неизвестный режим доступности.')
    acquired = _utc(acquired_at, 'загрузка архива')
    actual_launch = profile.get('launch_time') is not None
    launch = _utc(profile.get('launch_time') or profile.get('nominal_time'), 'запуск или номинальный срок')
    levels = profile.get('levels')
    if not isinstance(levels, list) or not levels or len(levels) > 10000:
        raise ValueError('Нужен непустой ограниченный список уровней.')
    pid = profile_identity(profile)
    revision = profile.get('revision', 0)
    if type(revision) is not int or revision < 0:
        raise ValueError('Ревизия профиля должна быть неотрицательной.')
    base_quality = _number(profile.get('quality', 1.0), 'качество профиля', 1e-9, 1)
    launch_lat = profile.get('launch_latitude')
    launch_lon = profile.get('launch_longitude')
    if launch_lat is not None:
        launch_lat = _number(launch_lat, 'широта запуска', -90, 90)
    if launch_lon is not None:
        launch_lon = _number(launch_lon, 'долгота запуска', -180, 180)
    records = []
    for index, level in enumerate(levels):
        if not isinstance(level, dict):
            raise ValueError('Уровень профиля должен быть объектом.')
        pressure = _number(level.get('pressure_pa'), 'давление уровня', 1, 120000)
        if level.get('observed_at') is not None:
            observed = _utc(level['observed_at'], 'уровень зонда')
            time_basis = 'reported_level_time'
        elif level.get('elapsed_seconds') is not None:
            if not actual_launch:
                raise ValueError('Для времени от запуска требуется фактическое launch_time.')
            elapsed = _number(level['elapsed_seconds'], 'время от запуска', 0, 24 * 3600)
            observed = launch + timedelta(seconds=elapsed)
            time_basis = 'launch_plus_elapsed_seconds'
        else:
            observed = launch
            time_basis = 'launch_time_fallback' if actual_launch else 'nominal_time_fallback'
        if observed > acquired:
            raise ValueError('Измерение позже получения исходных данных.')
        if actual_launch and observed < launch:
            raise ValueError('Измерение уровня раньше запуска зонда.')
        latitude = level.get('latitude', launch_lat)
        longitude = level.get('longitude', launch_lon)
        if latitude is None or longitude is None:
            raise ValueError('Нужны координаты уровня либо координаты запуска.')
        latitude = _number(latitude, 'широта уровня', -90, 90)
        longitude = _number(longitude, 'долгота уровня', -180, 180)
        position_basis = ('reported_level_position' if 'latitude' in level and 'longitude' in level
                          else 'launch_position_fallback')
        available, availability_basis = _availability(
            level, profile, observed, acquired_at, availability_mode, latency_minutes)
        quality = base_quality * _number(level.get('quality', 1.0), 'качество уровня', 1e-9, 1)
        valid = level.get('valid', True)
        if type(valid) is not bool:
            raise ValueError('Маска уровня должна быть логической.')
        common = {
            'source': 'radiosonde', 'provider': provider, 'profile_id': pid,
            'provider_message_id': profile.get('provider_message_id'),
            'platform_id': profile.get('platform_id'), 'station_id': profile.get('station_id'),
            'station_registered': bool(profile.get('station_id')),
            'launch_time': launch.isoformat() if actual_launch else None,
            'nominal_time': profile.get('nominal_time'), 'observed_at': observed.isoformat(),
            'available_at': available.isoformat(), 'acquired_at': acquired.isoformat(),
            'availability_basis': availability_basis, 'time_basis': time_basis,
            'position_basis': position_basis, 'latitude': latitude, 'longitude': longitude,
            'pressure_pa': pressure, 'revision': revision, 'quality': quality, 'valid': valid,
            'sequence': index, 'sonde_type': profile.get('sonde_type'),
        }
        omitted = []
        common['omitted_derivations'] = omitted
        variables = []
        temperature = level.get('temperature_k')
        if temperature is not None:
            temperature = _number(temperature, 'температура уровня', 150, 350)
            variables.append(('temperature', temperature, 'K'))
        humidity = level.get('specific_humidity')
        if humidity is not None:
            humidity = _number(humidity, 'удельная влажность', 0, .2)
            variables.append(('specific_humidity', humidity, 'kg kg-1'))
        elif level.get('dewpoint_k') is not None or (level.get('relative_humidity') is not None and temperature is not None):
            humidity = specific_humidity(pressure, temperature_k=temperature,
                                         dewpoint_k=level.get('dewpoint_k'),
                                         relative_humidity=level.get('relative_humidity'))
            variables.append(('specific_humidity', humidity, 'kg kg-1'))
        elif level.get('relative_humidity') is not None:
            omitted.append('specific_humidity_requires_temperature')
        if level.get('u_ms') is not None or level.get('v_ms') is not None:
            for component in ('u', 'v'):
                if level.get(component + '_ms') is not None:
                    value = _number(level[component + '_ms'], 'компонента ' + component, -200, 200)
                    variables.append((component, value, 'm s-1'))
        elif level.get('wind_speed_ms') is not None or level.get('wind_direction_deg') is not None:
            if level.get('wind_speed_ms') is None or level.get('wind_direction_deg') is None:
                omitted.append('wind_components_require_speed_and_direction')
            else:
                u, v = wind_components(level['wind_speed_ms'], level['wind_direction_deg'])
                variables.extend((('u', u, 'm s-1'), ('v', v, 'm s-1')))
        if level.get('geopotential_m2_s2') is not None:
            phi = _number(level['geopotential_m2_s2'], 'геопотенциал', -10000, 1000000)
            variables.append(('geopotential', phi, 'm2 s-2'))
        elif level.get('geopotential_height_m') is not None:
            height = _number(level['geopotential_height_m'], 'геопотенциальная высота', -1000, 100000)
            variables.append(('geopotential', height * G0, 'm2 s-2'))
        if level.get('omega_pa_s') is not None:
            omega = _number(level['omega_pa_s'], 'omega', -100, 100)
            variables.append(('omega', omega, 'Pa s-1'))
        for name, value, units in variables:
            record = dict(common, variable=name, value=value, units=units)
            record['observation_id'] = observation_identity(record)
            records.append(record)
    return records
