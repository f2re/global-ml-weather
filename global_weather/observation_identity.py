"""Coordinate-native identities for fixed, mobile and anonymous observations.

A station identifier is useful provenance, but it is not a physical coordinate
and is never required for admission.  When a provider does not expose a stable
message/profile identifier, a deterministic coordinate-time identity is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re

UTC = timezone.utc
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+@=-]{0,511}")


def _utc(value) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and len(value) <= 64:
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Неверное время наблюдения.") from exc
    else:
        raise ValueError("Требуется время ISO 8601 с часовым поясом.")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Время наблюдения должно иметь часовой пояс.")
    return result.astimezone(UTC)


def _text(value, label, *, required=False, maximum=256):
    if value is None or value == "":
        if required:
            raise ValueError(f"Не указан {label}.")
        return None
    if not isinstance(value, str) or len(value) > maximum or not _ID.fullmatch(value):
        raise ValueError(f"Неверный {label}.")
    if "://" in value:
        raise ValueError("Сохраните идентификатор, а не URL с реквизитами.")
    return value


def _number(value, label, lower, upper, digits):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"Неверное значение: {label}.")
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def profile_identity(profile: dict) -> str:
    """Return a stable ascent identity without requiring a station catalogue.

    Provider profile/message identifiers are preferred.  Otherwise the launch
    time and launch position form an explicit coordinate-native fallback.
    """
    if not isinstance(profile, dict):
        raise ValueError("Профиль должен быть объектом.")
    explicit = profile.get("profile_id")
    if explicit:
        return _text(explicit, "идентификатор профиля", required=True)
    provider = _text(profile.get("provider"), "поставщик", required=True)
    message = _text(profile.get("provider_message_id"), "идентификатор сообщения")
    if message:
        payload = {"schema": "upper-air-profile-1", "provider": provider, "message": message}
    else:
        launch = _utc(profile.get("launch_time") or profile.get("nominal_time"))
        latitude = profile.get("launch_latitude")
        longitude = profile.get("launch_longitude")
        if latitude is None or longitude is None:
            levels = profile.get("levels")
            first = levels[0] if isinstance(levels, list) and levels else {}
            latitude, longitude = first.get("latitude"), first.get("longitude")
        payload = {
            "schema": "upper-air-profile-1",
            "provider": provider,
            "platform": _text(profile.get("platform_id"), "идентификатор платформы"),
            "launch_time": launch.isoformat(timespec="microseconds"),
            "launch_latitude": _number(latitude, "широта запуска", -90, 90, 5),
            "launch_longitude": _number(longitude, "долгота запуска", -180, 180, 5),
        }
    return "profile:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()


def observation_identity(record: dict) -> str:
    """Return an event identity stable across value revisions.

    The fallback intentionally excludes measured value and quality.  A revised
    value therefore retains its identity and uses the explicit revision field.
    """
    if not isinstance(record, dict):
        raise ValueError("Наблюдение должно быть объектом.")
    explicit = record.get("observation_id")
    if explicit:
        return _text(explicit, "идентификатор наблюдения", required=True)
    source = _text(record.get("source"), "источник", required=True)
    variable = _text(record.get("variable"), "величина", required=True)
    observed = _utc(record.get("observed_at"))
    profile = _text(record.get("profile_id"), "идентификатор профиля")
    message = _text(record.get("provider_message_id"), "идентификатор сообщения")
    platform = _text(record.get("platform_id") or record.get("station_id"), "идентификатор платформы")
    payload = {
        "schema": "coordinate-observation-1",
        "source": source,
        "provider": _text(record.get("provider"), "поставщик"),
        "variable": variable,
        "observed_at": observed.isoformat(timespec="microseconds"),
        "profile_id": profile,
        "provider_message_id": message,
        "platform_id": platform if not (profile or message) else None,
    }
    pressure = record.get("pressure_pa")
    if pressure not in (None, 0, 0.0):
        payload["pressure_pa"] = _number(pressure, "давление", 1, 120000, 1)
    # A provider/profile identifier survives later coordinate corrections.  In
    # its absence coordinates are the only reproducible physical identity.
    if not (profile or message):
        payload["latitude"] = _number(record.get("latitude"), "широта", -90, 90, 5)
        payload["longitude"] = _number(record.get("longitude"), "долгота", -180, 180, 5)
    sequence = record.get("sequence")
    if sequence is not None:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Неверный номер измерения.")
        payload["sequence"] = sequence
    return "coord:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()


def observation_group_identity(record: dict) -> str:
    """Group all levels of one profile for leakage-safe withholding."""
    for key in ("profile_id", "provider_message_id"):
        value = record.get(key)
        if value:
            payload = {"source": _text(record.get("source"), "источник", required=True),
                       "provider": _text(record.get("provider"), "поставщик"),
                       "kind": key, "id": _text(value, key, required=True)}
            return "group:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()
    source = _text(record.get("source"), "источник", required=True)
    platform = record.get("platform_id") or record.get("station_id")
    launch = record.get("launch_time")
    if platform and launch:
        payload = {"source": source, "platform": _text(platform, "платформа", required=True),
                   "launch_time": _utc(launch).isoformat(timespec="microseconds")}
        return "group:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return observation_identity(record)
