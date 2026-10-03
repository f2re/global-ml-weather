"""Availability-aware observation selection and a physical-units admission gate."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Time must be timezone-aware; store observations in UTC.")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ObservationEvent:
    """One version of a physical observation, not a forecast or interpolated value.

    observation_id is stable across revisions. available_at must include decoding
    and QC latency, not merely the arrival of the first packet.
    """
    observation_id: str
    source: str
    observed_at: datetime
    available_at: datetime
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.observation_id or not self.source or self.revision < 0:
            raise ValueError("Observation ID/source required; revision must be nonnegative.")
        if _utc(self.available_at) < _utc(self.observed_at):
            raise ValueError("Availability cannot precede the actual measurement.")


def select_as_issued(events: Iterable[ObservationEvent], issue_time: datetime,
                     history_hours: float = 12.0) -> list[ObservationEvent]:
    """Latest available revisions within (issue-history, issue], never future data.

    Historical latency must come from archive logs. This function cannot recover
    an unknown historical availability time from a filename.
    """
    if history_hours <= 0:
        raise ValueError("history_hours must be positive.")
    issue = _utc(issue_time)
    lower = issue - timedelta(hours=history_hours)
    chosen: dict[tuple[str, str], ObservationEvent] = {}
    for event in events:
        if not (lower < _utc(event.observed_at) <= issue
                and _utc(event.available_at) <= issue):
            continue
        key = (event.source, event.observation_id)
        old = chosen.get(key)
        if old is None or (event.revision, _utc(event.available_at)) > (
                old.revision, _utc(old.available_at)):
            chosen[key] = event
    return sorted(chosen.values(), key=lambda x: (_utc(x.observed_at), x.source, x.observation_id))


@dataclass(frozen=True)
class RadiometrySpec:
    """Metadata admission gate, NOT a radiometric calibration implementation.

    Map telemetry channel indices to documented physical channels upstream.
    Files satisfying this gate still need numerical QC and independent checks.
    """
    instrument: str
    quantity: str
    units: str
    physical_channel_ids: tuple[str, ...]
    calibration_id: str | None
    channel_mapping_verified: bool
    layout: str = "native"

    def assert_physical_ready(self) -> None:
        if self.quantity == "raw_counts":
            raise ValueError("Raw counts are not brightness temperatures or reflectance.")
        accepted_units = {
            "brightness_temperature": {"K"},
            "reflectance": {"1"},
            "spectral_radiance": {"W m-2 sr-1 um-1", "mW m-2 sr-1 (cm-1)-1"},
        }
        if self.quantity not in accepted_units or self.units not in accepted_units[self.quantity]:
            raise ValueError("Unsupported physical quantity or units; convert explicitly upstream.")
        if not self.calibration_id or not self.channel_mapping_verified:
            raise ValueError("Verified channel mapping and calibration provenance are required.")
        ids = self.physical_channel_ids
        if not self.instrument or not ids or len(set(ids)) != len(ids) or any(not x for x in ids):
            raise ValueError("Instrument and unique, nonempty physical channel IDs are required.")
