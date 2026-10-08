"""Explicit physical-value records -> sparse spherical token links.

JSONL is an interchange contract, not a SatDump raster decoder or a microwave
antenna operator. Footprints bigger than a cell are rejected by the prototype.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .contracts import ObservationEvent, RadiometrySpec, select_as_issued
from .grid import SphereGrid, unit_xyz
from .vertical import validate_levels
from .observation_identity import observation_identity

SOURCES = ('station', 'radiosonde', 'electro_l', 'arktika_m', 'meteor_msu_mr', 'meteor_mtvza')
SATELLITES = frozenset(SOURCES[2:])


@dataclass(frozen=True)
class Variable:
    units: str
    offset: float
    scale: float
    vertical: str  # pressure, surface, column
    source: str | None = None
    platform: str | None = None
    channel_id: str | None = None
    product: str | None = None
    method: str | None = None
    history_hours: int = 12
    product_depth_m: float | None = None

    def __post_init__(self):
        if not self.units or not np.isfinite([self.offset,self.scale]).all() or self.scale <= 0:
            raise ValueError('Invalid normalization/units.')
        if self.vertical not in ('pressure','surface','column'):
            raise ValueError('Unknown vertical placement.')
        if self.product is not None:
            from .products.ingest import check_variable
            check_variable(self)
        elif self.product_depth_m is not None or self.method is not None or type(self.history_hours) is not int or self.history_hours != 12:
            raise ValueError('Raw observations retain a 12-hour history and no retrieval method.')
        if self.product is None and self.vertical == 'column' and (self.source not in SATELLITES or not self.platform or not self.channel_id):
            raise ValueError('A column radiance requires explicit source/platform/channel binding.')


# Engineering scales only; real runs must persist training-only normalization.
DEFAULT_VARIABLES = {
    'temperature': Variable('K',250.,30.,'pressure'),
    'specific_humidity': Variable('kg kg-1',0.,.005,'pressure'),
    'u': Variable('m s-1',0.,20.,'pressure'),
    'v': Variable('m s-1',0.,20.,'pressure'),
    'geopotential': Variable('m2 s-2',0.,100_000.,'pressure'),
    't2m': Variable('K',273.15,30.,'surface'),
    'td2m': Variable('K',273.15,30.,'surface'),
    'u10': Variable('m s-1',0.,20.,'surface'),
    'v10': Variable('m s-1',0.,20.,'surface'),
    'surface_pressure': Variable('Pa',100_000.,10_000.,'surface'),
}


@dataclass
class PackedObservations:
    features: torch.Tensor  # [M,12], normalized value/time/vertical/geometry metadata
    cells: torch.Tensor
    levels: torch.Tensor  # -1 column radiance; last index is surface
    slots: torch.Tensor   # 0 oldest ... 11 newest
    sources: torch.Tensor
    variables: torch.Tensor
    weights: torch.Tensor
    issue_time: datetime
    grid_fingerprint: str
    pressure_pa: torch.Tensor
    vocabulary: tuple[str, ...]
    schema_fingerprint: str
    accepted_records: int
    rejected: dict[str, int]
    normalization_fingerprint: str = ""
    evidence_ids: tuple[str, ...] = ()  # one identity per placed token, stable across releases

    def coverage(self, n_cells):
        """Direct evidence at issue time, not confidence or future coverage."""
        size = n_cells*len(SOURCES)
        age = self.features.new_full((size,), torch.inf)
        if len(self.cells):
            indices = self.cells*len(SOURCES)+self.sources
            age.scatter_reduce_(0,indices,self.features[:,1]*12.,reduce='amin',include_self=True)
        age = age.reshape(n_cells,len(SOURCES))
        return torch.isfinite(age),age

    def to(self, device):
        fields = {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in vars(self).items()}
        return PackedObservations(**fields)


def utc(text):
    value = datetime.fromisoformat(text.replace('Z','+00:00')) if isinstance(text,str) else text
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Timezone-aware timestamps are required.')
    return value.astimezone(timezone.utc)


def read_variables(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict) or not data:
        raise ValueError('Variable manifest must be a nonempty mapping.')
    return {name:Variable(**spec) for name,spec in data.items()}


def read_jsonl(path):
    records = []
    with Path(path).open(encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            if line.strip():
                try:
                    item = json.loads(line)
                    if not isinstance(item,dict):
                        raise ValueError('Expected an object.')
                    records.append(item)
                except (ValueError,TypeError) as exc:
                    raise ValueError(f'{path}:{number}: {exc}') from exc
    return records


def pack_observations(records, grid: SphereGrid, pressure_pa, issue_time,
                      variables=None, *, normalization=None):
    variables = DEFAULT_VARIABLES if variables is None else variables
    p = validate_levels(pressure_pa)
    if normalization is not None:
        for name,var in variables.items():
            stat=normalization.get(name,var.units)
            if bool(stat.pressure_pa) != (var.vertical == 'pressure'):
                raise ValueError('Normalisation vertical meaning differs from the variable registry.')
    vocab = tuple(sorted(variables))
    issue = utc(issue_time)
    selected = {}
    rejected = {}
    def reject(reason):
        rejected[reason] = rejected.get(reason,0)+1
    for original in records:
        try:
            if not isinstance(original, dict):
                raise ValueError('invalid_contract')
            rec = dict(original)
            rec['observation_id'] = observation_identity(rec)
            source, name = rec['source'],rec['variable']
            if source not in SOURCES or name not in variables:
                raise ValueError('unknown_source_or_variable')
            # A stable ID identifies one scalar/level/channel across revisions.
            event = ObservationEvent(str(rec['observation_id']), source, utc(rec['observed_at']),
                                     utc(rec['available_at']), int(rec.get('revision',0)))
            key = (event.source,event.observation_id)
            if not select_as_issued([event],issue,history_hours=variables[name].history_hours):
                reject('not_available_in_12h_window'); continue
            old = selected.get(key)
            if old and (event.revision,event.available_at) < (old[0].revision,old[0].available_at):
                continue
            if old and (event.revision,event.available_at) == (old[0].revision,old[0].available_at) and rec != old[1]:
                raise ValueError('Conflicting duplicate revision; fix the archive.')
            selected[key] = (event,rec)
        except (KeyError,TypeError,ValueError) as exc:
            if 'Conflicting duplicate' in str(exc):
                raise
            reject('invalid_contract')
    feats, cells, levels, slots, sources, var_ids, weights = [], [], [], [], [], [], []
    accepted = 0
    evidence_ids = []
    for event,rec in selected.values():
        try:
            var = variables[rec['variable']]
            if not isinstance(rec.get('valid',True),bool):
                raise ValueError('valid_must_be_boolean')
            value = float(rec['value'])
            latitude,longitude = float(rec['latitude']),float(rec['longitude'])
            quality = float(rec.get('quality',1.))
            if not rec.get('valid',True) or not np.isfinite([value,latitude,longitude,quality]).all() or not 0 < quality <= 1:
                raise ValueError('invalid_value_or_quality')
            if rec['units'] != var.units:
                raise ValueError('unit_mismatch')
            c = int(grid.locate(latitude,longitude))
            footprint = float(rec.get('footprint_km',0.))
            if not np.isfinite(footprint) or footprint < 0:
                raise ValueError('invalid_footprint')
            if var.product is not None:
                from .products.ingest import check_record
                check_record(rec,var,issue)
                if 'view_zenith_deg' not in rec or footprint <= 0:
                    raise ValueError('derived_product_geometry_missing')
                if footprint > np.sqrt(grid.areas_m2[c])/1000:
                    raise ValueError('footprint_operator_required')
            elif rec.get('derivation') is not None:
                raise ValueError('A derived product must not impersonate a raw channel.')
            elif event.source in SATELLITES:
                if var.source != event.source or var.platform != rec.get('platform') or var.channel_id != rec.get('channel_id'):
                    raise ValueError('sensor_binding_mismatch')
                if 'view_zenith_deg' not in rec:
                    raise ValueError('satellite_view_geometry_missing')
                spec = RadiometrySpec(**rec['radiometry'])
                spec.assert_physical_ready()
                if spec.quantity == 'reflectance':
                    solar = float(rec['solar_zenith_deg'])
                    if not np.isfinite(solar) or not 0 <= solar < 90:
                        raise ValueError('no_daylight_reflectance')
                if rec.get('channel_id') not in spec.physical_channel_ids or spec.units != var.units:
                    raise ValueError('channel_mismatch')
                if var.vertical != 'column' or footprint <= 0:
                    raise ValueError('satellite_geometry_missing')
                if footprint > np.sqrt(grid.areas_m2[c])/1000:
                    raise ValueError('footprint_operator_required')
            elevation = float(rec.get('elevation_m',0.))
            zenith = float(rec.get('view_zenith_deg',0.))
            if not np.isfinite([elevation,zenith]).all() or not 0 <= zenith < 90:
                raise ValueError('invalid_geometry')
            pressure = float(rec.get('pressure_pa',0.))
            if not np.isfinite(pressure):
                raise ValueError('invalid_pressure_metadata')
            age = (issue-event.observed_at).total_seconds()/3600
            if var.vertical == 'pressure':
                if not np.isfinite(pressure) or not p[-1] <= pressure <= p[0]:
                    raise ValueError('pressure_outside_supported_levels')
                # Two bracketing levels in log(p), no vertical extrapolation.
                x = -np.log(p); target = -np.log(pressure)
                j = int(np.clip(np.searchsorted(x,target),1,len(p)-1))
                alpha = float((target-x[j-1])/(x[j]-x[j-1]))
                links = [(j-1,1-alpha),(j,alpha)]
            else:
                links = [(len(p) if var.vertical == 'surface' else -1,1.)]
            normalized = (value-var.offset)/var.scale
            if normalization is not None:
                normalized = float(normalization.normalise(
                    rec['variable'], value, var.units,
                    pressure if var.vertical == 'pressure' else None))
            feature = [normalized, age/12.,
                       np.log(pressure/100_000.)/7 if pressure > 0 else 0.,
                       float(var.vertical == 'pressure'), elevation/5000.,
                       float('elevation_m' in rec), footprint/100., np.cos(np.deg2rad(zenith)),
                       *unit_xyz(latitude,longitude).tolist(),quality]
            for level,weight in links:
                if weight <= 0: continue
                feats.append(feature); cells.append(c); levels.append(level)
                slots.append(11-min(11,int(np.floor(age))))  # Keep real age; slow context uses the oldest update.
                sources.append(SOURCES.index(event.source)); var_ids.append(vocab.index(rec['variable']))
                weights.append(weight*quality)
                identity = [event.source,event.observation_id,event.revision,event.available_at.isoformat()]
                if var.product is not None:
                    identity.append(dict(product=var.product,history_hours=var.history_hours))
                evidence_ids.append(json.dumps(identity,separators=(',',':')))
            accepted += 1
        except (KeyError,TypeError,ValueError) as exc:
            reject(str(exc) if str(exc) in ('unit_mismatch','footprint_operator_required',
                                            'pressure_outside_supported_levels') else 'invalid_physical_record')
    schema = json.dumps({'version':1,'variables':{k:asdict(variables[k]) for k in vocab},
                         'pressure_pa':p.tolist(),
                         'normalization':normalization.fingerprint if normalization else ''},sort_keys=True,allow_nan=False)
    fingerprint = hashlib.sha256(schema.encode()).hexdigest()
    long = lambda a: torch.tensor(a,dtype=torch.long)
    return PackedObservations(torch.tensor(feats,dtype=torch.float32).reshape(-1,12),
                              long(cells),long(levels),long(slots),long(sources),long(var_ids),
                              torch.tensor(weights,dtype=torch.float32), issue,grid.fingerprint,torch.tensor(p,dtype=torch.float32),
                              vocab,fingerprint,accepted,rejected,
                              normalization.fingerprint if normalization else "",tuple(evidence_ids))
