"""R11 physically constrained assimilation-forecast model.

Structure (decision 0020):

1. Encoder. Every admitted observation token sends a message to its k nearest
   latent nodes (Gaussian kernel, offsets in the node's local frame, wind
   components rotated into that frame). Messages are pooled into vertical
   slots (8 ln p bands, surface, limb, column) together with the log data
   density, so each node knows how much it actually observed.
2. Analysis processor. Multimesh interaction layers spread information
   globally before any physical field is produced (no node stays "empty").
3. Physical state. The analysis head emits a normalised physical state on the
   latent mesh: ln p_s, T(37), RH logit(37), Cartesian wind(37x3), near-surface
   T, RH logit and wind. q = RH q_s(T, p) <= 1.05 q_s; winds are tangent.
4. Step. A multimesh processor maps the state, astronomy and static fields to
   a 3-hour increment. After every step the state is re-constrained: winds are
   projected to the tangent plane and a uniform p_s offset restores the
   global dry-air mass of the analysis (ACE-style corrector).
5. Diagnostics. Phi is integrated hydrostatically upward from Phi_s = g z_s;
   omega follows continuity with the kinematic lower boundary dp_s/dt
   (O'Brien correction). Neither is a free network output.

The network remains a learned dynamical model: these constraints close the
named relations only, not the primitive equations, moisture or energy budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..vertical import PRESSURE_HPA
from . import thermo
from .mesh import LatentMesh, divergence, gradient, torch_buffers
from .tokens import COLUMN_VARIABLES, FEATURE_NAMES, PLATFORMS, PRESSURE_BANDS, VARIABLES, EncodedTokens

ARCHITECTURE = 'physical-assimilation-multimesh-v1'
LEVELS = 37
LNPS = 0
TEMPERATURE = slice(1, 1 + LEVELS)
HUMIDITY = slice(1 + LEVELS, 1 + 2 * LEVELS)
WIND = slice(1 + 2 * LEVELS, 1 + 5 * LEVELS)
SURFACE_TEMPERATURE = 1 + 5 * LEVELS
SURFACE_HUMIDITY = SURFACE_TEMPERATURE + 1
SURFACE_WIND = slice(SURFACE_HUMIDITY + 1, SURFACE_HUMIDITY + 4)
CHANNELS = SURFACE_HUMIDITY + 4
STATIC_FEATURES = 7
FORCING_FEATURES = 5
SLOTS = PRESSURE_BANDS + 3


@dataclass(frozen=True)
class StateNormalization:
    """Fixed scales of the internal state; all values come from pinned GraphCast statistics."""
    profile_mean: tuple   # 37 x 5 (T, q, u, v, Phi), descending pressure
    profile_std: tuple
    mslp_mean: float
    mslp_std: float
    t2m_mean: float
    t2m_std: float
    u10_std: float
    provenance: str = 'graphcast_pinned_statistics'

    @property
    def temperature_mean(self) -> np.ndarray:
        return np.asarray(self.profile_mean)[:, 0]

    @property
    def temperature_std(self) -> np.ndarray:
        return np.asarray(self.profile_std)[:, 0]

    @property
    def wind_std(self) -> np.ndarray:
        std = np.asarray(self.profile_std)
        return np.sqrt(0.5 * (std[:, 2] ** 2 + std[:, 3] ** 2))

    @property
    def log_pressure_scale(self) -> float:
        return self.mslp_std / self.mslp_mean

    @property
    def scale_height_m(self) -> float:
        """Reference H = Rd T / g with the pinned 2 m temperature mean; a decoder offset, not a profile."""
        return thermo.RD * self.t2m_mean / thermo.G0

    @classmethod
    def from_arrays(cls, profile_mean, profile_std, *, mslp_mean, mslp_std, t2m_mean, t2m_std, u10_std,
                    provenance='graphcast_pinned_statistics') -> 'StateNormalization':
        mean, std = np.asarray(profile_mean, float), np.asarray(profile_std, float)
        if mean.shape != (LEVELS, 5) or std.shape != (LEVELS, 5) or (std <= 0).any() or not np.isfinite(mean).all():
            raise ValueError('Profile normalization needs finite 37 x 5 means and positive deviations.')
        for value in (mslp_mean, mslp_std, t2m_mean, t2m_std, u10_std):
            if not (math.isfinite(value) and value > 0):
                raise ValueError('Surface normalization scalars must be finite and positive.')
        as_tuple = lambda array: tuple(tuple(float(x) for x in row) for row in array)
        return cls(as_tuple(mean), as_tuple(std), float(mslp_mean), float(mslp_std),
                   float(t2m_mean), float(t2m_std), float(u10_std), provenance)

    @classmethod
    def from_graphcast(cls) -> 'StateNormalization':
        from ..import_climatology import PINNED_HASHES, bundled_directory, import_graphcast
        from ..profile_graphcast_normalization import _reference
        profile = _reference()
        directory = bundled_directory()
        bundle = import_graphcast(directory / 'mean_by_level.nc', directory / 'stddev_by_level.nc',
                                  expected_hashes=PINNED_HASHES)
        mslp, t2m = bundle.get('mslp', 'Pa'), bundle.get('t2m', 'K')
        u10, v10 = bundle.get('u10', 'm s-1'), bundle.get('v10', 'm s-1')
        return cls.from_arrays(profile['mean'], profile['std'], mslp_mean=mslp.mean[0], mslp_std=mslp.std[0],
                               t2m_mean=t2m.mean[0], t2m_std=t2m.std[0],
                               u10_std=math.sqrt(0.5 * (u10.std[0] ** 2 + v10.std[0] ** 2)),
                               provenance='graphcast_pinned_statistics:' + ','.join(sorted(PINNED_HASHES.values())))

    def payload(self) -> dict:
        return {'profile_mean': [list(row) for row in self.profile_mean],
                'profile_std': [list(row) for row in self.profile_std],
                'mslp_mean': self.mslp_mean, 'mslp_std': self.mslp_std, 't2m_mean': self.t2m_mean,
                't2m_std': self.t2m_std, 'u10_std': self.u10_std, 'provenance': self.provenance}


def mlp(inputs: int, hidden: int, outputs: int, *, norm: bool = True) -> nn.Sequential:
    layers = [nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs)]
    if norm:
        layers.append(nn.LayerNorm(outputs))
    return nn.Sequential(*layers)


class InteractionLayer(nn.Module):
    """GraphCast interaction network: residual edge update, sum aggregation, residual node update."""

    def __init__(self, hidden: int):
        super().__init__()
        self.edge = mlp(3 * hidden, hidden, hidden)
        self.node = mlp(2 * hidden, hidden, hidden)

    def forward(self, nodes: Tensor, edges: Tensor, senders: Tensor, receivers: Tensor):
        edges = edges + self.edge(torch.cat((edges, nodes[senders], nodes[receivers]), -1))
        aggregated = torch.zeros_like(nodes).index_add(0, receivers, edges)
        return nodes + self.node(torch.cat((nodes, aggregated), -1)), edges


class Processor(nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.edge_encoder = mlp(4, hidden, hidden)
        self.layers = nn.ModuleList(InteractionLayer(hidden) for _ in range(layers))
        self.checkpoint_layers = False

    def forward(self, nodes, edge_features, senders, receivers):
        edges = self.edge_encoder(edge_features)
        for layer in self.layers:
            if self.checkpoint_layers and torch.is_grad_enabled():
                # Recompute edge activations in backward: memory ~ one layer instead of all layers.
                nodes, edges = checkpoint(layer, nodes, edges, senders, receivers, use_reentrant=False)
            else:
                nodes, edges = layer(nodes, edges, senders, receivers)
        return nodes


@dataclass
class Rollout:
    """Normalised states and constrained physical fields at leads 0, 3, ..., H."""
    issue: datetime
    leads: list
    states: list
    physics: list
    members: int = 1

    def stack(self, key: str) -> Tensor:
        return torch.stack([frame[key] for frame in self.physics])


class PhysicalAssimilationModel(nn.Module):
    def __init__(self, mesh: LatentMesh, normalization: StateNormalization, static: dict, *,
                 channels: int = 1, hidden: int = 256, message_dim: int = 64,
                 processor_layers: int = 8, analysis_layers: int = 8, encoder_neighbours: int = 7,
                 noise_dim: int = 0, conserve_dry_mass: bool = True, climatology=None,
                 checkpoint_steps: bool = False, step_hours: int = 3):
        super().__init__()
        if step_hours != 3:
            raise ValueError('R11 uses a 3-hour step; a different step needs its own experiment.')
        self.mesh, self.normalization = mesh, normalization
        self.hidden, self.message_dim = hidden, message_dim
        self.encoder_neighbours, self.noise_dim = encoder_neighbours, noise_dim
        self.conserve_dry_mass, self.checkpoint_steps = conserve_dry_mass, checkpoint_steps
        self.step_hours = step_hours
        self.noise_scale = 0.0  # 0: deterministic member; 1: stochastic member for CRPS ensembles
        for name, value in torch_buffers(mesh).items():
            self.register_buffer(name, value, persistent=False)
        self.register_buffer('pressure', torch.tensor(PRESSURE_HPA, dtype=torch.float32) * 100., persistent=False)
        elevation = np.asarray(static['elevation_m'], float)
        land = np.asarray(static['land_fraction'], float)
        roughness = np.asarray(static.get('orography_std_m', np.zeros_like(elevation)), float)
        slope = np.asarray(static.get('slope', np.zeros_like(elevation)), float)
        for array in (elevation, land, roughness, slope):
            if array.shape != (mesh.n_nodes,) or not np.isfinite(array).all():
                raise ValueError('Static fields must be finite on every latent node.')
        if np.any((land < 0) | (land > 1)) or np.any(roughness < 0) or np.any(slope < 0):
            raise ValueError('Invalid land fraction, orography spread or slope.')
        self.register_buffer('elevation', torch.tensor(elevation, dtype=torch.float32))
        self.register_buffer('land_fraction', torch.tensor(land, dtype=torch.float32))
        self.register_buffer('orography_std', torch.tensor(roughness, dtype=torch.float32))
        self.register_buffer('slope', torch.tensor(slope, dtype=torch.float32))
        n = normalization
        self.register_buffer('t_mean', torch.tensor(n.temperature_mean, dtype=torch.float32))
        self.register_buffer('t_std', torch.tensor(n.temperature_std, dtype=torch.float32))
        self.register_buffer('wind_std', torch.tensor(n.wind_std, dtype=torch.float32))
        self.climatology = climatology
        climate_features = 0 if climatology is None else 2 * LEVELS * 5
        self.climate_features = climate_features
        self.platform_embedding = nn.Embedding(len(PLATFORMS), 16)
        self.variable_embedding = nn.Embedding(len(VARIABLES), 16)
        self.channel_embedding = nn.Embedding(channels + 1, 16)
        self.token = mlp(len(FEATURE_NAMES) + 48, message_dim, message_dim)
        self.message = mlp(message_dim + 7, message_dim, message_dim)
        self.kernel_scale = nn.Parameter(torch.zeros(SLOTS))
        node_inputs = (STATIC_FEATURES + FORCING_FEATURES + climate_features + CHANNELS + 1
                       + SLOTS * message_dim + SLOTS + noise_dim)
        self.analysis_encoder = mlp(node_inputs, hidden, hidden)
        self.analysis = Processor(hidden, analysis_layers)
        self.analysis_head = mlp(hidden, hidden, CHANNELS, norm=False)
        self.step_encoder = mlp(CHANNELS + STATIC_FEATURES + FORCING_FEATURES + climate_features + noise_dim,
                                hidden, hidden)
        self.dynamics = Processor(hidden, processor_layers)
        self.step_head = mlp(hidden, hidden, CHANNELS, norm=False)
        self.analysis.checkpoint_layers = checkpoint_steps
        self.dynamics.checkpoint_layers = checkpoint_steps
        self.column_operator = mlp(CHANNELS + 16 + 2, hidden, 1, norm=False)
        self.channel_bias = nn.Embedding(channels + 1, 1)
        nn.init.zeros_(self.channel_bias.weight)
        for head in (self.analysis_head, self.step_head):
            nn.init.normal_(head[-1].weight, std=1e-3)
            nn.init.zeros_(head[-1].bias)

    # ----- static, forcing and climate features -------------------------------------------
    def static_features(self) -> Tensor:
        return torch.stack((self.elevation / 1e4, self.land_fraction, self.orography_std / 1e3,
                            self.slope * 10, self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2]), -1)

    def forcing_features(self, start: datetime) -> Tensor:
        """Mean TOA insolation over [start, start + 3 h], local solar time and season."""
        start = start.astimezone(timezone.utc)
        insolation = thermo.mean_insolation(self.xyz.double(), start, self.step_hours).to(self.xyz.dtype)
        middle = start + timedelta(hours=self.step_hours / 2)
        hours = middle.hour + middle.minute / 60 + middle.second / 3600
        longitude = torch.atan2(self.xyz[:, 1], self.xyz[:, 0])
        local = 2 * math.pi * hours / 24 + longitude
        year = datetime(middle.year, 1, 1, tzinfo=timezone.utc)
        phase = 2 * math.pi * (middle - year).total_seconds() / (366 if middle.year % 4 == 0 else 365) / 86400
        ones = torch.ones_like(local)
        return torch.stack((insolation / thermo.SOLAR_CONSTANT, torch.sin(local), torch.cos(local),
                            ones * math.sin(phase), ones * math.cos(phase)), -1)

    def climate_context(self, when: datetime) -> Tensor:
        if self.climatology is None:
            return self.xyz.new_zeros(self.mesh.n_nodes, 0)
        mean, support = self.climatology.sample(when)
        mean = np.asarray(mean, dtype=np.float32)
        expected = (self.mesh.n_nodes, LEVELS, 5)
        if mean.shape != expected or support.shape != expected:
            raise ValueError('Climate context shape differs from the latent mesh.')
        centre = np.asarray(self.normalization.profile_mean, dtype=np.float32)
        scale = np.asarray(self.normalization.profile_std, dtype=np.float32)
        anomaly = np.where(support, (np.nan_to_num(mean) - centre) / scale, 0.).astype(np.float32)
        features = np.concatenate((anomaly.reshape(len(anomaly), -1), support.reshape(len(anomaly), -1)), -1)
        return torch.tensor(features, dtype=self.xyz.dtype, device=self.xyz.device)

    # ----- state decoding and constraints ---------------------------------------------------
    def decode(self, state: Tensor) -> dict:
        """Normalised state [..., N, C] -> physical fields with built-in bounds."""
        n = self.normalization
        p = self.pressure.to(state.dtype)
        log_reference = math.log(n.mslp_mean) - self.elevation.to(state.dtype) / n.scale_height_m
        surface_pressure = torch.exp(log_reference + n.log_pressure_scale * state[..., LNPS])
        temperature = 100. + F.softplus(self.t_mean + self.t_std * state[..., TEMPERATURE] - 100.)
        humidity = thermo.specific_humidity_from_logit(state[..., HUMIDITY], temperature, p)
        wind = state[..., WIND].unflatten(-1, (LEVELS, 3)) * self.wind_std[:, None]
        xyz = self.xyz.to(state.dtype)
        wind = wind - (wind * xyz[:, None, :]).sum(-1, keepdim=True) * xyz[:, None, :]
        surface_temperature = 100. + F.softplus(n.t2m_mean + n.t2m_std * state[..., SURFACE_TEMPERATURE] - 100.)
        surface_humidity = thermo.specific_humidity_from_logit(state[..., SURFACE_HUMIDITY], surface_temperature,
                                                               surface_pressure)
        surface_wind = state[..., SURFACE_WIND] * n.u10_std
        surface_wind = surface_wind - (surface_wind * xyz).sum(-1, keepdim=True) * xyz
        return self._diagnose({'surface_pressure': surface_pressure, 'temperature': temperature,
                               'humidity': humidity, 'wind': wind, 'surface_temperature': surface_temperature,
                               'surface_humidity': surface_humidity, 'surface_wind': surface_wind})

    def _diagnose(self, fields: dict) -> dict:
        p = self.pressure.to(fields['temperature'].dtype)
        elevation = self.elevation.to(p.dtype)
        fields['geopotential'] = thermo.geopotential_from_surface(
            fields['temperature'], fields['humidity'], p, fields['surface_pressure'], thermo.G0 * elevation)
        fields['surface_virtual_temperature'] = thermo.virtual_temperature(fields['surface_temperature'],
                                                                           fields['surface_humidity'])
        fields['lapse_rate'] = thermo.local_lapse_rate(fields['temperature'], fields['geopotential'], p,
                                                       fields['surface_pressure'])
        fields['mslp'] = thermo.mean_sea_level_pressure(fields['surface_pressure'],
                                                        fields['surface_virtual_temperature'], elevation)
        return fields

    def encode_state(self, state: Tensor, fields: dict) -> Tensor:
        """Write constrained p_s and tangent winds back into the normalised state."""
        n = self.normalization
        log_reference = math.log(n.mslp_mean) - self.elevation.to(state.dtype) / n.scale_height_m
        lnps = (torch.log(fields['surface_pressure']) - log_reference) / n.log_pressure_scale
        wind = (fields['wind'] / self.wind_std[:, None]).flatten(-2)
        surface_wind = fields['surface_wind'] / n.u10_std
        return torch.cat((lnps[..., None], state[..., TEMPERATURE], state[..., HUMIDITY], wind,
                          state[..., SURFACE_TEMPERATURE:SURFACE_HUMIDITY + 1], surface_wind), -1)

    def dry_mass(self, fields: dict) -> Tensor:
        p = self.pressure.double()
        return thermo.global_dry_air_mass(fields['surface_pressure'].double(), fields['humidity'].double(), p,
                                          self.areas)

    def constrain(self, state: Tensor, target_mass: Tensor | None) -> tuple[Tensor, dict]:
        fields = self.decode(state)
        if target_mass is not None and self.conserve_dry_mass:
            corrected = thermo.correct_dry_air_mass(fields['surface_pressure'].double(), fields['humidity'].double(),
                                                    self.pressure.double(), self.areas, target_mass)
            fields['surface_pressure'] = corrected.to(state.dtype)
            fields['surface_humidity'] = thermo.specific_humidity_from_logit(
                state[..., SURFACE_HUMIDITY], fields['surface_temperature'], fields['surface_pressure'])
            fields = self._diagnose(fields)
        return self.encode_state(state, fields), fields

    # ----- encoder ----------------------------------------------------------------------------
    def encode_observations(self, tokens: EncodedTokens | None) -> Tensor:
        n = self.mesh.n_nodes
        pooled = self.xyz.new_zeros(n, SLOTS, self.message_dim)
        weight = self.xyz.new_zeros(n, SLOTS)
        if tokens is None or len(tokens) == 0:
            return torch.cat((pooled.flatten(1), weight), -1)
        index, distance, offset = self.mesh.neighbours(tokens.latitude, tokens.longitude, self.encoder_neighbours)
        index = torch.tensor(index, device=self.xyz.device)
        distance = torch.tensor(distance / self.mesh.node_spacing_m, dtype=self.xyz.dtype, device=self.xyz.device)
        offset = torch.tensor(offset, dtype=self.xyz.dtype, device=self.xyz.device)
        embedded = torch.cat((tokens.features, self.platform_embedding(tokens.platform),
                              self.variable_embedding(tokens.variable), self.channel_embedding(tokens.channel)), -1)
        token = self.token(embedded)
        local_vector = torch.einsum('pkij,pj->pki', self.node_frames[index], tokens.direction)
        k = index.shape[1]
        message = self.message(torch.cat((token[:, None].expand(-1, k, -1), offset, distance[..., None],
                                          local_vector), -1))
        # Learned per-slot Gaussian length scale, initialised to one node spacing. Only the
        # nonzero (token, slot) pairs are expanded: at most two ln p bands per token.
        scale = torch.exp(self.kernel_scale)
        pairs = torch.nonzero(tokens.slot > 0)
        row, slot = pairs[:, 0], pairs[:, 1]
        kernel = torch.exp(-0.5 * (distance[row] / scale[slot][:, None]) ** 2) * tokens.slot[row, slot][:, None]
        flat = (index[row] * SLOTS + slot[:, None]).reshape(-1)
        weight = weight.reshape(-1).index_add(0, flat, kernel.reshape(-1)).reshape(n, SLOTS)
        contribution = (kernel[..., None] * message[row]).reshape(-1, self.message_dim)
        pooled = pooled.reshape(n * SLOTS, -1).index_add(0, flat, contribution).reshape(n, SLOTS, self.message_dim)
        mean = pooled / weight.clamp_min(1e-6)[..., None]
        return torch.cat((mean.flatten(1), torch.log1p(weight)), -1)

    def analyse(self, tokens: EncodedTokens | None, issue: datetime, background: Tensor | None = None) -> Tensor:
        observed = self.encode_observations(tokens)
        if background is None:
            background_features = self.xyz.new_zeros(self.mesh.n_nodes, CHANNELS + 1)
        else:
            background_features = torch.cat((background, torch.ones_like(background[:, :1])), -1)
        noise = (self.noise_scale * torch.randn(self.mesh.n_nodes, self.noise_dim, device=self.xyz.device)
                 if self.noise_dim else self.xyz.new_zeros(self.mesh.n_nodes, 0))
        nodes = self.analysis_encoder(torch.cat((self.static_features(), self.forcing_features(issue - timedelta(hours=3)),
                                                 self.climate_context(issue), background_features, observed, noise), -1))
        nodes = self.analysis(nodes, self.edge_features, self.senders, self.receivers)
        # Physical state stays float32 even under bf16 autocast: 1 K needs ~12 bits at 300 K.
        state = self.analysis_head(nodes).float()
        return state if background is None else background.float() + state

    # ----- dynamics ---------------------------------------------------------------------------
    def _step(self, state: Tensor, forcing: Tensor, climate: Tensor, noise: Tensor) -> Tensor:
        nodes = self.step_encoder(torch.cat((state, self.static_features(), forcing, climate, noise), -1))
        nodes = self.dynamics(nodes, self.edge_features, self.senders, self.receivers)
        return state + self.step_head(nodes).float()

    def step(self, state: Tensor, valid: datetime) -> Tensor:
        forcing = self.forcing_features(valid)
        climate = self.climate_context(valid + timedelta(hours=self.step_hours))
        noise = (self.noise_scale * torch.randn(self.mesh.n_nodes, self.noise_dim, device=state.device, dtype=state.dtype)
                 if self.noise_dim else state.new_zeros(self.mesh.n_nodes, 0))
        if self.checkpoint_steps and torch.is_grad_enabled():
            return checkpoint(self._step, state, forcing, climate, noise, use_reentrant=False)
        return self._step(state, forcing, climate, noise)

    def forward(self, tokens: EncodedTokens | None, issue: datetime, horizon_hours: int, *,
                background: Tensor | None = None) -> Rollout:
        if type(horizon_hours) is not int or not 0 <= horizon_hours <= 72 or horizon_hours % self.step_hours:
            raise ValueError('Horizon must be 0..72 h in 3-hour steps.')
        issue = issue.astimezone(timezone.utc)
        state, fields = self.constrain(self.analyse(tokens, issue, background), None)
        target = self.dry_mass(fields)
        states, physics = [state], [fields]
        for lead in range(self.step_hours, horizon_hours + 1, self.step_hours):
            state, fields = self.constrain(self.step(state, issue + timedelta(hours=lead - self.step_hours)), target)
            states.append(state)
            physics.append(fields)
        return Rollout(issue, list(range(0, horizon_hours + 1, self.step_hours)), states, physics)

    # ----- diagnostics and products -----------------------------------------------------------
    def omega(self, rollout: Rollout) -> Tensor:
        """[F, N, 37] Pa/s from continuity with the kinematic surface boundary."""
        ps = rollout.stack('surface_pressure')
        dt = self.step_hours * 3600.
        if len(ps) == 1:
            tendency = torch.zeros_like(ps)
        else:
            tendency = torch.cat(((ps[1:2] - ps[:1]) / dt, (ps[2:] - ps[:-2]) / (2 * dt), (ps[-1:] - ps[-2:-1]) / dt))
        p = self.pressure.to(ps.dtype)
        out = []
        for index, fields in enumerate(rollout.physics):
            div = divergence(fields['wind'], self.xyz.to(ps.dtype), self.dual_pairs, self.dual_midpoint.to(ps.dtype),
                             self.dual_normal.to(ps.dtype), self.dual_length.to(ps.dtype), self.areas)
            grad = gradient(fields['surface_pressure'], self.xyz.to(ps.dtype), self.dual_pairs,
                            self.dual_normal.to(ps.dtype), self.dual_length.to(ps.dtype), self.areas)
            surface_omega = tendency[index] + (fields['surface_wind'] * grad).sum(-1)
            omega, _ = thermo.omega_with_lower_boundary(div, p, fields['surface_pressure'], surface_omega)
            out.append(omega)
        return torch.stack(out)

    def frames(self, rollout: Rollout) -> list:
        """ForecastFrame list on the latent mesh: T, q, u, v, Phi, omega; below-ground masked."""
        from ..model import ForecastFrame
        from ..vertical import tangent_components
        with torch.no_grad():
            omega = self.omega(rollout)
            p = self.pressure
            frames = []
            for index, (lead, fields) in enumerate(zip(rollout.leads, rollout.physics)):
                u, v = tangent_components(fields['wind'], self.xyz[:, None, :])
                profiles = torch.stack((fields['temperature'], fields['humidity'], u, v,
                                        fields['geopotential'], omega[index]), -1)
                above = p[None, :] <= fields['surface_pressure'][:, None]
                surface = torch.full((self.mesh.n_nodes, 8), float('nan'), device=p.device)
                surface[:, 4] = fields['surface_pressure']
                surface[:, 5] = fields['mslp']
                surface_mask = torch.zeros_like(surface, dtype=torch.bool)
                surface_mask[:, 4:6] = True
                frame = ForecastFrame(lead, rollout.issue + timedelta(hours=lead),
                                      torch.where(above[..., None], profiles, torch.full_like(profiles, float('nan'))),
                                      surface, above, surface_mask)
                frame.profile_variable_mask = above[..., None].expand_as(profiles).clone()
                frame.wind_basis = 'local_enu_vector'
                frame.near_surface = {'temperature': fields['surface_temperature'],
                                      'dew_point': thermo.dew_point_from_humidity(fields['surface_humidity'],
                                                                                 fields['surface_pressure']),
                                      'convention': 'station_native_unknown_instrument_height'}
                frames.append(frame)
            return frames
