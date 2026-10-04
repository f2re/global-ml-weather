"""Compact, physically conditioned spherical model (research architecture v0.2).

The mesh is fixed. Adaptation means state-conditioned exchanges, NOT online
retraining, automatic mesh refinement or guaranteed full physical conservation.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import numpy as np
import torch
from torch import nn
from .grid import EARTH_RADIUS_M
from .model import GlobalWeatherModel, GraphOps, ForecastFrame
from .physics import physical_context
from .vertical import (PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES,
                       SURFACE_UNITS, above_ground, tangent_components)
from .observations import utc
from .products.ingest import evidence_history


class DirectedGraph(GraphOps):
    def __init__(self, grid):
        super().__init__(grid)
        src, dst = grid.edges
        delta = grid.xyz[dst] - grid.xyz[src]
        direction = delta/np.linalg.norm(delta, axis=1, keepdims=True)
        length = np.arccos(np.clip((grid.xyz[src]*grid.xyz[dst]).sum(-1), -1, 1))*EARTH_RADIUS_M
        self.register_buffer('src', torch.tensor(src, dtype=torch.long))
        self.register_buffer('dst', torch.tensor(dst, dtype=torch.long))
        self.register_buffer('direction', torch.tensor(direction, dtype=torch.float32))
        self.register_buffer('length', torch.tensor(length, dtype=torch.float32))
        self.register_buffer('degree', torch.bincount(self.dst, minlength=grid.n_cells).float())

    def edge_features(self, wind, context, step_hours):
        i, j = self.src, self.dst
        velocity = .5*(wind[i]+wind[j])
        courant = (velocity*self.direction).sum(-1)*step_hours*3600./self.length
        # Clipping a FEATURE is not a CFL-stability guarantee for a numerical solver.
        return torch.cat((self.direction, torch.log(self.length[:, None]/1e6),
                          courant.clamp(-8, 8)[:, None],
                          (context[j, 6]-context[i, 6])[:, None]), -1)


class AdaptiveBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.sender = nn.Linear(hidden, hidden, bias=False)
        self.receiver = nn.Linear(hidden, hidden, bias=False)
        self.geometry = nn.Linear(6, hidden, bias=False)
        self.context = nn.Linear(8, hidden)
        self.column = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.local = nn.Sequential(nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, hidden))
        self.regime = nn.Linear(hidden+8, 3)

    def forward(self, x, graph, context, wind, step_hours):
        h = self.norm(x)
        geom = self.geometry(graph.edge_features(wind, context, step_hours))
        source, destination = self.sender(h), self.receiver(h)
        exchange = torch.zeros_like(x)
        # Bound temporary edge tensors; the full graph state still scales with N.
        for start in range(0, len(graph.src), 4096):
            sl = slice(start, start+4096)
            i, j = graph.src[sl], graph.dst[sl]
            g = geom[sl, None, :] + self.context(context[i])[:, None, :]
            msg = torch.tanh(source[i]-destination[j]+g)
            gate = torch.sigmoid(source[i]+destination[j]+g)
            exchange = exchange.index_add(0, j, gate*msg)
        exchange = exchange/graph.degree[:, None, None]
        vertical, _ = self.column(h, h, h, need_weights=False)
        condition = context[:, None, :].expand(-1, x.shape[1], -1)
        weights = torch.softmax(self.regime(torch.cat((h, condition), -1)), -1)
        mixed = weights[..., 0, None]*exchange + weights[..., 1, None]*vertical + weights[..., 2, None]*self.local(h)
        return x + .1*mixed


class AdaptiveProcessor(nn.Module):
    def __init__(self, grids, hidden):
        super().__init__()
        self.graphs = nn.ModuleList([DirectedGraph(g) for g in grids])
        self.down = nn.ModuleList([AdaptiveBlock(hidden) for _ in grids])
        self.up = nn.ModuleList([AdaptiveBlock(hidden) for _ in grids[:-1]])
        for i in range(len(grids)-1):
            self.register_buffer(f'parent_{i}', torch.tensor(grids[i+1].tree.query(grids[i].xyz)[1], dtype=torch.long))

    def forward(self, x, context, wind, step_hours):
        saved = []
        for i, (graph, block) in enumerate(zip(self.graphs, self.down)):
            x = block(x, graph, context, wind, step_hours)
            saved.append((x, context, wind))
            if i < len(self.graphs)-1:
                parent = getattr(self, f'parent_{i}')
                nc = len(self.graphs[i+1].area)
                denominator = x.new_zeros(nc).index_add(0, parent, graph.area)
                def pool(a):
                    shape = (-1,) + (1,)*(a.ndim-1)
                    return a.new_zeros((nc, *a.shape[1:])).index_add(
                        0, parent, a*graph.area.reshape(shape))/denominator.reshape(shape)
                x, context, wind = pool(x), pool(context), pool(wind)
        for i in range(len(saved)-2, -1, -1):
            previous, context, wind = saved[i]
            x = previous + x[getattr(self, f'parent_{i}')]
            x = self.up[i](x, self.graphs[i], context, wind, step_hours)
        return x


@dataclass(frozen=True)
class AnalysisState:
    latent: torch.Tensor
    valid_time: datetime
    schema: dict
    evidence: tuple[tuple[str, datetime], ...] = ()


class AdaptiveWeatherModel(GlobalWeatherModel):
    def __init__(self, grids, vocabulary, *, observation_schema, hidden=32,
                 latent_slots=8, step_hours=3, pressure_hpa=None,
                 normalization=None, allow_unscaled_synthetic=False):
        if hidden % 4 or latent_slots not in (4, 8, 16, 38):
            raise ValueError('Hidden size must divide into 4 heads; slots in {4,8,16,38}.')
        if normalization is None and not allow_unscaled_synthetic:
            raise ValueError('Verified reanalysis normalisation required outside explicit synthetic tests.')
        kwargs = {} if pressure_hpa is None else {'pressure_hpa': pressure_hpa}
        super().__init__(grids, vocabulary, observation_schema=observation_schema,
                         hidden=hidden, step_hours=step_hours, **kwargs)
        self.latent_slots, self.normalization = latent_slots, normalization
        self.latent_queries = nn.Parameter(torch.randn(latent_slots, hidden)*.02)
        self.compress = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.expand = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.processor = AdaptiveProcessor(grids, hidden)
        if normalization is not None:
            pm, ps, sm, ss = [], [], [], []
            for name, unit in zip(PROFILE_VARIABLES, PROFILE_UNITS):
                mean, std = normalization.get(name, unit).at(self.pressure_pa.cpu().numpy())
                pm.append(mean); ps.append(std)
            for name, unit in zip(SURFACE_VARIABLES, SURFACE_UNITS):
                interval = step_hours if name == 'precipitation_step' else None
                mean, std = normalization.get(name, unit).at(interval_hours=interval)
                sm.append(mean); ss.append(std)
            for name, value in [('profile_mean', np.array(pm).T), ('profile_std', np.array(ps).T),
                                ('surface_mean', sm), ('surface_std', ss)]:
                self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def get_extra_state(self):
        return dict(architecture='adaptive-v2', grid=self.grid_fingerprint,
                    observations=self.observation_schema, vocabulary=self.vocabulary,
                    pressure_pa=self.pressure_pa.detach().cpu().tolist(),
                    hidden=self.hidden, latent_slots=self.latent_slots, step_hours=self.step_hours,
                    normalization=self.normalization.fingerprint if self.normalization else 'synthetic-unscaled')

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Checkpoint architecture, grid, cadence or normalisation mismatch.')

    def _compress(self, state):
        q = self.latent_queries[None].expand(len(state), -1, -1)
        return self.compress(q, state, state, need_weights=False)[0]

    def _expand(self, latent):
        query = self.level_embed(self.vertical_coordinates)[None].expand(len(latent), -1, -1)
        return query + self.expand(query, latent, latent, need_weights=False)[0]

    def decode(self, state, lead, issue_time, product_mask=None):
        expanded = self._expand(state)
        if self.normalization is None:
            return super().decode(expanded, lead, issue_time, product_mask)
        rp, rs = self.profile_head(expanded[:, :-1]), self.surface_head(expanded[:, -1])
        u, v = tangent_components(rp[..., 2:5], self.xyz[:, None, :])
        z = torch.stack((rp[..., 0], rp[..., 1], u, v, rp[..., 5], rp[..., 6]), -1)
        profiles = z*self.profile_std + self.profile_mean
        profiles = torch.stack((profiles[..., 0], profiles[..., 1].clamp_min(0),
                                *[profiles[..., k] for k in range(2, 6)]), -1)
        u, v = tangent_components(rs[:, 2:5], self.xyz)
        z = torch.stack((rs[:, 0], rs[:, 1], u, v, rs[:, 5], rs[:, 6], rs[:, 7], rs[:, 8]), -1)
        raw = z*self.surface_std + self.surface_mean
        surface = torch.stack((raw[:, 0], torch.minimum(raw[:, 0], raw[:, 1]), raw[:, 2], raw[:, 3],
                               raw[:, 4].clamp_min(1), raw[:, 5].clamp_min(1),
                               raw[:, 6].clamp_min(0), raw[:, 7].clamp(0, 1)), -1)
        if not torch.isfinite(profiles).all() or not torch.isfinite(surface).all():
            raise FloatingPointError('Nonfinite forecast.')
        region = torch.ones(len(state), dtype=torch.bool, device=state.device) if product_mask is None else product_mask
        if region.dtype != torch.bool or region.shape != (len(state),):
            raise ValueError('Product mask must be bool[N].')
        pm = above_ground(self.pressure_pa, surface[:, 4]) & region[:, None]
        sm = region[:, None].expand(-1, 8).clone()
        if lead == 0: sm[:, 6] = False
        return ForecastFrame(lead, issue_time+timedelta(hours=lead), profiles, surface, pm, sm)

    def _validate_inputs(self, obs, elevation, land):
        if (obs.grid_fingerprint != self.grid_fingerprint or obs.vocabulary != self.vocabulary
                or obs.schema_fingerprint != self.observation_schema
                or not torch.equal(obs.pressure_pa, self.pressure_pa)):
            raise ValueError('Observation schema/grid/pressure mismatch.')
        expected = self.normalization.fingerprint if self.normalization else ''
        if obs.normalization_fingerprint != expected:
            raise ValueError('Input and decoder normalisation differ.')
        if elevation.shape != (len(self.xyz),) or land.shape != elevation.shape:
            raise ValueError('Static fields must have shape [N].')
        if not torch.isfinite(elevation).all() or not torch.isfinite(land).all() or ((land < 0) | (land > 1)).any():
            raise ValueError('Invalid static fields.')
        if len(obs.evidence_ids) != len(obs.cells):
            raise ValueError('Stable observation identities required for causal cycling.')

    def _process(self, state, elevation, land, when):
        frame = self.decode(state, 0, when)
        context, wind = physical_context(frame, self.pressure_pa, self.xyz, elevation, land)
        return self.processor(state, context, wind, self.step_hours)

    def analysis_state(self, obs, elevation, land, background=None):
        self._validate_inputs(obs, elevation, land)
        evidence = {}
        if background is not None:
            if (background.schema != self.get_extra_state() or utc(background.valid_time) != obs.issue_time
                    or background.latent.shape != (len(self.xyz), self.latent_slots, self.hidden)
                    or not torch.isfinite(background.latent).all()):
                raise ValueError('Background time, schema or latent state mismatch.')
            evidence = dict(background.evidence)
            keep = torch.tensor([key not in evidence for key in obs.evidence_ids], device=obs.cells.device, dtype=torch.bool)
            fields = {k: getattr(obs, k)[keep] for k in ('features', 'cells', 'levels', 'slots', 'sources', 'variables', 'weights')}
            ids = tuple(key for key, use in zip(obs.evidence_ids, keep.tolist()) if use)
            obs = replace(obs, **fields, evidence_ids=ids, accepted_records=len(set(ids)))
        if background is not None and not len(obs.cells):
            state = background.latent
        else:
            level = self.level_embed(self.vertical_coordinates)
            if background is None:
                static = torch.cat((self.xyz, elevation[:, None]/5000., land[:, None]), -1)
                column = self.static_embed(static)+self.forcing_embed(self.forcing(obs.issue_time))
                initial = column[:, None, :]+level[None, :, :]
                encoded = self.encoder(obs, initial, level)
                state = self._compress(encoded)
            else:
                initial = self._expand(background.latent)
                encoded = self.encoder(obs, initial, level)
                state = background.latent + self._compress(encoded)-self._compress(initial)
            state = self._process(state, elevation, land, obs.issue_time)
        for key, age in zip(obs.evidence_ids, obs.features[:, 1].detach().cpu().tolist()):
            evidence[key] = obs.issue_time-timedelta(hours=age*12.)
        retained = tuple(sorted((key, time) for key, time in evidence.items()
                                if obs.issue_time-timedelta(hours=evidence_history(key)) < time <= obs.issue_time))
        return AnalysisState(state, obs.issue_time, self.get_extra_state(), retained)

    def analyse(self, obs, elevation_m, land_fraction):
        return self.analysis_state(obs, elevation_m, land_fraction).latent

    def _step(self, state, elevation, land, when):
        processed = self._process(state, elevation, land, when-timedelta(hours=self.step_hours))
        forcing = self.forcing_embed(self.forcing(when))[:, None, :]
        return state + (self.step_hours/3)*.1*self.transition(processed+forcing)

    def advance_background(self, background, valid_time, elevation, land):
        valid_time = utc(valid_time)
        hours = (valid_time-utc(background.valid_time)).total_seconds()/3600
        if background.schema != self.get_extra_state() or not 0 <= hours <= 72 or hours % self.step_hours:
            raise ValueError('Background advance must match the trained step (no implicit hourly interpolation).')
        state = background.latent
        for step in range(self.step_hours, int(hours)+1, self.step_hours):
            state = self._step(state, elevation, land, background.valid_time+timedelta(hours=step))
        evidence = tuple((k, t) for k, t in background.evidence if t > valid_time-timedelta(hours=evidence_history(k)))
        return AnalysisState(state, valid_time, self.get_extra_state(), evidence)

    def forward(self, obs, elevation_m, land_fraction, *, horizon_hours=72, product_mask=None, background=None):
        if not isinstance(horizon_hours, int) or not 0 <= horizon_hours <= 72 or horizon_hours % self.step_hours:
            raise ValueError('Invalid forecast horizon.')
        state = self.analysis_state(obs, elevation_m, land_fraction, background).latent
        yield self.decode(state, 0, obs.issue_time, product_mask)
        for lead in range(self.step_hours, horizon_hours+1, self.step_hours):
            state = self._step(state, elevation_m, land_fraction, obs.issue_time+timedelta(hours=lead))
            yield self.decode(state, lead, obs.issue_time, product_mask)
