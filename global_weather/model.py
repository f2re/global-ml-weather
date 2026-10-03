"""Factorised vertical / multiscale horizontal graph model.

A trainable research implementation, without pretrained weights. Forecast masks
are applied only at decoding; they never crop the global prognostic state.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .grid import SphereGrid
from .observations import SOURCES, PackedObservations, utc
from .vertical import PRESSURE_HPA, above_ground, tangent_components, validate_levels


@dataclass
class ForecastFrame:
    lead_hours: int
    valid_time: datetime
    profiles: Tensor     # [N,L,6], physical units; mask separately
    surface: Tensor      # [N,8], physical units
    profile_mask: Tensor # [N,L], above ground AND product region
    surface_mask: Tensor # [N,8]; lead-0 precipitation is not an accumulation forecast


class GraphOps(nn.Module):
    def __init__(self, grid):
        super().__init__()
        src,dst = grid.edges
        degree = np.bincount(dst,minlength=grid.n_cells)
        adjacency = torch.sparse_coo_tensor(torch.tensor(np.stack([dst,src])),
            torch.tensor(1/degree[dst],dtype=torch.float32),(grid.n_cells,grid.n_cells)).coalesce()
        self.register_buffer('adjacency',adjacency)
        self.register_buffer('area',torch.tensor(grid.areas_m2/grid.areas_m2.mean(),dtype=torch.float32))

    def neighbours(self,x):
        return torch.sparse.mm(self.adjacency,x.flatten(1)).reshape_as(x)


class GraphBlock(nn.Module):
    """O(edges*levels*hidden); pressure coordinates enter through level embeddings."""
    def __init__(self,hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.horizontal = nn.Sequential(nn.Linear(2*hidden,2*hidden),nn.GELU(),nn.Linear(2*hidden,hidden))
        self.vertical = nn.Sequential(nn.Conv1d(hidden,hidden,3,padding=1,groups=hidden),
                                      nn.GELU(),nn.Conv1d(hidden,hidden,1))

    def forward(self,x,graph):
        h = self.norm(x)
        return x + .1*self.horizontal(torch.cat([h,graph.neighbours(h)],-1)) + .1*self.vertical(
            h.transpose(1,2)).transpose(1,2)


class MultiscaleProcessor(nn.Module):
    """Area-weighted algebraic grouping. Not conservative polygon remapping."""
    def __init__(self,grids,hidden):
        super().__init__()
        self.graphs = nn.ModuleList([GraphOps(g) for g in grids])
        self.down = nn.ModuleList([GraphBlock(hidden) for _ in grids])
        self.up = nn.ModuleList([GraphBlock(hidden) for _ in grids[:-1]])
        for i in range(len(grids)-1):
            parent = grids[i+1].tree.query(grids[i].xyz)[1]
            self.register_buffer(f'parent_{i}',torch.tensor(parent,dtype=torch.long))

    def forward(self,x):
        saved = []
        for i,(graph,block) in enumerate(zip(self.graphs,self.down)):
            x = block(x,graph)
            saved.append(x)
            if i < len(self.graphs)-1:
                parent = getattr(self,f'parent_{i}')
                nc = self.graphs[i+1].area.numel()
                sums = x.new_zeros(nc).index_add(0,parent,graph.area)
                x = x.new_zeros((nc,*x.shape[1:])).index_add(0,parent,x*graph.area[:,None,None]) / sums[:,None,None]
        for i in range(len(self.graphs)-2,-1,-1):
            x = saved[i]+x[getattr(self,f'parent_{i}')]
            x = self.up[i](x,self.graphs[i])
        return x


class ObservationEncoder(nn.Module):
    """Source-separated sparse pooling, then 12 chronological GRU updates.

    Surface/sonde observations keep their vertical location. Column radiances
    have learned level-dependent gates, not an assumed brightness-T profile.
    """
    def __init__(self,n_variables,hidden):
        super().__init__()
        self.variable = nn.Embedding(n_variables,hidden)
        self.source = nn.Embedding(len(SOURCES),hidden)
        self.value = nn.Sequential(nn.Linear(12+2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.source_projection = nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in SOURCES])
        self.column_gate = nn.Linear(hidden,hidden)
        self.update = nn.GRUCell(hidden,hidden)

    def forward(self,obs: PackedObservations,state,level_features):
        n,l,d = state.shape
        if not len(obs.cells):
            return state
        token = self.value(torch.cat([obs.features,self.variable(obs.variables),
                                      self.source(obs.sources)],dim=-1))
        for hour in range(12):
            merged = torch.zeros_like(state)
            source_count = state.new_zeros(n,l)
            for s in range(len(SOURCES)):
                pick = (obs.slots == hour) & (obs.sources == s)
                if not bool(pick.any()):
                    continue
                cells,levels = obs.cells[pick],obs.levels[pick]
                values,weights = token[pick],obs.weights[pick]
                # Keep levels separate; a missing profile level is not a zero observation.
                placed = levels >= 0
                total = state.new_zeros(n*l,d)
                count = state.new_zeros(n*l)
                indices = cells[placed]*l+levels[placed]
                total = total.index_add(0,indices,values[placed]*weights[placed,None])
                count = count.index_add(0,indices,weights[placed])
                pooled = (total/count.clamp_min(1e-12)[:,None]).reshape(n,l,d)
                supported = count.reshape(n,l) > 0
                column = ~placed
                if bool(column.any()):
                    csum = state.new_zeros(n,d).index_add(0,cells[column],values[column]*weights[column,None])
                    cw = state.new_zeros(n).index_add(0,cells[column],weights[column])
                    columns = csum/cw.clamp_min(1e-12)[:,None]
                    pooled = pooled + columns[:,None,:]*torch.sigmoid(self.column_gate(level_features))[None,:,:]
                    supported = supported | (cw[:,None] > 0)
                merged = merged+self.source_projection[s](pooled)
                source_count = source_count+supported.to(state.dtype)
            active = source_count > 0
            merged = merged/source_count.clamp_min(1.)[:,:,None]
            updated = self.update(merged.reshape(-1,d),state.reshape(-1,d)).reshape_as(state)
            state = torch.where(active[:,:,None],updated,state)
        return state


class GlobalWeatherModel(nn.Module):
    def __init__(self,grids: list[SphereGrid], vocabulary, *, observation_schema: str, hidden=32, step_hours=3,
                 pressure_hpa=PRESSURE_HPA):
        super().__init__()
        if not grids or hidden < 8 or step_hours not in (1,3,6):
            raise ValueError('Need grids, hidden >=8, step_hours in {1,3,6}.')
        if any(a.level != b.level+1 for a,b in zip(grids,grids[1:])):
            raise ValueError('Pyramid must descend one refinement at a time.')
        p = validate_levels(pressure_hpa)*100
        self.vocabulary = tuple(vocabulary)
        if not self.vocabulary or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError('A unique variable vocabulary is required.')
        if not observation_schema:
            raise ValueError("Observation schema fingerprint is required.")
        self.observation_schema = observation_schema
        self.grid_fingerprint = grids[0].fingerprint
        self.hidden,self.step_hours = hidden,step_hours
        self.register_buffer('xyz',torch.tensor(grids[0].xyz,dtype=torch.float32))
        self.register_buffer('pressure_pa',torch.tensor(p,dtype=torch.float32))
        vertical = np.stack([np.r_[np.log(p/100_000.)/7,0.], np.r_[np.zeros(len(p)),1.]],axis=1)
        self.register_buffer('vertical_coordinates',torch.tensor(vertical,dtype=torch.float32))
        self.level_embed = nn.Sequential(nn.Linear(2,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.static_embed = nn.Linear(5,hidden)
        self.forcing_embed = nn.Linear(5,hidden)
        self.encoder = ObservationEncoder(len(vocabulary),hidden)
        self.processor = MultiscaleProcessor(grids,hidden)
        self.transition = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.Tanh())
        self.profile_head = nn.Linear(hidden,7)
        self.surface_head = nn.Linear(hidden,9)

    def forcing(self,when):
        """UTC/season cycles and low-order solar-zenith proxy, not a radiation model."""
        when = utc(when)
        hour = when.hour+when.minute/60+when.second/3600
        days = (when-datetime(when.year,1,1,tzinfo=when.tzinfo)).total_seconds()/86400
        annual,daily = 2*math.pi*days/365.2425,2*math.pi*hour/24
        decl = math.radians(23.44)*math.sin(annual-2*math.pi*79/365.2425)
        sun = self.xyz.new_tensor([-math.cos(decl)*math.cos(daily),
                                  math.cos(decl)*math.sin(daily),math.sin(decl)])
        cycles = self.xyz.new_tensor([math.sin(daily),math.cos(daily),math.sin(annual),math.cos(annual)])
        return torch.cat([cycles.expand(len(self.xyz),-1),(self.xyz@sun)[:,None]],-1)

    def analyse(self,obs,elevation_m,land_fraction):
        if obs.grid_fingerprint != self.grid_fingerprint or obs.vocabulary != self.vocabulary or obs.schema_fingerprint != self.observation_schema:
            raise ValueError('Observation grid/vocabulary/normalization/pressure axis does not match the model.')
        if not torch.equal(obs.pressure_pa,self.pressure_pa):
            raise ValueError('Observation vertical coordinate does not match model pressure levels.')
        if len(elevation_m) != len(self.xyz) or elevation_m.shape != land_fraction.shape:
            raise ValueError('Static fields must have shape [N].')
        if elevation_m.ndim != 1 or not torch.isfinite(elevation_m).all() or not torch.isfinite(land_fraction).all():
            raise ValueError('Invalid static fields.')
        if ((land_fraction < 0) | (land_fraction > 1)).any():
            raise ValueError('Land fraction outside [0,1].')
        static = torch.cat([self.xyz,elevation_m[:,None]/5000.,land_fraction[:,None]],-1)
        column = self.static_embed(static)+self.forcing_embed(self.forcing(obs.issue_time))
        vertical = self.level_embed(self.vertical_coordinates)
        state = column[:,None,:]+vertical[None,:,:]
        return self.processor(self.encoder(obs,state,vertical))

    def decode(self,state,lead,issue_time,product_mask=None):
        rp,rs = self.profile_head(state[:,:-1]),self.surface_head(state[:,-1])
        u,v = tangent_components(rp[...,2:5]*20.,self.xyz[:,None,:])
        profiles = torch.stack([100.+50.*F.softplus(rp[...,0]+3.),
                                .005*F.softplus(rp[...,1]),u,v,rp[...,5]*100_000.,rp[...,6]*.5],-1)
        u10,v10 = tangent_components(rs[:,2:5]*20.,self.xyz)
        t2 = 273.15+30.*rs[:,0]
        surface = torch.stack([t2,t2-10.*F.softplus(rs[:,1]),u10,v10,
                               100_000.*torch.exp(.1*rs[:,5]),100_000.*torch.exp(.1*rs[:,6]),
                               F.softplus(rs[:,7]),torch.sigmoid(rs[:,8])],-1)
        if not torch.isfinite(profiles).all() or not torch.isfinite(surface).all():
            raise FloatingPointError('Nonfinite model output; no valid forecast may be issued.')
        region = torch.ones(len(self.xyz),dtype=torch.bool,device=state.device) if product_mask is None else product_mask
        if region.dtype != torch.bool or region.shape != (len(self.xyz),):
            raise ValueError('Product mask must be bool[N].')
        valid = above_ground(self.pressure_pa,surface[:,4]) & region[:,None]
        surface_valid = region[:,None].expand(-1,8).clone()
        if lead == 0:
            surface_valid[:,6] = False
        return ForecastFrame(lead,issue_time+timedelta(hours=lead),profiles,surface,valid,surface_valid)

    def forward(self,obs,elevation_m,land_fraction,*,horizon_hours=72,product_mask=None):
        """Yield analysis (lead 0) and fixed-step forecasts. No future observations.

        Call under torch.inference_mode() for inference. During training the
        generator retains gradients through the autoregressive rollout.
        """
        if not isinstance(horizon_hours,int) or not 0 <= horizon_hours <= 72 or horizon_hours % self.step_hours:
            raise ValueError('Horizon must be 0..72 and divisible by model step.')
        state = self.analyse(obs,elevation_m,land_fraction)
        yield self.decode(state,0,obs.issue_time,product_mask)
        for lead in range(self.step_hours,horizon_hours+1,self.step_hours):
            forcing = self.forcing_embed(self.forcing(obs.issue_time+timedelta(hours=lead)))[:,None,:]
            state = state + (self.step_hours/3)*.1*self.transition(self.processor(state)+forcing)
            yield self.decode(state,lead,obs.issue_time,product_mask)
