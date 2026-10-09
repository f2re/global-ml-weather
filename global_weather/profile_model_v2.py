"""Pressure-conditioned observed-profile model; no reanalysis observation inputs."""
from __future__ import annotations
from datetime import timedelta
import numpy as np
import torch
from torch import nn
from .grid import unit_xyz
from .model import ForecastFrame, GraphOps
from .observations import utc
from .profile_training import VARIABLES
from .vertical import PRESSURE_HPA
from .profile_normalization import PressureNormalization


class PressureProfileModel(nn.Module):
    def __init__(self, grid, normalization: PressureNormalization, hidden: int = 32):
        super().__init__(); self.grid=grid; self.normalization=normalization; self.hidden=hidden
        means,stds,support=normalization.arrays()
        # Unsupported entries remain NaN in the immutable norm artifact. These
        # internal neutral operands are never emitted or used as observed norms.
        self.register_buffer('mean',torch.tensor(np.where(support,means,0.),dtype=torch.float32))
        self.register_buffer('std',torch.tensor(np.where(support,stds,1.),dtype=torch.float32))
        self.register_buffer('norm_support',torch.tensor(support,dtype=torch.bool))
        self.register_buffer('pressure_pa',torch.tensor(PRESSURE_HPA,dtype=torch.float32)*100)
        self.register_buffer('xyz',torch.tensor(grid.xyz,dtype=torch.float32))
        self.register_buffer('log_pressure',torch.log(self.pressure_pa/100000.)[:,None])
        self.geometry=nn.Linear(3,hidden); self.pressure_encoder=nn.Sequential(nn.Linear(1,hidden),nn.Tanh(),nn.Linear(hidden,hidden))
        self.levels=nn.Parameter(torch.randn(37,hidden)*.01); self.variable=nn.Embedding(5,hidden)
        self.token=nn.Sequential(nn.Linear(hidden+6,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.ingest=nn.GRUCell(hidden,hidden); self.graph=GraphOps(grid)
        self.dynamic=nn.Linear(hidden*4,hidden); self.step=nn.GRUCell(hidden,hidden); self.head=nn.Linear(hidden,5)
        # Start near fixed level means, with finite gradient through
        # every head. This is a predicted background, never a missing target.
        nn.init.normal_(self.head.weight,std=.001); nn.init.zeros_(self.head.bias)

    def _initial_state(self, issue, pressure):
        return self.geometry(self.xyz)[:,None]+pressure[None]

    def _forcing(self, issue, lead, pressure):
        return pressure[None].expand(self.grid.n_cells,-1,-1)

    def _decode(self,state,issue,lead):
        transformed=self.head(state)*self.std+self.mean
        # Mask before the nonlinear inverse: exp(invalid) can overflow and
        # poison backward even when a later output mask removes that value.
        q_transformed=torch.where(self.norm_support[:,1],transformed[...,1],torch.zeros_like(transformed[...,1]))
        q=(q_transformed.clamp_min(0.) if self.normalization.humidity_transform=='identity'
           else self.normalization.q_scale*torch.expm1(q_transformed).clamp_min(0.))
        values=torch.stack((transformed[...,0],q,*[transformed[...,i] for i in range(2,5)]),-1)
        profiles=torch.cat((values,values.new_full((*values.shape[:-1],1),float('nan'))),-1)
        mask=torch.cat((self.norm_support,torch.zeros(37,1,dtype=torch.bool,device=state.device)),-1)[None].expand_as(profiles)
        profiles=torch.where(mask,profiles,torch.full_like(profiles,float('nan')))
        frame=ForecastFrame(lead,issue+timedelta(hours=lead),profiles,values.new_full((self.grid.n_cells,8),float('nan')),
                            mask[...,:5].any(-1),torch.zeros(self.grid.n_cells,8,dtype=torch.bool,device=state.device))
        frame.profile_variable_mask=mask; frame.wind_basis='local_enu_vector'
        return frame

    def forward(self,inputs,issue):
        issue=utc(issue); n,d=self.grid.n_cells,self.hidden
        pressure=self.pressure_encoder(self.log_pressure)+self.levels
        state=self._initial_state(issue,pressure)
        for hour in range(12):
            rows=[r for r in inputs if issue-timedelta(hours=12-hour)<utc(r['observed_at'])<=issue-timedelta(hours=11-hour)
                  and utc(r['available_at'])<=issue]
            rows=[r for r in rows if self.normalization.at(r['variable'],r['pressure_pa'])[2]]
            if not rows: continue
            variables=torch.tensor([VARIABLES.index(r['variable']) for r in rows],device=state.device)
            pressures=np.array([r['pressure_pa'] for r in rows]); levels=np.abs(np.log(pressures[:,None])-np.log(np.array(PRESSURE_HPA)[None]*100)).argmin(1)
            cells=self.grid.locate([r['latitude'] for r in rows],[r['longitude'] for r in rows])
            indices=torch.tensor(cells*37+levels,device=state.device)
            normalized=[self.normalization.normalize(r['variable'],r['value'],r['pressure_pa']) for r in rows]
            xyz=unit_xyz([r['latitude'] for r in rows],[r['longitude'] for r in rows])
            extra=np.column_stack((normalized,np.log(pressures/100000),[(issue-utc(r['observed_at'])).total_seconds()/43200 for r in rows],xyz))
            features=torch.tensor(extra,dtype=state.dtype,device=state.device)
            tokens=self.token(torch.cat((features,self.variable(variables)),-1))
            sums=state.new_zeros(n*37,d).index_add(0,indices,tokens)
            counts=state.new_zeros(n*37).index_add(0,indices,torch.ones(len(rows),device=state.device))
            updated=self.ingest(sums/counts.clamp_min(1)[:,None],state.reshape(-1,d))
            state=torch.where((counts>0)[:,None],updated,state.reshape(-1,d)).reshape(n,37,d)
        frames=[self._decode(state,issue,0)]
        for lead in range(3,73,3):
            context=torch.cat((state,self.graph.neighbours(state),state.mean(1,keepdim=True).expand_as(state),self._forcing(issue,lead,pressure)),-1)
            updated=self.step(torch.tanh(self.dynamic(context)).reshape(-1,d),state.reshape(-1,d)).reshape(n,37,d)
            state=state+.1*(updated-state)
            frames.append(self._decode(state,issue,lead))
        return frames
