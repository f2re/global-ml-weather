"""Локальное усвоение и объединение источников; маска покрытия не означает уверенность."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from ..grid import EARTH_RADIUS_M, unit_xyz
from ..observations import SOURCES
from .networks import ImagerEncoder, MicrowaveEncoder
from .contracts import utc


class SparseObservationEncoder(nn.Module):
    """Запросы ячеек к ближайшим сообщениям отдельно по источнику и давлению.

    T/q/u/v неполного профиля остаются самостоятельными токенами. Маска по уровню
    использует размещение по log(p) из прежнего адаптера. Новые уровни не выдумываются.
    Радиус, число соседей и время входят явно; это не байесовская ковариация.
    """
    def __init__(self,n_variables,hidden,xyz,*,radius_km=500.,neighbors=16,heads=4):
        super().__init__()
        if heads<1 or hidden%heads or not 1<=neighbors<=64 or not 0<radius_km<=20020:
            raise ValueError("Неверная конфигурация разреженного внимания.")
        self.radius,self.neighbors,self.heads,self.hidden=radius_km,neighbors,heads,hidden
        self.register_buffer('query_xyz',torch.as_tensor(xyz,dtype=torch.float32))
        self.variable=nn.Embedding(n_variables,hidden)
        self.source=nn.Embedding(len(SOURCES),hidden)
        self.token=nn.Sequential(nn.Linear(12+2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.q=nn.Linear(hidden,hidden,bias=False);self.k=nn.Linear(hidden,hidden,bias=False)
        self.v=nn.Linear(hidden,hidden,bias=False);self.out=nn.Linear(hidden,hidden,bias=False)
        self.distance_scale=nn.Parameter(torch.zeros(heads))
        self.age_scale=nn.Parameter(torch.zeros(heads))
        self.update=nn.GRUCell(hidden,hidden)

    def forward(self,obs,state,level_features):
        if not len(obs.cells): return state
        n,l,d=state.shape
        token=self.token(torch.cat((obs.features,self.variable(obs.variables),self.source(obs.sources)),1))
        keys,values=self.k(token),self.v(token)
        query=self.q(state+level_features[None])
        merged=torch.zeros_like(state);count=state.new_zeros(n,l)
        query_xyz=self.query_xyz.detach().cpu().numpy()
        radius=2*math.sin(min(math.pi,self.radius*1000/EARTH_RADIUS_M)/2)
        for source in range(len(SOURCES)):
            for level in range(l):
                selected=torch.nonzero((obs.sources==source)&((obs.levels==level)|(obs.levels==-1)),as_tuple=False).flatten()
                if not len(selected):continue
                coords=obs.features[selected,8:11].detach().cpu().numpy()
                tree=cKDTree(coords);k=min(self.neighbors,len(selected))
                for start in range(0,n,128):
                    end=min(n,start+128);b=end-start
                    dist,idx=tree.query(query_xyz[start:end],k=k,distance_upper_bound=radius)
                    dist=np.asarray(dist).reshape(b,k);idx=np.asarray(idx).reshape(b,k)
                    valid=np.isfinite(dist)&(idx<len(selected))
                    safe=np.where(valid,idx,0)
                    ids=selected[torch.as_tensor(safe,device=state.device)]
                    active=torch.as_tensor(valid,device=state.device)
                    qq=query[start:end,level].reshape(b,self.heads,d//self.heads)
                    kk=keys[ids].reshape(b,k,self.heads,d//self.heads).transpose(1,2)
                    vv=values[ids].reshape(b,k,self.heads,d//self.heads).transpose(1,2)
                    scores=(qq[:,:,None,:]*kk).sum(-1)/math.sqrt(d//self.heads)
                    spatial=torch.as_tensor(np.where(valid,dist/radius,0.),device=state.device,dtype=state.dtype)
                    scores=scores-F.softplus(self.distance_scale)[None,:,None]*spatial[:,None,:].square()
                    scores=scores-F.softplus(self.age_scale)[None,:,None]*obs.features[ids,1][:,None,:]
                    scores=scores+obs.weights[ids].clamp_min(1e-12).log()[:,None,:]
                    scores=scores.masked_fill(~active[:,None,:],-1e4)
                    attention=torch.softmax(scores,-1)*active[:,None,:]
                    attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-12)
                    pooled=(attention[:,:,:,None]*vv).sum(2).reshape(b,d)
                    support=active.any(1)
                    merged[start:end,level]=merged[start:end,level]+self.out(pooled)
                    count[start:end,level]=count[start:end,level]+support.to(state.dtype)
        average=merged/count.clamp_min(1)[:,:,None]
        updated=self.update(average.reshape(-1,d),state.reshape(-1,d)).reshape_as(state)
        return torch.where((count>0)[:,:,None],updated,state)


def project_sequence(features,support,age,sequence,sensor,grid):
    """Перенос ПРИЗНАКОВ, не консервативная перепроекция физических полей.

    Оптика: пиксель -> содержащая его ячейка с весом площади пикселя.
    СВЧ: входная нормированная таблица антенной поддержки. Большое пятно не
    заменяется единичной точкой; интегратор отклика прибора остаётся upstream.
    """
    d,h,w=features.shape;n=grid.n_cells
    source=features.reshape(d,-1).T
    if sensor.kind=='imager':
        valid_ids=torch.nonzero(support.flatten(),as_tuple=False).flatten()
        if not len(valid_ids): return features.new_zeros(n,d),torch.zeros(n,dtype=torch.bool,device=features.device),features.new_full((n,),float('inf'))
        lat=sequence.latitude.flatten()[valid_ids].detach().cpu().numpy()
        lon=sequence.longitude.flatten()[valid_ids].detach().cpu().numpy()
        cells=torch.as_tensor(grid.locate(lat,lon),device=features.device,dtype=torch.long)
        scale=np.sqrt(grid.areas_m2[cells.cpu().numpy()])/1000
        _,raw_support=sequence.normalized(sensor)
        footprints=torch.where(raw_support.any(1),sequence.footprint_km,torch.zeros_like(sequence.footprint_km))
        fp=footprints.flatten(1)[:,valid_ids].max(0).values.detach().cpu().numpy()
        if (fp>scale).any():raise ValueError("Пятно больше ячейки: требуется площадной оператор.")
        pixels=valid_ids;weights=sequence.area_m2.flatten()[pixels].to(features.dtype)
    else:
        sequence.validate(sensor,n_cells=n,grid_fingerprint=grid.fingerprint)
        keep=support.flatten()[sequence.link_pixel]
        pixels=sequence.link_pixel[keep];cells=sequence.link_cell[keep]
        weights=sequence.link_weight[keep].to(features.dtype)
    denominator=features.new_zeros(n).index_add(0,cells,weights)
    total=features.new_zeros(n,d).index_add(0,cells,source[pixels]*weights[:,None])
    times=features.new_zeros(n).index_add(0,cells,age.flatten()[pixels]*weights)
    return total/denominator.clamp_min(1e-12)[:,None],denominator>0,torch.where(
        denominator>0,times/denominator.clamp_min(1e-12),times.new_full((),float('inf')))


class SatelliteBank(nn.Module):
    def __init__(self,sensors,hidden,base=16):
        super().__init__()
        self.sensors=tuple(sensors)
        self.encoders=nn.ModuleDict({s.id:(ImagerEncoder(len(s.channels),hidden,base)
                             if s.kind=='imager' else MicrowaveEncoder(len(s.channels),hidden)) for s in sensors})
        self.time=nn.Sequential(nn.Linear(1,hidden),nn.Tanh())
        self.sensor_embed=nn.Embedding(len(sensors),hidden)
        self.hidden=hidden

    def forward(self,sequences,issue_time,grid):
        by_id={s.id:s for s in self.sensors};index={s.id:i for i,s in enumerate(self.sensors)}
        fields=[];masks=[];groups={}
        for seq in sequences:
            if seq.sensor_id not in by_id: raise ValueError("Последовательность не зарегистрирована в модели.")
            sensor=by_id[seq.sensor_id]
            seq=seq.causal(issue_time,sensor)
            if seq is None:continue
            values,valid,age=self.encoders[sensor.id](seq,sensor,utc(issue_time).timestamp())
            features,support,age=project_sequence(values,valid,age,seq,sensor,grid)
            safe_age=torch.where(support,age,torch.zeros_like(age))
            features=features+self.time(safe_age[:,None]/12)+self.sensor_embed.weight[index[sensor.id]][None]
            contribution=torch.where(support[:,None],features,torch.zeros_like(features))
            if sensor.id not in groups:
                groups[sensor.id]=(contribution,support.to(features.dtype))
            else:
                total,count=groups[sensor.id]
                groups[sensor.id]=(total+contribution,count+support.to(features.dtype))
        for sensor in self.sensors:
            if sensor.id in groups:
                total,count=groups[sensor.id]
                fields.append(total/count.clamp_min(1)[:,None]);masks.append(count>0)
        return fields,masks


class SourceFusion(nn.Module):
    """Перекрёстное внимание по доступным источникам; пустая группа даёт ровно ноль."""
    def __init__(self,hidden,heads=4):
        super().__init__()
        self.attention=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.gate=nn.Linear(2*hidden,hidden)

    def forward(self,state,fields,masks):
        if not fields:return state
        values=torch.stack(fields,1);present=torch.stack(masks,1)
        active=present.any(1)
        if not active.any():return state
        output=state.clone()
        ids=torch.nonzero(active,as_tuple=False).flatten()
        for start in range(0,len(ids),128):
            pick=ids[start:start+128];q=state[pick]
            update,_=self.attention(q,values[pick],values[pick],key_padding_mask=~present[pick],need_weights=False)
            weight=torch.sigmoid(self.gate(torch.cat((q,update),-1)))
            output[pick]=q+weight*update
        return output
