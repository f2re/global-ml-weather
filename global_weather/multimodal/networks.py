"""Исполняемые спутниковые кодировщики: ResU-Net, ConvGRU и СВЧ-MLP."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.path=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(4,cout),nn.GELU(),
                                nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(4,cout))
        self.skip=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1)

    def forward(self,x): return F.gelu(self.path(x)+self.skip(x))


class ResUNet(nn.Module):
    """Три уменьшения масштаба, три связи пропуска. Выход на исходной локальной сетке."""
    def __init__(self, channels, hidden, base=16):
        super().__init__()
        if base < 8 or base%4: raise ValueError("Ширина U-Net должна быть кратна четырём.")
        self.enc=nn.ModuleList([Residual(2*channels+4,base),Residual(base,2*base),
                               Residual(2*base,4*base),Residual(4*base,8*base)])
        self.dec=nn.ModuleList([Residual(12*base,4*base),Residual(6*base,2*base),Residual(3*base,base)])
        self.out=nn.Conv2d(base,hidden,1)

    def forward(self,x):
        levels=[]
        for i,block in enumerate(self.enc):
            if i: x=F.avg_pool2d(x,2,ceil_mode=True)
            x=block(x);levels.append(x)
        for block,skip in zip(self.dec,reversed(levels[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False)
            x=block(torch.cat((x,skip),1))
        return self.out(x)


class MaskedConvGRU(nn.Module):
    """Пропуск не обновляет состояние. Реальный интервал и возраст входят в ячейку."""
    def __init__(self,hidden):
        super().__init__()
        self.gates=nn.Conv2d(2*hidden+2,2*hidden,3,padding=1)
        self.candidate=nn.Conv2d(2*hidden+2,hidden,3,padding=1)

    def forward(self,x,state,observed,delta_hours,age_hours):
        times=x.new_tensor([delta_hours/12,age_hours/12])[None,:,None,None].expand(x.shape[0],-1,*x.shape[-2:])
        r,z=torch.sigmoid(self.gates(torch.cat((x,state,times),1))).chunk(2,1)
        candidate=torch.tanh(self.candidate(torch.cat((x,r*state,times),1)))
        update=(1-z)*state+z*candidate
        return torch.where(observed,update,state)


class ImagerEncoder(nn.Module):
    """Отдельный адаптер на прибор/платформу; без будущих кадров и общей нормы воздуха."""
    def __init__(self,channels,hidden,base=16):
        super().__init__()
        self.spatial=ResUNet(channels,hidden,base)
        self.temporal=MaskedConvGRU(hidden)
        self.hidden=hidden

    def forward(self,sequence,sensor,issue_unix):
        x,mask=sequence.normalized(sensor)
        t,_,h,w=x.shape
        state=x.new_zeros((1,self.hidden,h,w));support=torch.zeros((h,w),dtype=torch.bool,device=x.device)
        age=x.new_full((h,w),float('inf'));last=None
        for i in range(t):
            valid=mask[i].any(0)
            current_age=(issue_unix-float(sequence.observed_unix[i]))/3600
            delta=0. if last is None else (float(sequence.observed_unix[i])-last)/3600
            if not valid.any():continue
            last=float(sequence.observed_unix[i])
            geo=torch.stack((torch.cos(torch.deg2rad(sequence.view_zenith_deg[i])),
                             torch.cos(torch.deg2rad(torch.nan_to_num(sequence.solar_zenith_deg[i],nan=90.))),
                             torch.isfinite(sequence.solar_zenith_deg[i]).to(x.dtype),
                             x.new_full((h,w),current_age/12)),0)
            geo=torch.where(valid[None],geo,torch.zeros_like(geo))
            encoded=self.spatial(torch.cat((x[i],mask[i].to(x.dtype),geo),0)[None])
            state=self.temporal(encoded,state,valid[None,None],delta,current_age)
            support |= valid
            age=torch.where(valid,age.new_full((),current_age),age)
        return state[0],support,age


class MicrowaveEncoder(nn.Module):
    """Кодирование согласованных СВЧ-пятен; нет свёртки по номерам каналов или пролётам."""
    def __init__(self,channels,hidden):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(2*channels+3,2*hidden),nn.GELU(),nn.Linear(2*hidden,hidden))
        self.time=nn.GRUCell(hidden+2,hidden)
        self.hidden=hidden

    def forward(self,sequence,sensor,issue_unix):
        x,mask=sequence.normalized(sensor)
        t,c,h,w=x.shape;p=h*w
        state=x.new_zeros((p,self.hidden));support=torch.zeros(p,dtype=torch.bool,device=x.device)
        age=x.new_full((p,),float('inf'));last=None
        for i in range(t):
            valid=mask[i].any(0).flatten()
            current_age=(issue_unix-float(sequence.observed_unix[i]))/3600
            delta=0. if last is None else (float(sequence.observed_unix[i])-last)/3600
            if not valid.any():continue
            last=float(sequence.observed_unix[i])
            geo=torch.stack((torch.cos(torch.deg2rad(sequence.view_zenith_deg[i])).flatten(),
                             sequence.footprint_km[i].flatten()/100,
                             x.new_full((p,),current_age/12)),1)
            geo=torch.where(valid[:,None],geo,torch.zeros_like(geo))
            features=torch.cat((x[i].flatten(1).T,mask[i].flatten(1).T.to(x.dtype),geo),1)
            encoded=self.mlp(features)
            clock=x.new_tensor([delta/12,current_age/12])[None].expand(p,-1)
            candidate=self.time(torch.cat((encoded,clock),1),state)
            state=torch.where(valid[:,None],candidate,state);support |= valid
            age=torch.where(valid,age.new_full((),current_age),age)
        return state.T.reshape(self.hidden,h,w),support.reshape(h,w),age.reshape(h,w)
