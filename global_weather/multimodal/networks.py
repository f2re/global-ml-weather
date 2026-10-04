"""Small mask-aware ResU-Net/ConvGRU, microwave MLP and local sparse attention.

Geometry masks are support, never calibrated confidence. No full N-by-N attention.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def groups(width):
    return math.gcd(8, width)


class ResidualImageBlock(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.skip = nn.Conv2d(inputs, width, 1)
        self.body = nn.Sequential(nn.Conv2d(inputs,width,3,padding=1),nn.GroupNorm(groups(width),width),
                                  nn.GELU(),nn.Conv2d(width,width,3,padding=1),
                                  nn.GroupNorm(groups(width),width),nn.GELU())

    def forward(self, x):
        return self.skip(x)+self.body(x)


class ConvGRU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gates = nn.Conv2d(2*width,2*width,3,padding=1)
        self.candidate = nn.Conv2d(2*width,width,3,padding=1)
        self.decay = nn.Parameter(torch.full((1,width,1,1),-3.))

    def forward(self, x, h, support, elapsed_hours):
        h = torch.zeros_like(x) if h is None else h
        h = h*torch.exp(-F.softplus(self.decay)*elapsed_hours/12.)
        reset, update = torch.sigmoid(self.gates(torch.cat((x,h),1))).chunk(2,1)
        candidate = torch.tanh(self.candidate(torch.cat((x,reset*h),1)))
        result = (1-update)*h+update*candidate
        return torch.where(support, result, h)


class ResUNet(nn.Module):
    """Three-scale U-shaped encoder/decoder; recurrent only on registered images."""
    def __init__(self, channels, hidden, width=16):
        super().__init__()
        self.enc0 = ResidualImageBlock(2*channels+8,width)
        self.enc1 = ResidualImageBlock(width,2*width)
        self.enc2 = ResidualImageBlock(2*width,4*width)
        self.memory = ConvGRU(4*width)
        self.dec1 = ResidualImageBlock(6*width,2*width)
        self.dec0 = ResidualImageBlock(3*width,width)
        self.out = nn.Conv2d(width,hidden,1)

    def forward(self, scene, previous=None, elapsed_hours=0., recurrent=True):
        mask = scene.valid.any(0)[None,None]
        x = torch.cat((torch.where(scene.valid,scene.values,0.),scene.valid.to(scene.values.dtype),scene.geometry),0)[None]
        a = self.enc0(x)
        b = self.enc1(F.avg_pool2d(a,2))
        c = self.enc2(F.avg_pool2d(b,2))
        support = F.adaptive_max_pool2d(mask.float(),c.shape[-2:]).bool()
        h = self.memory(c,previous,support,elapsed_hours) if recurrent else c
        d = self.dec1(torch.cat((F.interpolate(h,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        e = self.dec0(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return self.out(e)[0]*mask[0], h


class MicrowaveEncoder(nn.Module):
    """Joint physical channel identity/value encoding; no convolution between swaths."""
    def __init__(self, channels, hidden):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2*channels+8,2*hidden),nn.GELU(),nn.Linear(2*hidden,hidden))

    def forward(self, scene, previous=None, elapsed_hours=0., recurrent=False):
        x = torch.cat((torch.where(scene.valid,scene.values,0.),scene.valid.to(scene.values.dtype),scene.geometry),0)
        return self.network(x.permute(1,2,0)).permute(2,0,1)*scene.valid.any(0), None


def scatter_scene(features, scene, n_cells):
    """Normalized transpose encoding, not inversion of antenna measurements."""
    values = features.flatten(1).T[scene.pixels]
    total = values.new_zeros(n_cells,values.shape[-1]).index_add(0,scene.cells,values*scene.weights[:,None])
    amount = values.new_zeros(n_cells).index_add(0,scene.cells,scene.weights)
    return total/amount.clamp_min(1e-12)[:,None], amount > 0


class SatelliteEncoder(nn.Module):
    def __init__(self, spec, hidden):
        super().__init__()
        self.spec = spec
        self.image = (ResUNet if spec['encoder']=='unet' else MicrowaveEncoder)(len(spec['channels']),hidden)
        self.sequence = nn.GRUCell(hidden,hidden)
        self.hidden = hidden

    def forward(self, scenes, n_cells):
        device = next(self.parameters()).device
        state = torch.zeros(n_cells,self.hidden,device=device)
        supported = torch.zeros(n_cells,dtype=torch.bool,device=device)
        memories = {}
        for scene in sorted(scenes,key=lambda s:(s.observed,s.identity)):
            memory, previous = memories.get(scene.grid_id,(None,None))
            delta = 0. if previous is None else (scene.observed-previous.observed).total_seconds()/3600
            recurrent = self.spec['temporal']=='registered'
            if recurrent and previous is not None:
                if scene.grid_id != previous.grid_id or scene.values.shape != previous.values.shape:
                    raise ValueError('ConvGRU требует одной пространственной сетки.')
                overlap = scene.valid.any(0).cpu().numpy() & previous.valid.any(0).cpu().numpy()
                if np.any(np.abs(scene.coordinates[overlap]-previous.coordinates[overlap])>1e-6):
                    raise ValueError('ConvGRU требует совмещённых кадров. Перепроецируйте данные или используйте events.')
            image, memory = self.image(scene, memory if recurrent else None, delta, recurrent)
            encoded, present = scatter_scene(image,scene,n_cells)
            candidate = self.sequence(encoded,state)
            state = torch.where(present[:,None],candidate,state)
            supported |= present
            memories[scene.grid_id] = (memory,scene)
        return state,supported


class SourceFusion(nn.Module):
    """Latent slots query per-source evidence; all-missing cells remain unchanged."""
    def __init__(self, hidden, sensors):
        super().__init__()
        self.identities = nn.Parameter(torch.randn(sensors,hidden)*.02)
        self.attention = nn.MultiheadAttention(hidden,4,batch_first=True)
        self.gate = nn.Linear(2*hidden,hidden)

    def forward(self, latent, features, masks):
        present = masks.any(1)
        output = latent.clone()
        if bool(present.any()):
            keys = features[present]+self.identities[None]
            update,_ = self.attention(latent[present],keys,keys,key_padding_mask=~masks[present],need_weights=False)
            gate = torch.sigmoid(self.gate(torch.cat((latent[present],update),-1)))
            output[present] = latent[present]+gate*update
        return output


class SparseObservationEncoder(nn.Module):
    """Local multihead query from state to station/profile/product tokens.

    Uses the prepared vertical links. Sparse attention preserves source and hour;
    support is finite even when a cell/source/level has no neighbours.
    """
    def __init__(self, n_variables, hidden, grid, neighbours=4):
        super().__init__()
        self.hidden,self.heads,self.neighbours = hidden,4,min(neighbours,grid.n_cells)
        self.tree = grid.tree
        self.register_buffer('xyz',torch.tensor(grid.xyz,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(np.sqrt(grid.areas_m2)/6371008.8,dtype=torch.float32))
        self.variable = nn.Embedding(n_variables,hidden)
        self.source = nn.Embedding(6,hidden)
        self.value = nn.Sequential(nn.Linear(12+2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.query = nn.Linear(hidden,hidden,bias=False)
        self.key = nn.Linear(hidden,hidden,bias=False)
        self.relative = nn.Linear(5,4,bias=False)
        self.update = nn.GRUCell(hidden,hidden)

    def forward(self, obs, state, level_features):
        if not len(obs.cells):
            return state
        n,l,d = state.shape; dh=d//4
        token = self.value(torch.cat((obs.features,self.variable(obs.variables),self.source(obs.sources)),-1))
        xyz = obs.features[:,8:11]
        neighbours = self.tree.query(xyz.detach().cpu().numpy(),k=self.neighbours)[1]
        neighbours = torch.as_tensor(neighbours,device=state.device,dtype=torch.long).reshape(-1,self.neighbours)
        for hour in range(12):
            summed = torch.zeros_like(state); sources = state.new_zeros(n,l)
            for src in range(6):
                pick = torch.nonzero((obs.slots==hour)&(obs.sources==src)).flatten()
                if not len(pick): continue
                # Column products use explicit learned pressure queries, not made-up T profiles.
                destinations=[]; origins=[]; weights=[]
                for column in (False,True):
                    rows=pick[(obs.levels[pick]<0)==column]
                    if not len(rows): continue
                    cells=neighbours[rows]
                    levels=(torch.arange(l,device=state.device)[None,None,:].expand(len(rows),self.neighbours,l)
                            if column else obs.levels[rows,None,None].expand(-1,self.neighbours,1))
                    cells=cells[:,:,None].expand_as(levels)
                    destinations.append((cells*l+levels).reshape(-1))
                    origins.append(rows[:,None,None].expand_as(levels).reshape(-1))
                    weights.append(obs.weights[rows,None,None].expand_as(levels).reshape(-1))
                dst,origin,w = torch.cat(destinations),torch.cat(origins),torch.cat(weights)
                cell=dst//l
                displacement=self.xyz[cell]-xyz[origin]
                dist=torch.linalg.vector_norm(displacement,dim=-1)/self.scale[cell].clamp_min(1e-8)
                # Adjacent interpolation support grows with actual cell size, not arbitrary global reach.
                active=dist<=2.
                dst,origin,w,dist,displacement=dst[active],origin[active],w[active],dist[active],displacement[active]
                if not len(dst):continue
                q=self.query(state.flatten(0,1)).reshape(n*l,4,dh)
                keys=self.key(token[origin]).reshape(-1,4,dh)
                relative=torch.cat((displacement,obs.features[origin,1,None],dist[:,None]),-1)
                logits=(q[dst]*keys).sum(-1)/math.sqrt(dh)+self.relative(relative)-.5*dist[:,None]**2
                logits=logits+torch.log(w.clamp_min(1e-12))[:,None]
                index=dst[:,None].expand(-1,4)
                maximum=state.new_full((n*l,4),-torch.inf).scatter_reduce(0,index,logits,reduce='amax',include_self=True)
                scores=torch.exp(logits-maximum[dst]); den=state.new_zeros(n*l,4).index_add(0,dst,scores)
                attention=scores/den[dst].clamp_min(1e-12)
                message=state.new_zeros(n*l,4,dh).index_add(0,dst,attention[:,:,None]*token[origin].reshape(-1,4,dh))
                present=den.sum(-1)>0
                summed=summed+message.reshape(n,l,d)
                sources=sources+present.reshape(n,l)
            present=sources>0
            normalized=summed/sources.clamp_min(1)[:,:,None]
            candidate=self.update(normalized.reshape(-1,d),state.reshape(-1,d)).reshape_as(state)
            state=torch.where(present[:,:,None],candidate,state)
        return state
