"""Проверка всех ветвей на аналитическом примере, не оценка точности погоды."""
from __future__ import annotations
from pathlib import Path
import torch
from ..grid import build_pyramid
from ..checkpoints import save_checkpoint,load_checkpoint
from .fixture import fixture
from .model import MultimodalWeatherModel
from .io import write_json,sha256,exclusive
import numpy as np


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError('Нужен новый каталог проверки.')
    out.mkdir(parents=True)
    torch.set_num_threads(1);torch.manual_seed(71)
    grids=build_pyramid(1);sensors,obs=fixture(grids)
    def model():
        return MultimodalWeatherModel(grids,obs.vocabulary,observation_schema=obs.schema_fingerprint,
            hidden=16,latent_slots=8,sensors=sensors,sensor_signature=obs.sensor_signature,
            base_channels=8,neighbors=8,radius_km=1000.,allow_unscaled_synthetic=True)
    m=model();opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    zero=torch.zeros(grids[0].n_cells);history=[];gradient_report={}
    target=265.+8.*m.xyz[:,2,None]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        frame=list(m(obs,zero,zero,horizon_hours=6))[-1]
        loss=((frame.profiles[...,0]-target)/30).square().mean()+((frame.surface[:,0]-280)/20).square().mean()
        loss.backward()
        modules={'sparse_encoder':m.encoder,'source_fusion':m.source_fusion,'vertical_compress':m.compress,
                 'graph_processor':m.processor,'vertical_decode':m.expand,'profile_head':m.profile_head,'surface_head':m.surface_head,
                 **dict(m.satellite_bank.encoders.items())}
        for name,mod in modules.items():
            grads=[p.grad for p in mod.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('Нет конечных градиентов: '+name)
            norm=sum(float(g.abs().sum()) for g in grads)
            if norm==0:raise RuntimeError('Ветвь не влияет на обучение: '+name)
            gradient_report[name]=norm
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        history.append(float(loss.detach()))
    save_checkpoint(out/'synthetic-weights.pt',m)
    restored=model();load_checkpoint(out/'synthetic-weights.pt',restored)
    m.eval();restored.eval();leads=[]
    with torch.inference_mode():
        a=list(m(obs,zero,zero,horizon_hours=72));b=list(restored(obs,zero,zero,horizon_hours=72))
        identical=all(torch.equal(x.profiles,y.profiles) and torch.equal(x.surface,y.surface) for x,y in zip(a,b))
        if not identical:raise RuntimeError('Контрольная точка не воспроизводит прогноз.')
        for frame in b:
            leads.append(frame.lead_hours)
            if not torch.isfinite(frame.profiles).all():raise RuntimeError('Неконечный прогноз.')
            data={k:getattr(frame,k).cpu().numpy() for k in ('profiles','surface','profile_mask','surface_mask')}
            exclusive(out/f'frame-{frame.lead_hours:03}.npz',lambda f,d=data:np.savez_compressed(f,**d))
    report={'status':'synthetic_multimodal_integration_passed','data_kind':'synthetic','meteorologically_validated':False,
        'cells':grids[0].n_cells,'profile_levels':37,'lead_hours':leads,'optimizer_steps':len(history),'loss':history,
        'nonzero_finite_gradients':gradient_report,'checkpoint_reproduces_rollout':identical,
        'checkpoint_sha256':sha256(out/'synthetic-weights.pt'),'real_downloads_performed':False}
    write_json(out/'report.json',report)
    return report
