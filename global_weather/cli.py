"""CPU runnable geometry, synthetic rollout and optimizer smoke checks."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import numpy as np
import torch
from .grid import build_pyramid,latlon
from .observations import pack_observations,read_jsonl,DEFAULT_VARIABLES,SOURCES,read_variables
from .model import GlobalWeatherModel
from .losses import forecast_loss
from .vertical import (PRESSURE_HPA,PROFILE_VARIABLES,SURFACE_VARIABLES,
                       PROFILE_UNITS,SURFACE_UNITS)


def synthetic_records(grid,issue):
    """Analytic fixture only: not station downloads, not satellite retrievals."""
    records = []
    for c in range(0,grid.n_cells,max(1,grid.n_cells//12)):
        lat,lon = latlon(grid.xyz[c])
        for age in (0,3,6,11):
            for variable,value,pressure in [('t2m',278.+5*grid.xyz[c,2]+age*.1,None),
                                            ('temperature',250.+10*grid.xyz[c,2],50000.)]:
                rec = dict(observation_id=f'{c}-{age}-{variable}',source='station' if pressure is None else 'radiosonde',
                           variable=variable,value=value,units='K',latitude=float(lat),longitude=float(lon),
                           observed_at=(issue-timedelta(hours=age)).isoformat(),available_at=issue.isoformat(),
                           quality=1.,valid=True)
                if pressure: rec['pressure_pa']=pressure
                records.append(rec)
    return records


def make_parser():
    parser = argparse.ArgumentParser(description='Research prototype; no validated weather weights are supplied.')
    parser.add_argument('command',choices=('grid','demo','train-smoke','validate-observations'))
    parser.add_argument('--mesh-level',type=int,default=1)
    parser.add_argument('--hidden',type=int,default=16)
    parser.add_argument('--step-hours',type=int,choices=(1,3,6),default=3)
    parser.add_argument('--horizon-hours',type=int,default=72)
    parser.add_argument('--output',type=Path,default=Path('outputs/demo'))
    parser.add_argument('--observations',type=Path)
    parser.add_argument('--variables',type=Path,help='Explicit physical-variable/channel registry JSON')
    parser.add_argument('--issue-time',default='2026-10-03T00:00:00+00:00')
    parser.add_argument('--optimizer-steps',type=int,default=2)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--region',nargs=4,type=float,metavar=('SOUTH','NORTH','WEST','EAST'))
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.threads < 1 or args.optimizer_steps < 1:
        raise ValueError('Positive threads and optimizer steps required.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    grids = build_pyramid(args.mesh_level)
    grid = grids[0]
    args.output.mkdir(parents=True,exist_ok=True)
    grid.save(args.output/'grid.npz')
    if args.command == 'grid':
        print(json.dumps({'cells':grid.n_cells,'pentagons':12,'level':grid.level,'fingerprint':grid.fingerprint})); return
    from .observations import utc
    issue = utc(args.issue_time)
    if args.command == 'validate-observations' and args.observations is None:
        raise ValueError('--observations is required.')
    if args.command != 'validate-observations' and args.observations is not None:
        raise ValueError('demo/train-smoke use synthetic inputs only; no operational forecast mode exists yet.')
    records = read_jsonl(args.observations) if args.observations else synthetic_records(grid,issue)
    variables = read_variables(args.variables) if args.variables else DEFAULT_VARIABLES
    obs = pack_observations(records,grid,np.array(PRESSURE_HPA)*100,issue,variables)
    if args.command == 'validate-observations':
        print(json.dumps({'accepted':obs.accepted_records,'rejected':obs.rejected},ensure_ascii=False)); return
    model = GlobalWeatherModel(grids,obs.vocabulary,observation_schema=obs.schema_fingerprint,hidden=args.hidden,step_hours=args.step_hours)
    # Placeholder static fields ONLY for the synthetic fixture; real training supplies terrain and land fraction.
    elevation = torch.zeros(grid.n_cells)
    land = torch.zeros(grid.n_cells)
    mask = torch.tensor(grid.region_mask(*args.region)) if args.region else None
    report = dict(status='synthetic_untrained',scientifically_validated=False,
                  grid_level=grid.level,cells=grid.n_cells,pressure_levels_hpa=PRESSURE_HPA,
                  step_hours=args.step_hours,horizon_hours=args.horizon_hours,seed=args.seed,
                  parameters=sum(p.numel() for p in model.parameters()),input_records=obs.accepted_records,
                  grid_fingerprint=grid.fingerprint,torch=torch.__version__,numpy=np.__version__)
    if args.command == 'train-smoke':
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        losses = []
        # Frozen synthetic teacher values demonstrate backward/optimizer wiring, not forecast skill.
        with torch.no_grad():
            teacher = list(model(obs,elevation,land,horizon_hours=args.step_hours))[-1]
            target_p = teacher.profiles.detach().clone()
            target_s = teacher.surface.detach().clone()
            target_p[...,0] += 1.
            target_s[:,0] += 1.
        for _ in range(args.optimizer_steps):
            optimizer.zero_grad(set_to_none=True)
            frame = list(model(obs,elevation,land,horizon_hours=args.step_hours))[-1]
            loss = forecast_loss(frame,target_p,teacher.profile_mask[...,None].expand_as(target_p),
                                 target_s,teacher.surface_mask,torch.tensor(grid.areas_m2,dtype=torch.float32))
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('Nonfinite gradients.')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        report.update(status='synthetic_optimizer_smoke',losses=losses)
    model.eval()
    with torch.inference_mode():
        frames = list(model(obs,elevation,land,horizon_hours=args.horizon_hours,product_mask=mask))
    profiles = np.stack([f.profiles.numpy() for f in frames])
    surface = np.stack([f.surface.numpy() for f in frames])
    pmask = np.stack([f.profile_mask.numpy() for f in frames])
    smask = np.stack([f.surface_mask.numpy() for f in frames])
    coverage,age = obs.coverage(grid.n_cells)
    np.savez_compressed(args.output/'synthetic_forecast.npz',
                        observation_mask=coverage.numpy(),input_age_at_issue_hours=age.numpy(),
                        observation_sources=np.array(SOURCES),
                        profiles=np.where(pmask[...,None],profiles,np.nan),
                        surface=np.where(smask,surface,np.nan),profile_mask=pmask,surface_mask=smask,
                        lead_hours=np.array([f.lead_hours for f in frames]),
                        valid_time_utc=np.array([f.valid_time.isoformat() for f in frames]),
                        issue_time_utc=issue.isoformat(),pressure_hpa=np.array(PRESSURE_HPA),
                        profile_variables=np.array(PROFILE_VARIABLES),surface_variables=np.array(SURFACE_VARIABLES),
                        profile_units=np.array(PROFILE_UNITS),surface_units=np.array(SURFACE_UNITS),
                        grid_fingerprint=grid.fingerprint,status=report['status'])
    report['frames'] = len(frames)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    main()
