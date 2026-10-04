"""Local scene preparation and neural connectivity audit; no hidden downloads."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    c=commands.add_parser('demo-dataset');c.add_argument('--output',required=True);c.add_argument('--horizon-hours',type=int,default=3)
    c=commands.add_parser('from-capsules');c.add_argument('--job',required=True);c.add_argument('--output',required=True)
    c=commands.add_parser('tile-scene');c.add_argument('--input',required=True);c.add_argument('--output',required=True);c.add_argument('--tile-size',type=int,default=128)
    c=commands.add_parser('attach');c.add_argument('--dataset',required=True);c.add_argument('--plan',required=True);c.add_argument('--output',required=True)
    c=commands.add_parser('fit-channels');c.add_argument('--dataset',required=True);c.add_argument('--output',required=True)
    c=commands.add_parser('audit');c.add_argument('--dataset',required=True);c.add_argument('--config');c.add_argument('--output')
    args=p.parse_args(argv)
    if args.command=='demo-dataset':
        from .fixture import create_dataset
        report={'data_kind':'synthetic','dataset':str(create_dataset(args.output,horizon_hours=args.horizon_hours))}
    elif args.command=='from-capsules':
        from .prepare import from_capsules
        report=from_capsules(args.job,args.output)
    elif args.command=='tile-scene':
        from .prepare import tile_scene
        report=tile_scene(args.input,args.output,tile_size=args.tile_size)
    elif args.command=='attach':
        from .prepare import attach
        report=attach(args.dataset,args.plan,args.output)
    elif args.command=='fit-channels':
        from .prepare import fit_channels
        report=fit_channels(args.dataset,args.output)
    else:
        import torch
        from .dataset import TrainingDataset
        from .model import make_model,describe
        from ..pipeline.runner import TrainConfig,config_from_json,target_tensors
        from ..pipeline.io import read_json,atomic_json
        from ..training import train_step
        ds=TrainingDataset(args.dataset)
        cfg=config_from_json(read_json(args.config)) if args.config else TrainConfig()
        cfg.validate(ds);torch.set_num_threads(cfg.threads);torch.manual_seed(cfg.seed)
        model=make_model(ds,cfg)
        sample=ds.subset('train')[0];obs=ds.packed(sample)
        test=train_step(model,torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate),obs,
                        torch.tensor(ds.elevation,dtype=torch.float32),torch.tensor(ds.land,dtype=torch.float32),
                        target_tensors(ds,sample,ds.step))
        gradients={name:any(p.grad is not None and bool((p.grad!=0).any()) for p in module.parameters())
                   for name,module in model.named_children() if any(True for _ in module.parameters())}
        sensor_gradients={name:any(p.grad is not None and bool((p.grad!=0).any()) for p in module.parameters())
                           for name,module in getattr(model,'satellites',{}).items()}
        report={**describe(model),'data_kind':ds.kind,'training_check':test,'gradient_blocks':gradients,
                'gradient_sensors':sensor_gradients,'test_set_read':False,
                'note':'Градиент подтверждает связность, не качество прогноза. Веса проверки не сохраняются.'}
        if args.output:atomic_json(Path(args.output),report)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))


if __name__=='__main__':main()
