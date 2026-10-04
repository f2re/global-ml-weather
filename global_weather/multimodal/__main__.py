"""Явные команды подготовки, загрузки и проверки многомодальной модели."""
import argparse
import json
from .io import write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('selftest');p.add_argument('--output',required=True)
    p=sub.add_parser('demo-dataset');p.add_argument('--output',required=True);p.add_argument('--horizon-hours',type=int,default=72)
    p=sub.add_parser('prepare-capsules');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('fit-sensors');p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('plan-data');p.add_argument('--start',required=True);p.add_argument('--end',required=True)
    p.add_argument('--station',action='append',default=[]);p.add_argument('--output',required=True)
    p=sub.add_parser('acquire');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--network',action='store_true');p.add_argument('--limit',type=int,default=4);p.add_argument('--offset',type=int,default=0)
    args=parser.parse_args(argv)
    if args.action=='selftest':
        from .selftest import run
        report=run(args.output)
    elif args.action=='demo-dataset':
        from .fixture import dataset
        report={'dataset':str(dataset(args.output,horizon_hours=args.horizon_hours)),'data_kind':'synthetic'}
    elif args.action=='prepare-capsules':
        from .prepare import from_capsules
        report=from_capsules(args.plan,args.output)
    elif args.action=='fit-sensors':
        from .normalization import fit_sensors
        report=fit_sensors(args.dataset,args.output)
    elif args.action=='plan-data':
        from .acquisition import plan
        report=plan(args.start,args.end,stations=args.station);write_json(args.output,report)
    else:
        from .acquisition import execute
        report=execute(args.plan,args.output,network=args.network,limit=args.limit,offset=args.offset)
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
