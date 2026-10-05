"""CLI for plans, autonomous experiments and the fixed web worker."""
from __future__ import annotations
import argparse
from pathlib import Path
import signal
from ..pipeline.io import read_json, atomic_json
from .plan import parse_plan
from .execute import execute
from .service import Credentials


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run','worker'])
    p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path);p.add_argument('--job',type=Path);p.add_argument('--credentials',type=Path)
    args=p.parse_args(argv)
    if args.action=='plan':
        if not args.plan or not args.output:p.error('Нужны --plan и --output.')
        atomic_json(args.output,parse_plan(read_json(args.plan)).checked());return
    def interrupt(signum,frame):raise InterruptedError('Исполнение прервано; повторите запуск для продолжения.')
    signal.signal(signal.SIGTERM,interrupt)
    job=args.job
    if args.action=='worker':
        if job is None:p.error('Нужен --job.')
        plan=job/'request.json';output=job/'work'
    else:
        if not args.plan or not args.output:p.error('Нужны --plan и --output.')
        plan,output=args.plan,args.output
    credentials=Credentials(args.credentials).read() if args.credentials else {}
    try:
        execute(plan,output,cds_key=credentials.get('cds_key'),cancelled=(lambda:(job/'cancel').exists()) if job else None)
    except Exception as exc:
        message=str(exc)
        for value in credentials.values():
            if value:message=message.replace(value,'[СКРЫТО]')
        from ..providers._gptl.network import redact
        message=redact(message)
        if job:atomic_json(job/'error.json',{'reason':message,'type':type(exc).__name__})
        raise SystemExit(message) from None


if __name__=='__main__':main()
