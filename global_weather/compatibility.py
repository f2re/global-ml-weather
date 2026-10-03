"""Explicit local compatibility commands; never starts acquisition or executes inputs."""
import argparse
import json
from pathlib import Path
import subprocess
from .connectors.ecosystem import UPSTREAM, SOURCE_BLOBS, arktika_product, gptl_asset, inspect_satdump, read_json, sha256


def check_checkout(path, upstream):
    """Read Git identity. A source checkout does not identify an installed binary."""
    def git(*args):
        return subprocess.run(['git','-C',str(path),*args],capture_output=True,text=True,
                              timeout=10,check=True).stdout.strip()
    head = git('rev-parse','HEAD'); branch = git('rev-parse','--abbrev-ref','HEAD')
    clean = not git('status','--porcelain','--untracked-files=normal')
    blobs = {}
    for file, expected in SOURCE_BLOBS[upstream].items():
        try: blobs[file] = git('hash-object','--',file)
        except subprocess.CalledProcessError: blobs[file] = None
    matches = blobs == SOURCE_BLOBS[upstream]
    return dict(upstream=upstream, head=head, branch=branch, clean=clean, source_blobs=blobs,
                status='pinned_source_match' if head==UPSTREAM[upstream]['commit'] and branch in (UPSTREAM[upstream]['branch'],'HEAD') and clean and matches else 'review_required',
                installed_binary_verified=False)


def check_release_evidence(path):
    """Mechanical gate on run evidence. Does not certify scientific independence."""
    m = read_json(path)
    required = ('pytest','baseline_72h','adaptive_72h','browser','ecosystem_contracts','physics_review')
    reasons = []
    if m.get('schema') != 'release-evidence-v1': reasons.append('schema')
    commit = m.get('commit','')
    if not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit): reasons.append('commit')
    if not m.get('executor') or not m.get('reviewer') or m['executor']==m['reviewer']: reasons.append('independent_review')
    checks = m.get('checks',{})
    if not isinstance(checks,dict):
        reasons.append('checks_format'); checks = {}
    for key in required:
        item = checks.get(key)
        if not isinstance(item,dict): reasons.append(key+':missing'); continue
        from .connectors.ecosystem import child
        try:
            p = child(Path(path).parent,item['report']); r=read_json(p)
            if sha256(p)!=item['sha256'] or r.get('commit')!=commit or r.get('status')!='passed':
                reasons.append(key+':unverified')
            if r.get('check')!=key: reasons.append(key+':wrong_report')
        except (KeyError,ValueError): reasons.append(key+':invalid_report')
    if m.get('forecast_skill_claimed') is not False: reasons.append('forecast_skill_requires_separate_real_validation')
    return dict(status='blocked' if reasons else 'engineering_evidence_complete', reasons=reasons,
                forecast_quality_certified=False, note='Report hashes are integrity evidence, not proof that a reviewer is independent or that reports are truthful.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('satdump');a.add_argument('input',type=Path);a.add_argument('--require',nargs='*',default=[])
    a=s.add_parser('arktika');a.add_argument('input',type=Path)
    a=s.add_parser('gptl');a.add_argument('input',type=Path);a.add_argument('--raster',type=Path,required=True);a.add_argument('--source',choices=['arktika_m','electro_l'],required=True);a.add_argument('--asset-key');a.add_argument('--channel',type=int)
    for name in ('arktika','gptl'):
        a=s.choices[name];a.add_argument('--available-at',required=True);a.add_argument('--availability-reference',required=True)
        a.add_argument('--geometry',type=Path);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('export');a.add_argument('input',type=Path);a.add_argument('--output',type=Path,required=True);a.add_argument('--max-records',type=int,default=100_000)
    a=s.add_parser('checkout');a.add_argument('input',type=Path);a.add_argument('--upstream',choices=list(UPSTREAM),required=True)
    a=s.add_parser('release-gate');a.add_argument('input',type=Path)
    args=p.parse_args(argv)
    if args.command=='satdump': result=inspect_satdump(args.input,required_instruments=args.require)
    elif args.command in ('arktika','gptl'):
        from .connectors.raster_bridge import import_raster
        spec=arktika_product(args.input) if args.command=='arktika' else gptl_asset(args.input,args.raster,source=args.source,asset_key=args.asset_key,channel=args.channel)
        result=import_raster(spec,args.output,available_at=args.available_at,
                             availability_reference=args.availability_reference,geometry=args.geometry)
    elif args.command=='export':
        from .connectors.raster_bridge import export_observations
        result=export_observations(args.input,args.output,max_records=args.max_records)
    elif args.command=='checkout': result=check_checkout(args.input,args.upstream)
    else: result=check_release_evidence(args.input)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    # Inspection succeeds as an inventory, never as an implicit model-admission test.
    if result.get('status') in ('blocked','review_required'): return 2
    return 0


if __name__=='__main__': raise SystemExit(main())
