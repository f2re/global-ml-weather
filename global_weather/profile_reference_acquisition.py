"""Bounded external-reference acquisition after observation-only model freeze."""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
from pathlib import Path
from .observations import utc
from .observation_training import digest, save
from .providers.era5 import requests_for, retrieve
from .profile_verification import verify


def run(dataset, training, cache, output, *, network=False):
    training, cache = Path(training), Path(cache)
    if not (training/'complete.json').is_file(): raise ValueError('Freeze observation-trained weights first.')
    completion=json.loads((training/'complete.json').read_text())
    if completion.get('status')!='measured_upper_air_research_trained': raise ValueError('Only an observation-trained model is admitted.')
    ref=json.loads((training/'best.json').read_text())
    if completion.get('best_epoch')!=ref['epoch']: raise ValueError('Freeze best epoch before external acquisition.')
    from .profile_training import load_frozen
    model, observed_dataset=load_frozen(dataset,training)
    observed_dataset.verify()
    from .observation_training import checkpoint_path
    checkpoint=checkpoint_path(training,ref); frozen=digest(checkpoint)
    issue=utc('2022-08-01T00:00:00Z')
    days=[(issue+timedelta(days=offset)).date().isoformat() for offset in range(4)]
    plan={'spec':{'source_grid_degrees':2.5,'horizon_hours':72,'step_hours':3},
          'samples':[{'issue_time':issue.isoformat()}],'era5_dates':days}
    cache.mkdir(parents=True,exist_ok=True)
    files={'pressure':[],'surface':[]}; receipts=[]
    for query in requests_for(plan):
        category='pressure' if query['id'].endswith('-pressure') else 'surface' if query['id'].endswith('-surface') else None
        if category is None: continue  # No static fields or precipitation needed for this profile comparison.
        used=sum(path.stat().st_size for path in cache.rglob('*') if path.is_file())
        if used>=8*1024**3: raise RuntimeError('External-reference cache exceeds 8 GiB.')
        paths,receipt=retrieve(query,cache,network=network,max_bytes=min(1024**3,8*1024**3-used))
        files[category].extend(paths); receipts.append(receipt)
        print(query['id'], 'external_reference_received', flush=True)
    if digest(checkpoint)!=frozen or json.loads((training/'best.json').read_text())!=ref:
        raise ValueError('Weights changed during external-reference acquisition.')
    result=verify(dataset,training,files['pressure'],files['surface'],issue,output)
    save(Path(output)/'acquisition.json',{'source_role':'separate_frozen_model_verification_only',
          'checkpoint':ref,'receipts':receipts,'grid_degrees':2.5,'norm_updates':0,'weight_updates':0})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','training','cache','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--allow-network',action='store_true'); args=parser.parse_args(argv)
    run(args.dataset,args.training,args.cache,args.output,network=args.allow_network)


if __name__=='__main__': main()
