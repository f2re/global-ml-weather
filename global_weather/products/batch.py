"""Calculate and assemble a causal product context, without silent completion.

Each entry points to a hash-pinned existing product job and geometry. Successful
products remain usable when another source is missing. A failed REQUIRED product
keeps the whole plan blocked; the CLI then returns a nonzero status. No network,
normalization refitting, resampling or learned weights are hidden in this step.
"""
from pathlib import Path
import argparse
import json
import re
from .catalog import CATALOG
from .core import canonical, utc, validate_metadata
from .io import (read_json, resolve, load_product, sha256, exclusive_bytes)


def write_json(path, value):
    exclusive_bytes(path, lambda stream: stream.write((canonical(value)+'\n').encode()))


def _selection(plan):
    keys={'schema','issue_time','data_kind','items','max_records'}
    if (not isinstance(plan,dict) or set(plan)!=keys
            or plan['schema']!='satellite-context-plan-1'
            or plan['data_kind'] not in ('real','synthetic')):
        raise ValueError('Неверная схема плана геофизических продуктов.')
    utc(plan['issue_time'])
    if type(plan['max_records']) is not int or not 1<=plan['max_records']<=100000:
        raise ValueError('Предел контекста: 1–100000 записей.')
    items=plan['items']
    if not isinstance(items,list) or not 1<=len(items)<=32:
        raise ValueError('Нужно 1–32 задания продукции.')
    ids=set()
    for item in items:
        if (not isinstance(item,dict) or set(item)!={'id','job','geometry','required'}
                or not isinstance(item['id'],str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,47}',item['id'])
                or item['id'] in ids or type(item['required']) is not bool):
            raise ValueError('Идентификаторы заданий должны быть уникальны; required — Boolean.')
        ids.add(item['id'])
    return items


def build_context(plan_path, output):
    """Return report and write immutable products, registry and merged JSONL.

    This is calculation/admission preparation, not scientific certification.
    A plan with zero usable records is always blocked, even with optional items.
    """
    from .__main__ import calculate
    from .ingest import export_product
    root=Path(plan_path).absolute().parent
    plan_hash=sha256(plan_path)
    plan=read_json(plan_path); items=_selection(plan)
    target=Path(output).absolute()
    if target.is_symlink() or any(p.is_symlink() for p in target.parents):
        raise ValueError('Выходной каталог не должен содержать символические ссылки.')
    target.mkdir(parents=True,exist_ok=False)
    report=dict(schema='satellite-context-report-1',status='started',
                issue_time=utc(plan['issue_time']).isoformat(),data_kind=plan['data_kind'],
                plan_sha256=plan_hash,items=[],records=0,model_admission_granted=False,
                meteorologically_validated=False,normalization='not_fitted_here',
                dependence_note='Derived products and their parent channels are correlated features, not independent observations.')
    registries={}; files=[]; selected=[]
    try:
        for item in items:
            directory=target/item['id']; directory.mkdir()
            entry={'id':item['id'],'required':item['required'],'status':'blocked'}
            try:
                job_path=resolve(root,item['job'])
                job=read_json(job_path)
                if not isinstance(job,dict): raise ValueError('Задание должно быть объектом JSON.')
                if job.get('data_kind')!=plan['data_kind']:
                    raise ValueError('Происхождение задания отличается от происхождения контекста.')
                if utc(job['available_at'])>utc(plan['issue_time']):
                    raise ValueError('Задание ещё недоступно в момент выпуска.')
                geometry=resolve(root,item['geometry'])
                product_path=directory/'product.npz'
                calculate(job_path,product_path)
                p=load_product(product_path)
                validate_metadata(p.metadata,name=p.name,method=p.method,issue_time=plan['issue_time'])
                age=(utc(plan['issue_time'])-utc(p.metadata['observed_at'])).total_seconds()/3600.
                if not 0<=age<CATALOG[p.name].max_age_hours:
                    raise ValueError('Продукт устарел для зарегистрированной политики.')
                entry.update(product=p.name,method=p.method,units=CATALOG[p.name].units,
                             age_hours=age,valid_pixels=int(p.valid.sum()),
                             invalid_pixels=int((~p.valid).sum()),product_sha256=sha256(product_path))
                remaining=plan['max_records']-report['records']
                if remaining<=0: raise ValueError('Превышен предел записей контекста.')
                exported=export_product(product_path,geometry,directory/'observations.jsonl',max_records=remaining)
                for key,value in exported['registry'].items():
                    if key in registries and registries[key]!=value:
                        raise ValueError('Конфликт метода, глубины или политики возраста в реестре.')
                # Recheck identities after reading and exporting, before assembly.
                resolve(root,item['job']);resolve(root,item['geometry'])
                registries.update(exported['registry'])
                write_json(directory/'registry.json',exported['registry'])
                files.append(directory/'observations.jsonl')
                selected.append(p.metadata)
                report['records']+=exported['records']
                entry.update(status='exported',records=exported['records'],
                             parent_sha256=[d['sha256'] for d in p.metadata['dependencies']])
            except (ValueError,KeyError,OSError,TypeError) as exc:
                entry['reason']=str(exc)[:1500]
                entry['error_type']=type(exc).__name__
            report['items'].append(entry)
        if sha256(plan_path)!=plan_hash:
            raise ValueError('План изменился во время расчёта.')
        record_hashes={}; unique_count=0
        def merge(stream):
            nonlocal unique_count
            for file in files:
                with file.open('rb') as source:
                    for line in source:
                        r=json.loads(line)
                        identity=(r['source'],r['observation_id'],r['revision'],r['available_at'])
                        from .core import digest
                        value=digest(r)
                        if identity in record_hashes:
                            if record_hashes[identity]!=value:
                                raise ValueError('Две разные записи имеют одну идентичность версии.')
                            continue
                        record_hashes[identity]=value;unique_count+=1
                        stream.write(line)
                        if stream.tell()>256*1024**2:
                            raise ValueError('JSONL контекста превышает 256 МиБ.')
        exclusive_bytes(target/'derived.jsonl',merge)
        report['records']=unique_count
        write_json(target/'registry.json',registries)
        write_json(target/'dependencies.json',selected)
        failed=[x for x in report['items'] if x['status']!='exported']
        required_failed=any(x['required'] for x in failed)
        report['status']='blocked' if required_failed or not unique_count else ('partial' if failed else 'prepared')
        report['artifacts']={name:sha256(target/name) for name in ('derived.jsonl','registry.json','dependencies.json')}
        write_json(target/'report.json',report)
        return report
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__)
        if not (target/'report.json').exists():write_json(target/'report.json',report)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    report=build_context(args.plan,args.output)
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    if report['status']=='blocked': raise SystemExit(2)


if __name__=='__main__':main()
