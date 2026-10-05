"""Bounded, as-issued product preparation. No network, constants or hidden fill.

A prepared snapshot is input to the existing observation contract, not an
operational forecast approval. Recipe failures remain separate from data gaps.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re
from zipfile import BadZipFile
import numpy as np
from .catalog import CATALOG
from .core import canonical, utc, validate_metadata, QC
from .io import read_json, regular, resolve, load_field, load_product, exclusive_bytes, sha256

SCHEMA = 'satellite-product-batch-1'
MAX_BYTES = 64*1024**2
MAX_ITEMS = 64
MAX_RECORDS = 1_000_000


class PreparationError(ValueError):
    """An expected input/admission failure with a stable machine-readable code."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _json(path, value):
    exclusive_bytes(path, lambda f:f.write((canonical(value)+'\n').encode()))


def _load_plan(path):
    path = regular(path)
    plan = read_json(path)
    needed = {'schema', 'issue_time', 'data_kind', 'items'}
    if not isinstance(plan,dict) or not needed.issubset(plan) or set(plan)-needed-{'max_records'}:
        raise ValueError('Неверная структура плана продукции.')
    if plan['schema'] != SCHEMA or plan['data_kind'] not in ('real','synthetic'):
        raise ValueError('Неизвестная схема или происхождение данных.')
    utc(plan['issue_time'])
    if not isinstance(plan['items'],list) or not 1 <= len(plan['items']) <= MAX_ITEMS:
        raise ValueError('План должен содержать 1–64 задания.')
    count = plan.get('max_records',100000)
    if type(count) is not int or not 1 <= count <= MAX_RECORDS:
        raise ValueError('Неверный предел числа записей.')
    seen = set()
    for item in plan['items']:
        keys = {'id','product','job','geometry','required'}
        if not isinstance(item,dict) or not keys.issubset(item) or set(item)-keys-{'history_hours','min_valid_fraction','variable'}:
            raise ValueError('Неверная структура задания в плане.')
        name = item['id']
        if not isinstance(name,str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}',name) or name in seen:
            raise ValueError('Идентификаторы заданий должны быть безопасными и уникальными.')
        seen.add(name)
        if item['product'] not in CATALOG or type(item['required']) is not bool:
            raise ValueError('Нужны известный продукт и Boolean-признак обязательности.')
        age = item.get('history_hours',CATALOG[item['product']].max_age_hours)
        fraction = item.get('min_valid_fraction',0.)
        if type(age) is not int or not 0 < age <= CATALOG[item['product']].max_age_hours:
            raise ValueError('Неверная политика возраста продукции.')
        if type(fraction) not in (int,float) or not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Неверная минимальная доля пригодных пикселей.')
        if 'variable' in item and (not isinstance(item['variable'],str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,256}',item['variable'])):
            raise ValueError('Неверное имя переменной.')
    return path, plan


def _inspect(item, root, plan):
    """Check immutable dependencies. Retrieval-specific physics is checked at run."""
    from .__main__ import REQUIRED, OPTIONAL_INPUTS
    path = resolve(root,item['job'])
    job = read_json(path)
    required = {'schema','product','inputs','available_at','availability_reference','data_kind'}
    if (not isinstance(job,dict) or not required.issubset(job) or set(job)-required-{'lut','parameters'}
            or job['schema'] != 'satellite-product-job-1' or job['product'] != item['product']
            or job['data_kind'] != plan['data_kind']):
        raise PreparationError('recipe_mismatch','Задание, продукт или происхождение не соответствуют плану.')
    if not isinstance(job['availability_reference'],str) or not job['availability_reference'].strip():
        raise PreparationError('availability_unknown','Нет доказательства времени готовности.')
    if utc(job['available_at']) > utc(plan['issue_time']):
        raise PreparationError('not_available','Продукт ещё не был доступен при этом выпуске.')
    if (not isinstance(job['inputs'],dict) or not set(REQUIRED[item['product']]).issubset(job['inputs'])
            or set(job['inputs'])-set(REQUIRED[item['product']])-OPTIONAL_INPUTS.get(item['product'],set())):
        raise PreparationError('missing_dependencies','Состав необходимых входов не соответствует методу.')
    refs = []
    fields = {}
    for key,ref in job['inputs'].items():
        fp = resolve(path.parent,ref)
        field = load_field(fp)
        if plan['data_kind']=='real' and field.metadata.get('data_kind')!='real':
            raise PreparationError('data_kind_mismatch','Реальное задание содержит непроверенный или синтетический вход.')
        if utc(field.available_at)>utc(job['available_at']):
            raise PreparationError('dependency_not_available','Задание готово раньше необходимого входа.')
        # Composite intervals must be explicit; reprocessing never refreshes them.
        support = field.metadata.get('temporal_support')
        if support is not None:
            if not isinstance(support,dict) or set(support)!={'start','end'}:
                raise PreparationError('temporal_support','Нужны начало и конец временной поддержки.')
            start,end = utc(support['start']),utc(support['end'])
            if not start <= utc(field.observed_at) <= end <= utc(field.available_at):
                raise PreparationError('temporal_support','Нарушена временная поддержка входа.')
        elif field.metadata.get('composite') is True:
            raise PreparationError('temporal_support','Для составного продукта не задан временной интервал.')
        fields[key] = field
        refs.append((str(fp),ref['sha256']))
    primary = fields[REQUIRED[item['product']][0]]
    age = item.get('history_hours',CATALOG[item['product']].max_age_hours)
    if not utc(plan['issue_time'])-timedelta(hours=age) < utc(primary.observed_at) <= utc(plan['issue_time']):
        raise PreparationError('stale_product','Продукт вне собственного допустимого возраста.')
    if item['product']=='soil_moisture_surface':
        if 'lut' not in job:
            raise PreparationError('missing_calibration','Для влажности почвы нужна приборная таблица.')
        lp = resolve(path.parent,job['lut'])
        refs.append((str(lp),job['lut']['sha256']))
    elif 'lut' in job:
        raise PreparationError('unexpected_calibration','Метод не использует эту таблицу.')
    gp = resolve(root,item['geometry'])
    refs.extend(((str(path),item['job']['sha256']), (str(gp),item['geometry']['sha256'])))
    return path,gp,refs


def _error(exc):
    if isinstance(exc,PreparationError): return exc.code,str(exc)
    if isinstance(exc,(FileNotFoundError,NotADirectoryError)): return 'missing_file','Не найден обязательный локальный файл.'
    if isinstance(exc,ValueError) and 'обычный локальный файл' in str(exc):
        return 'missing_or_unsafe_file','Файл отсутствует либо путь содержит символическую ссылку.'
    # No traceback, file payload or secrets in the machine-readable report.
    return 'invalid_input',str(exc)[:500]


def preflight(plan_path):
    path,plan = _load_plan(plan_path)
    items = []
    for item in plan['items']:
        report = dict(id=item['id'],product=item['product'],required=item['required'])
        try:
            _,_,refs = _inspect(item,path.parent,plan)
            report.update(status='inputs_ready',dependencies=len(refs),retrieval_validated=False)
        except (ValueError,KeyError,TypeError,OSError,BadZipFile) as exc:
            code,message = _error(exc)
            report.update(status='blocked',reason=code,message=message)
        items.append(report)
    blocked = any(i['required'] and i['status']=='blocked' for i in items)
    any_ready = any(i['status']=='inputs_ready' for i in items)
    return dict(schema=SCHEMA,status='blocked' if blocked or not any_ready else 'inputs_ready',
                issue_time=utc(plan['issue_time']).isoformat(),data_kind=plan['data_kind'],
                plan_sha256=sha256(path),items=items,model_ready=False,
                note='Проверка зависимостей не заменяет расчёт, радиометрию и нормы.')


def run_batch(plan_path, output):
    """Calculate all available recipes; publish combined inputs only at commit.

    Mandatory failures withhold the aggregate; optional failures are reported.
    Individual diagnostic products may remain. Repeating into the same output
    directory is prohibited. This command does not establish sensor calibration.
    """
    from .__main__ import calculate
    from .ingest import export_product,check_record
    from ..observations import Variable
    path,plan = _load_plan(plan_path)
    plan_hash = sha256(path)
    out = Path(output).absolute()
    if out.is_symlink() or any(p.is_symlink() for p in out.parents):
        raise ValueError('Символические ссылки в выходном пути запрещены.')
    out.mkdir(parents=True,exist_ok=False)
    _json(out/'started.json',dict(schema=SCHEMA,plan_sha256=plan_hash,issue_time=plan['issue_time'],
          execution_time_utc=datetime.now(timezone.utc).isoformat(),status='running'))
    reports,accepted,dependencies,artifact_refs = [],[],{},[]
    total_records = 0
    for item in plan['items']:
        entry = dict(id=item['id'],product=item['product'],required=item['required'],status='blocked')
        reports.append(entry)
        try:
            jp,gp,refs = _inspect(item,path.parent,plan)
            dest = out/item['id'];dest.mkdir()
            calc = calculate(jp,dest/'product.npz')
            product = load_product(dest/'product.npz')
            validate_metadata(product.metadata,name=product.name,method=product.method,issue_time=plan['issue_time'])
            count = int(product.valid.sum());fraction = count/product.valid.size
            entry.update(method=product.method,valid_pixels=count,total_pixels=product.valid.size,
                         valid_fraction=fraction,units=CATALOG[product.name].units,
                         uncertainty_known_pixels=int(np.count_nonzero(product.valid & np.isfinite(product.uncertainty))),
                         qc_counts={flag.name:int(np.count_nonzero(product.qc & int(flag))) for flag in QC},
                         product_path=f'{item["id"]}/product.npz',product_sha256=calc['sha256'])
            if not count or fraction < item.get('min_valid_fraction',0.):
                raise PreparationError('insufficient_valid_pixels','Не выполнено условие пригодного покрытия.')
            export = export_product(dest/'product.npz',gp,dest/'observations.jsonl',
                      variable=item.get('variable'),history_hours=item.get('history_hours'),
                      max_records=plan.get('max_records',100000),max_output_bytes=MAX_BYTES)
            # Recheck age at issue, not only at the product's own ready timestamp.
            for name,spec in export['registry'].items():
                var=Variable(**spec)
                with (dest/'observations.jsonl').open(encoding='utf-8') as stream:
                    for line in stream: check_record(json.loads(line),var,plan['issue_time'])
            if any(sha256(p)!=checksum for p,checksum in refs):
                raise PreparationError('input_changed','Вход изменился во время обработки.')
            if total_records + export['records'] > plan.get('max_records',100000):
                raise PreparationError('record_limit','Общий предел числа записей превышен.')
            total_records += export['records']
            accepted.append((dest/'observations.jsonl',export['registry']))
            artifact_refs.extend(refs)
            entry.update(status='exported',records=export['records'],history_hours=next(iter(export['registry'].values()))['history_hours'])
            for dep in product.metadata['dependencies']:
                dependencies.setdefault(dep['sha256'],set()).add(item['id'])
        except (ValueError,KeyError,TypeError,OSError,BadZipFile) as exc:
            code,message = _error(exc);entry.update(status='blocked',reason=code,message=message)
    mandatory_failed = any(i['required'] and i['status']!='exported' for i in reports)
    report = dict(schema=SCHEMA,issue_time=utc(plan['issue_time']).isoformat(),
                  data_kind=plan['data_kind'],plan_sha256=plan_hash,items=reports,
                  meteorologically_validated=False,model_ready=False,
                  shared_dependencies={k:sorted(v) for k,v in dependencies.items() if len(v)>1},
                  dependence_policy='products_and_parent_channels_are_not_independent',
                  use='initial_state_features_only; no future products are injected',
                  physical_boundary_model=False,normalization_required=True)
    if mandatory_failed or not accepted:
        report['status']='blocked'
        report['reason']='required_product_failed' if mandatory_failed else 'no_usable_products'
    else:
        try:
            if sha256(path)!=plan_hash or any(sha256(p)!=h for p,h in artifact_refs):
                raise PreparationError('input_changed','План или вход изменился до фиксации снимка.')
            registry,unique,lines = {},{},[]
            written = 0;duplicates = 0
            for exported,specs in accepted:
                for name,spec in specs.items():
                    if name in registry and registry[name]!=spec:
                        raise PreparationError('registry_conflict','Одинаковое имя имеет разные методы, глубины или нормы.')
                    registry[name]=spec
                with exported.open('rb') as stream:
                    for line in stream:
                        rec=json.loads(line);key=(rec['source'],rec['observation_id'],rec['revision'])
                        text=canonical(rec)
                        if key in unique:
                            if unique[key]!=text:
                                raise PreparationError('revision_conflict','Одна версия наблюдения содержит разные значения.')
                            duplicates+=1;continue
                        unique[key]=text
                        written+=len(line)
                        if written>MAX_BYTES:
                            raise PreparationError('output_limit','Объём общего снимка превышен.')
                        lines.append(line)
            _json(out/'registry.json',registry)
            exclusive_bytes(out/'observations.jsonl',lambda stream:stream.writelines(lines))
            report.update(status='prepared_partial' if any(i['status']=='blocked' for i in reports) else 'prepared',
                          records=len(lines),deduplicated=duplicates,
                          artifacts={n:sha256(out/n) for n in ('registry.json','observations.jsonl')},
                          missing_optional=[i['id'] for i in reports if i['status']=='blocked'])
            # The snapshot is the commit marker. Consumers MUST require it.
            _json(out/'snapshot.json',dict(schema='satellite-product-snapshot-1',
                  issue_time=report['issue_time'],data_kind=plan['data_kind'],
                  plan_sha256=plan_hash,records=len(lines),
                  observations={'path':'observations.jsonl','sha256':report['artifacts']['observations.jsonl']},
                  registry={'path':'registry.json','sha256':report['artifacts']['registry.json']},
                  normalization_required=True,meteorologically_validated=False))
        except (ValueError,KeyError,TypeError,OSError,BadZipFile) as exc:
            code,message=_error(exc);report.update(status='blocked',reason=code,message=message)
    _json(out/'report.json',report)
    return report


def load_snapshot(path):
    """Verify a completed snapshot before joining it to an existing data sample."""
    p=regular(path);m=read_json(p)
    if (m.get('schema')!='satellite-product-snapshot-1' or m.get('data_kind') not in ('real','synthetic')
            or m.get('normalization_required') is not True):
        raise ValueError('Неверный снимок спутниковой продукции.')
    from ..observations import read_jsonl,read_variables
    op,rp=resolve(p.parent,m['observations']),resolve(p.parent,m['registry'])
    records=read_jsonl(op);registry=read_variables(rp)
    if len(records)!=m['records']:raise ValueError('Число записей изменилось.')
    from .ingest import check_record
    for record in records:
        check_record(record,registry[record['variable']],m['issue_time'])
        if record['derivation']['data_kind']!=m['data_kind']:
            raise ValueError('Происхождение записи не соответствует снимку.')
    return records,registry,m


def pack_snapshot(path, grid, pressure_pa, *, normalization=None,
                  records=(), variables=None, allow_synthetic_unscaled=False):
    """Join a product snapshot to station/channel inputs using the existing gate.

    Returns PackedObservations and the combined registry. A missing product
    normalisation, incompatible vocabulary or rejected footprint stops admission.
    Caller must use an architecture/checkpoint trained with the returned schema.
    """
    from ..observations import pack_observations
    derived, registry, meta = load_snapshot(path)
    if normalization is None and not (allow_synthetic_unscaled is True and meta['data_kind']=='synthetic'):
        raise ValueError('Нужны фиксированные нормы каждой продукции; инженерные масштабы запрещены.')
    checked = pack_observations(derived, grid, pressure_pa, meta['issue_time'], registry,
                                normalization=normalization)
    if checked.accepted_records != len(derived):
        raise ValueError('Часть продукции не прошла допуск модели: '+canonical(checked.rejected))
    combined = dict(variables or {})
    for key,value in registry.items():
        if key in combined and combined[key] != value:
            raise ValueError('Переменная продукции конфликтует с реестром наблюдений.')
        combined[key] = value
    packed = pack_observations([*records, *derived],grid,pressure_pa,meta['issue_time'],combined,
                               normalization=normalization)
    return packed, combined
