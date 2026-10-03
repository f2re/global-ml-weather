"""Read-only pinned-source and mandatory-agent checks; nonzero exit on drift.

This verifies a source checkout, not its installed binary or satellite quality.
No source imports, builds, git fetch, network, settings or credentials are used.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from .project_bridge import local_directory, local_file, load_json, publish_json, output_path

ROLES = ('coordinator','data-steward','radiometry','normalization','physics',
         'model-engineer','executor','verification','release-auditor')
PROTOCOL = 'docs/protocols/05-project-compatibility.md'


def blob_sha(path):
    data = local_file(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def check_source(root, specification):
    root = local_directory(root)
    checks = []
    def git(*args):
        result = subprocess.run(['git','-c','core.fsmonitor=false','-c','core.hooksPath=/dev/null',
                                 '-C',str(root),*args],capture_output=True,text=True,timeout=15,check=False)
        if result.returncode: raise ValueError('Не удалось проверить локальный Git; stderr не публикуется.')
        return result.stdout.strip()
    head = git('rev-parse','HEAD')
    checks.append(dict(check='revision',passed=head==specification['revision'],expected=specification['revision'],actual=head))
    dirty = bool(git('status','--porcelain','--untracked-files=normal'))
    checks.append(dict(check='clean_checkout',passed=not dirty))
    for name, expected in specification['files'].items():
        try: actual=blob_sha(local_file(root/name,root))
        except (OSError,ValueError): actual=None
        checks.append(dict(check='contract_file',file=name,passed=actual==expected,expected=expected,actual=actual))
    return dict(repository=specification['repository'],passed=all(x['passed'] for x in checks),checks=checks,
                scope='source_checkout_only; binary/data not certified')


def check_agent_contracts(root):
    root=local_directory(root); failures=[]
    required=['AGENTS.md','agents/AGENTS.md',PROTOCOL]
    required += ['agents/'+role+'.md' for role in ROLES]
    required += ['.claude/agents/'+role+'.md' for role in ROLES]
    for name in required:
        try:
            p=local_file(root/name,root); content=p.read_text(encoding='utf-8')
            if not content.strip(): failures.append(name+': empty')
        except (OSError,ValueError): failures.append(name+': missing_or_unsafe')
    for role in ('executor','verification','release-auditor'):
        name='agents/'+role+'.md'
        try:
            content=(root/name).read_text(encoding='utf-8')
            for literal in (PROTOCOL,'Стоп-условия','Обязательные проверки','Отчёт'):
                if literal not in content: failures.append(name+': missing '+literal)
        except OSError: pass
    return dict(passed=not failures,failures=failures,
                scope='mandatory instruction presence; not proof an agent complied')


def run_preflight(project_root, *, arktika_source=None, satdump_source=None, source_checker=check_source):
    root=local_directory(project_root)
    lock=load_json(root/'configs/upstream_contracts.json')
    if lock.get('schema')!='global-weather.upstream-contracts/1': raise ValueError('Неизвестный контракт зависимостей.')
    projects={}
    for key,path in (('arktika',arktika_source),('satdump',satdump_source)):
        if path is None:
            projects[key]=dict(passed=False,status='not_checked',reason='Local producer source path required.')
            continue
        try: projects[key]=source_checker(path,lock['projects'][key])
        except (OSError,ValueError,subprocess.TimeoutExpired):
            projects[key]=dict(passed=False,status='failed',reason='Source checkout unavailable or timed out.')
    agents=check_agent_contracts(root)
    return dict(schema='global-weather.preflight/1',time=datetime.now(timezone.utc).isoformat(),
                passed=agents['passed'] and all(p['passed'] for p in projects.values()),
                agents=agents,projects=projects,model_skill_validated=False,live_data_validated=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p.add_argument('--arktika-source',type=Path);p.add_argument('--satdump-source',type=Path)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args(argv)
    for producer in (a.arktika_source,a.satdump_source):
        if producer is not None and output_path(a.report).is_relative_to(local_directory(producer)):
            p.error('Отчёт нельзя писать в исходники производителя.')
    result=run_preflight(a.project_root,arktika_source=a.arktika_source,satdump_source=a.satdump_source)
    publish_json(a.report,result)
    print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
