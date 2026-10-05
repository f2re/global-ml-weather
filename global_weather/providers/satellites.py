"""GPTL navigation and bounded local downloads; no inferred physical admission."""
from __future__ import annotations
from datetime import timedelta, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from urllib.parse import urlsplit, urlunsplit, urlencode
from ._gptl.auth import AuthClient
from ._gptl.network import STAC, NetworkError, redact
from ._gptl.model import normalize_asset, level_of, parse_time, platform_of
from ..pipeline.dataset import utc
from ..pipeline.io import atomic_json, sha256


class SatelliteCatalog:
    def __init__(self, cache, root=None, credentials=None, save_credentials=None):
        self.cache=Path(cache);self.root=Path(root).absolute() if root else None
        self.credentials=credentials or (lambda:{})
        self.save_credentials=save_credentials
        self.assets={};self.local={};self.lock=threading.Lock()

    def client(self):
        c=self.credentials()
        save=self.save_credentials
        class StoredAuth(AuthClient):
            def exchange(self, fields):
                result=super().exchange(fields)
                if save:save({'gptl_token':self.token,'gptl_refresh':self.refresh_token})
                return result
        client=StoredAuth(token=c.get('gptl_token',''),oauth={'client_id':c.get('gptl_client_id',''),'redirect_uri':c.get('gptl_redirect_uri','')})
        if c.get('gptl_refresh') and c.get('gptl_client_id'):
            client.replace(c.get('gptl_token',''),c['gptl_refresh'])
        return client

    def collections(self, *, network=False):
        if network is not True:raise ValueError('Разрешите сетевой просмотр каталога.')
        try:
            result=self.client().api_json(STAC.rsplit('/',1)[0]+'/collections')
        except NetworkError as exc:raise ValueError(redact(exc)) from None
        rows=result.get('collections')
        if not isinstance(rows,list):raise ValueError('Каталог не предоставил перечень коллекций.')
        return [{'id':r.get('id'),'title':r.get('title'),
                 'platforms':r.get('summaries',{}).get('platform',[])} for r in rows[:200] if isinstance(r,dict)]

    def search(self, *, start, end, platform='', collection='', network=False, max_pages=3):
        if network is not True:raise ValueError('Разрешите сетевой поиск спутниковых данных.')
        a,b=utc(start),utc(end)
        if not a<b or b-a>timedelta(days=7):raise ValueError('Поиск ограничен семью сутками.')
        for value in (platform,collection):
            if value and not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',value):raise ValueError('Неверный идентификатор каталога.')
        if not platform and not collection:raise ValueError('Выберите аппарат или коллекцию из каталога GPTL.')
        if type(max_pages) is not int or not 1<=max_pages<=10:raise ValueError('Неверный предел страниц.')
        params={'datetime':a.isoformat()+'/'+b.isoformat(),'limit':100}
        if platform:params['platforms']=platform
        if collection:params['collections']=collection
        url=STAC+'?'+urlencode(params);method='GET';body=None;seen=set();items=[];truncated=False
        client=self.client();assets={}
        try:
            for page in range(max_pages):
                signature=(url,json.dumps(body,sort_keys=True))
                if signature in seen:raise ValueError('Каталог повторил страницу.')
                seen.add(signature)
                result=client.api_json(url,method,body)
                if not isinstance(result.get('features'),list):raise ValueError('Нет массива features в ответе STAC.')
                for item in result['features']:
                    props=item.get('properties',{});stamp=props.get('datetime') or props.get('start_datetime')
                    date,assumed=parse_time(stamp)
                    actual=platform_of(item)
                    if not date or not a<=utc(date)<b or platform and actual!=platform:continue
                    context={'id':item.get('id'),'platform':actual,'time':date,'time_original':stamp,
                             'time_assumed':assumed,'level':level_of(item),'source':'stac'}
                    for key,raw in (item.get('assets') or {}).items():
                        asset=normalize_asset(raw,context,key)
                        if not asset:continue
                        client.validate(asset['uri'] if not asset['uri'].startswith('s3:') else 'https://s3.gptl.ru')
                        asset['queried_at']=datetime.now(timezone.utc).isoformat();assets[asset['id']]=asset
                next_link=next((l for l in result.get('links',[]) if l.get('rel')=='next'),None)
                if not next_link:break
                from urllib.parse import urljoin
                url=urljoin(url,next_link['href']);method=next_link.get('method','GET').upper();body=next_link.get('body')
                if method not in ('GET','POST'):raise ValueError('Неподдерживаемая пагинация STAC.')
                if next_link.get('merge') and body is not None:body=dict(params,**body)
            else:truncated=bool(next_link)
        except NetworkError as exc:raise ValueError(redact(exc)) from None
        with self.lock:
            self.assets=assets  # Signed URLs stay in memory and never enter the UI/log.
        return {'items':[self.public(a) for a in assets.values()],'truncated':truncated,
                'note':'Каталог и загрузка не подтверждают радиометрический допуск.'}

    @staticmethod
    def public(a):
        return {k:a.get(k) for k in ('id','platform','time','channel','level','category','size','filename')} | {'model_ready':False}

    def download(self, identity, *, network=False, max_bytes=512*1024**2):
        if network is not True:raise ValueError('Разрешите скачивание выбранного файла.')
        with self.lock:asset=dict(self.assets.get(identity) or {})
        if not asset:raise ValueError('Обновите каталог: выбранный ресурс отсутствует в текущем поиске.')
        directory=self.cache/identity
        if any(p.is_symlink() for p in (directory,*directory.parents)):raise ValueError('Ссылка в каталоге загрузки.')
        receipt=directory/'receipt.json'
        if receipt.exists():
            old=json.loads(receipt.read_text());path=directory/'source.tif'
            if sha256(path)!=old['sha256']:raise ValueError('Скачанный файл повреждён.')
            return old
        if directory.exists():raise ValueError('Незавершённая загрузка без паспорта.')
        if asset.get('size') and asset['size']>max_bytes:raise ValueError('Снимок превышает предел 512 МиБ; используйте ограниченную сцену.')
        if asset['category'] not in ('channel','rgb'):
            raise ValueError('Этот формат не поддерживается загрузчиком научных продуктов.')
        self.cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.satellite-',dir=self.cache) as tmp:
            tmp=Path(tmp);source=tmp/'source.tif';client=self.client();total=0
            try:
                client.refresh()
                with client.open_object(asset['uri']) as response,source.open('xb') as out:
                    while block:=response.read(256*1024):
                        total+=len(block)
                        if total>max_bytes:raise ValueError('Превышен предел спутникового файла.')
                        out.write(block)
                    out.flush();os.fsync(out.fileno())
            except NetworkError as exc:raise ValueError(redact(exc)) from None
            if not total:raise ValueError('Хранилище вернуло пустой файл.')
            if asset.get('size') is not None and total!=asset['size']:raise ValueError('Размер файла отличается от каталога.')
            import rasterio
            with rasterio.open(source) as ds:
                if ds.driver != 'GTiff' or ds.crs is None:
                    raise ValueError('Получен не геопривязанный GeoTIFF.')
            checksum=sha256(source);stamp=datetime.now(timezone.utc).isoformat()
            p=urlsplit(asset['uri']);clean=dict(asset,uri=urlunsplit((p.scheme,p.netloc,p.path,'','')))
            atomic_json(tmp/'asset.json',clean)
            atomic_json(tmp/'source.tif.download.json',{'asset_id':asset['id'],'size':total,'sha256':checksum,'time':stamp})
            row={'id':identity,'bytes':total,'sha256':checksum,'acquired_at':stamp,'model_ready':False,
                 'status':'downloaded_not_physically_admitted','level':asset['level'],'channel':asset['channel']}
            atomic_json(tmp/'receipt.json',row);tmp.rename(directory)
        return row

    def browse_local(self, *, day='', platform='', limit=100):
        if self.root is None:raise ValueError('Каталог arktika-worker задаётся при запуске: --arktika-root ПУТЬ.')
        if any(p.is_symlink() for p in (self.root,*self.root.parents)) or not self.root.is_dir():raise ValueError('Нет разрешённого каталога arktika-worker.')
        if day:
            from datetime import date
            date.fromisoformat(day)
        rows=[];paths={};scanned=0;truncated=False
        for directory,folders,files in os.walk(self.root,followlinks=False):
            depth=len(Path(directory).relative_to(self.root).parts)
            folders[:]=[f for f in folders if not f.startswith('.') and not (Path(directory)/f).is_symlink()] if depth<6 else []
            scanned+=len(files)
            if scanned>20000:truncated=True;break
            if 'product.json' not in files:continue
            path=Path(directory)/'product.json'
            if path.is_symlink() or path.stat().st_size>8*1024**2:continue
            try:
                m=json.loads(path.read_text());stamp=m.get('time','');sat=m.get('platform','')
                if day and not stamp.startswith(day) or platform and sat!=platform:continue
                if not isinstance(m.get('request'),dict):continue
                identity=hashlib.sha256(str(path.relative_to(self.root)).encode()).hexdigest()[:32]
                rows.append({'id':identity,'time':stamp,'platform':sat,'channel':m['request'].get('channel'),
                             'category':m.get('product'),'calibration_status':m.get('calibration_status'),
                             'filename':str(path.relative_to(self.root)),'model_ready':False})
                paths[identity]=path
                if len(rows)>=min(200,max(1,limit)):truncated=True;break
            except (ValueError,TypeError,KeyError):continue
        with self.lock:self.local=paths
        return {'items':sorted(rows,key=lambda x:x['time'],reverse=True),'truncated':truncated,
                'note':'Экспортированные продукты arktika-worker; исходные файлы остаются без изменений.'}

    def import_local(self, identity):
        with self.lock:path=self.local.get(identity)
        if path is None:raise ValueError('Сначала обновите локальный каталог.')
        from ..connectors.ecosystem import arktika_product
        spec=arktika_product(path)  # Rejects assumed times/calibration and nonphysical composites.
        directory=self.cache/('local-'+identity)
        if directory.exists():raise ValueError('Продукт уже импортирован.')
        self.cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.local-',dir=self.cache) as tmp:
            tmp=Path(tmp);records=[]
            for p in (path,spec['raster'],spec['quality']):
                if not p.resolve().is_relative_to(self.root.resolve()) or p.stat().st_size>512*1024**2:raise ValueError('Файл вне разрешённого каталога или слишком велик.')
                before=sha256(p);shutil.copyfile(p,tmp/p.name)
                if sha256(p)!=before or sha256(tmp/p.name)!=before:raise ValueError('Исходный продукт изменился во время чтения.')
                records.append({'name':p.name,'sha256':before})
            row={'id':identity,'status':'imported_physical_metadata','model_ready':False,'files':records,
                 'pending':['view_geometry','sensor_normalization','model_admission']}
            atomic_json(tmp/'receipt.json',row);tmp.rename(directory)
        return row
