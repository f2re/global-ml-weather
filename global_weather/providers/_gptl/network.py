"""GPTL transport: TLS, STAC Bearer, STS/S3 signing, public/presigned access.

Only standard-library dependencies. Public fallback is an ordinary anonymous GET,
not an attempt to change permissions. One access path is used for probes/previews/downloads.
"""
import base64
import datetime as dt
import hashlib
import hmac
import http.client
import io
import json
import math
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit, urlencode, quote, unquote, urljoin, parse_qsl
from .model import UTC, now, iso, safe_text, normalize_item

API = 'https://api.gptl.ru'
STAC = API + '/stac/api/v1/catalogs/roscosmos-opendata/search'
LEGACY = API + '/catalog/v2/search'
S3 = 'https://s3.gptl.ru'
REGION = 'ext-dc1'
S3_HOSTS = {'s3.gptl.ru', 's3ext.gptl.ru'}
TRUSTED_HOSTS = S3_HOSTS | {'api.gptl.ru', 'gptl.ru', 'www.gptl.ru', 'vtms.gptl.ru'}
# Bounded in-memory redaction registry; never persisted.
SECRETS = deque(maxlen=256)

class Cancelled(Exception):
    pass

class NetworkError(Exception):
    def __init__(self, message, status=0, code='', attempts=None):
        super().__init__(message)
        self.status, self.code = status, code
        self.attempts = attempts or []

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def redact(value):
    s = safe_text(value)
    for secret in sorted(set(SECRETS),key=len,reverse=True):
        if secret:
            s = s.replace(secret, '[СКРЫТО]')
    s = re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '[JWT]', s)
    s = re.sub(r'(?:https?|s3)://\S+', '[URL]', s)
    s = re.sub(r'(?i)(Bearer\s+|(?:AccessKeyId|SecretAccessKey|SessionToken|WebIdentityToken)[\s:=]+)\S+', r'\1[СКРЫТО]', s)
    return s[:1000]


def qurl(url, values):
    p = urlsplit(url)
    pairs = parse_qsl(p.query, keep_blank_values=True) + list(values.items())
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(pairs, quote_via=quote), ''))


def xml_value(root, name):
    return next((e.text or '' for e in root.iter() if e.tag.split('}')[-1] == name), '')


def error_code(body):
    try:
        root = ET.fromstring(body)
        return xml_value(root, 'Code'), xml_value(root, 'Message')
    except (ET.ParseError, ValueError):
        return '', ''


def error_hint(status, code, mode=''):
    hints = {
        'AccessDenied': 'Хранилище отказало в чтении этого объекта. Доступ к каталогу не равен праву скачивания.',
        'ExpiredToken': 'Истекла временная сессия S3. Повторное получение сессии не помогло; замените Bearer.',
        'InvalidToken': 'Хранилище отклонило временный токен S3. Переподключите Bearer.',
        'InvalidAccessKeyId': 'Хранилище не принимает временный ключ S3. Переподключите Bearer.',
        'SignatureDoesNotMatch': 'Не совпала подпись запроса S3. Проверьте регион, часы и прокси; сохраните диагностику.',
        'RequestTimeTooSkewed': 'Часы компьютера расходятся с часами S3. Синхронизируйте время системы.',
        'RequestExpired': 'Запрос или подписанная ссылка устарели. Обновите каталог и повторите.',
        'NoSuchKey': 'Файл отсутствует по адресу из каталога. Обновите дату в каталоге.',
        'NoSuchBucket': 'Хранилище не нашло указанный бакет.',
        'AuthorizationHeaderMalformed': 'Хранилище не приняло регион или формат авторизации S3.',
        'PreconditionFailed': 'Объект изменился после начала загрузки. Части разных версий не объединены.',
    }
    hint = hints.get(code)
    if not hint:
        hint = {401:'Авторизация не принята. Вставьте новый Bearer.',
                403:'Доступ отклонён. Полный код S3 отсутствует; возможен ответ шлюза или ограничение прав.',
                404:'Объект не найден. Обновите каталог.',
                500:'Ошибка сервера GPTL, а не отсутствие снимков.',
                429:'Сервис временно ограничил частоту запросов.'}.get(status, 'Сервис не выполнил запрос.')
    if mode == 'presigned' and status in (401, 403):
        hint += ' Для подписанной ссылки обновите её в каталоге; Bearer к ней не добавляется.'
    return 'HTTP {}{}: {}'.format(status, ' / '+code if code else '', hint)


def sign(url, creds, region=REGION, extra=None, clock=None):
    """S3 Signature V4 without path normalisation or double percent-encoding."""
    clock = clock or now()
    stamp, day = clock.strftime('%Y%m%dT%H%M%SZ'), clock.strftime('%Y%m%d')
    p = urlsplit(url)
    empty_hash = hashlib.sha256(b'').hexdigest()
    headers = {'host': p.netloc, 'x-amz-date': stamp, 'x-amz-content-sha256': empty_hash}
    if creds.get('SessionToken'):
        headers['x-amz-security-token'] = creds['SessionToken']
    headers.update({k.lower(): str(v) for k, v in (extra or {}).items()})
    query = '&'.join(k+'='+v for k,v in sorted((part.partition('=')[0],part.partition('=')[2]) for part in p.query.split('&') if part))
    keys = sorted(headers)
    canonical = '\n'.join(['GET', p.path or '/', query,
                           ''.join(k+':'+ ' '.join(headers[k].split())+'\n' for k in keys),
                           ';'.join(keys), empty_hash])
    scope = day+'/'+region+'/s3/aws4_request'
    value = '\n'.join(['AWS4-HMAC-SHA256', stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = ('AWS4'+creds['SecretAccessKey']).encode()
    for part in (day,region,'s3','aws4_request'):
        key=hmac.new(key,part.encode(),hashlib.sha256).digest()
    signature=hmac.new(key,value.encode(),hashlib.sha256).hexdigest()
    headers['Authorization']='AWS4-HMAC-SHA256 Credential='+creds['AccessKeyId']+'/'+scope+', SignedHeaders='+';'.join(keys)+', Signature='+signature
    return headers


def s3_parts(uri):
    p=urlsplit(uri)
    if p.scheme=='s3':
        return p.netloc,unquote(p.path.lstrip('/'))
    if p.scheme=='https' and p.hostname in S3_HOSTS and not p.query:
        bucket,_,key=p.path.lstrip('/').partition('/')
        return unquote(bucket),unquote(key)
    return None


def s3_url(bucket, key='', endpoint=S3):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,254}',bucket):
        raise ValueError('Недопустимое имя бакета.')
    return endpoint.rstrip('/')+'/'+quote(bucket,safe='')+'/'+quote(key,safe='/~')


class Response:
    def __init__(self, raw):
        self.raw=raw
        self.status=raw.code
        self.headers={k.lower():v for k,v in raw.headers.items()}
        self.mode=''
        self.attempts=[]
    def read(self, n=-1):
        return self.raw.read(n)
    def close(self):
        self.raw.close()
    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.close()


class Transport:
    """System CA verification remains enabled; redirects are evaluated by Client."""
    def __init__(self):
        self.opener=urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    def open(self,url,headers=None,method='GET',body=None,timeout=45):
        if isinstance(body,str):
            body=body.encode('utf-8')
        req=urllib.request.Request(url,data=body,method=method,headers=headers or {})
        req.add_header('User-Agent','Arktika-Web/2.0')
        req.add_header('Accept-Encoding','identity')
        try:
            return Response(self.opener.open(req,timeout=timeout))
        except urllib.error.HTTPError as e:
            return Response(e)
        except (urllib.error.URLError,OSError,http.client.HTTPException) as e:
            if 'CERTIFICATE_VERIFY_FAILED' in str(e):
                msg='Ошибка проверки TLS-сертификата. Установите доверенные CA для Python/системы. Проверка TLS не отключена.'
            else:
                msg='Сетевая ошибка: '+redact(e)
            raise NetworkError(msg) from e


class Client:
    def __init__(self,token='',transport=None,test_hosts=None):
        self.http=transport or Transport()
        self.lock=threading.RLock()
        self.sts_lock=threading.Lock()
        self.test_hosts=set(test_hosts or [])  # only injected by tests, no web API for this
        self.token=''; self.credentials=None; self.cred_expiry=0
        self.preferred={}; self.events=deque(maxlen=120); self.clock_offset=0
        self.mode='auto'
        self.set_token(token)
    def set_token(self,token):
        token=str(token).strip()
        token=re.sub(r'^Authorization\s*:\s*','',token,flags=re.I)
        token=re.sub(r'^Bearer\s+','',token,flags=re.I).strip()
        if len(token)>65536 or any(c.isspace() or ord(c)<32 for c in token):
            raise ValueError('Вставьте одно значение токена без переводов строк.')
        with self.lock:
            self.token=token;self.credentials=None;self.cred_expiry=0;self.preferred={}
            if token: SECRETS.append(token)
    def token_info(self):
        result={'present':bool(self.token),'expires':None,'remaining':None,'sts_expires':self.cred_expiry or None,'mode':self.mode}
        try:
            part=self.token.split('.')[1]
            expiry=float(json.loads(base64.urlsafe_b64decode(part+'='*(-len(part)%4)))['exp'])
            if not math.isfinite(expiry):
                raise ValueError('Неконечный срок токена.')
            result.update(expires=expiry,remaining=int(expiry-time.time()))
        except (ValueError,IndexError,KeyError,TypeError,OverflowError):
            pass
        return result
    def validate(self,url,api=False):
        p=urlsplit(url)
        if any(c in url for c in ('\r','\n','\x00')) or p.username or p.password or p.fragment:
            raise NetworkError('Недопустимый адрес.')
        if (p.scheme,p.netloc) in self.test_hosts:
            return p
        if p.scheme!='https' or p.port not in (None,443):
            raise NetworkError('Разрешён только HTTPS на стандартном порту.')
        allowed={'api.gptl.ru'} if api else TRUSTED_HOSTS
        if p.hostname not in allowed:
            raise NetworkError('Адрес вне доверенных сервисов GPTL. Bearer не отправлен.')
        return p
    def event(self,step,status,mode='',code=''):
        with self.lock:
            self.events.append({'time':iso(now()),'step':step,'http':status,'mode':mode,'code':code})
    def fail(self,r,step,mode=''):
        try:
            raw=r.read(32768)
        finally:
            r.close()
        code,_=error_code(raw)
        self.event(step,r.status,mode,code)
        # Do not persist server XML: it may echo credentials, signed canonical requests or private paths.
        raise NetworkError(error_hint(r.status,code,mode),r.status,code)
    def open_retry(self,url,headers=None,method='GET',body=None,cancel=None):
        for attempt in range(3):
            if cancel and cancel.is_set():raise Cancelled()
            try:
                r=self.http.open(url,headers,method,body)
            except NetworkError:
                if attempt==2:raise
                r=None
            if r and (r.status not in (429,502,503,504) or attempt==2):return r
            if r:r.close()
            if cancel:
                if cancel.wait(0.5*2**attempt):raise Cancelled()
            else:time.sleep(0.5*2**attempt)
        raise NetworkError('Сетевая ошибка.')
    def api_json(self,url,method='GET',body=None,cancel=None):
        for _ in range(4):
            self.validate(url,api=True)
            headers={'Accept':'application/json'}
            if self.token:headers['Authorization']='Bearer '+self.token
            data=json.dumps(body).encode() if body is not None else None
            if data:headers['Content-Type']='application/json'
            r=self.open_retry(url,headers,method,data,cancel)
            if r.status in (301,302,303,307,308) and r.headers.get('location'):
                url=urljoin(url,r.headers['location']);r.close();continue
            if r.status!=200:self.fail(r,'API','bearer')
            with r:
                content=r.read(32*1024*1024+1)
            if len(content)>32*1024*1024:raise NetworkError('Ответ каталога превышает 32 МиБ.')
            self.event('API',200,'bearer' if self.token else 'public')
            try:return json.loads(content)
            except ValueError:raise NetworkError('Вместо JSON каталог вернул другой формат.')
        raise NetworkError('Слишком много перенаправлений API.')
    def sts(self,cancel=None,force=False):
        with self.sts_lock:
            with self.lock:
                if self.credentials and self.cred_expiry>time.time()+90 and not force:return dict(self.credentials)
                token=self.token
            if not token:raise NetworkError('Для закрытых объектов вставьте Bearer в «Подключение».',401,'MissingBearer')
            payload=urlencode({'Action':'AssumeRoleWithWebIdentity','Version':'2011-06-15','DurationSeconds':3600,'WebIdentityToken':token})
            r=self.open_retry(S3,{'Content-Type':'application/x-www-form-urlencoded'},'POST',payload,cancel)
            if r.status!=200:self.fail(r,'STS','bearer')
            with r:content=r.read(262144)
            try:
                root=ET.fromstring(content)
                creds={k:xml_value(root,k) for k in ('AccessKeyId','SecretAccessKey','SessionToken')}
                expiry=dt.datetime.fromisoformat(xml_value(root,'Expiration').replace('Z','+00:00')).timestamp()
                if not all(creds.values()) or expiry<=time.time():raise ValueError()
            except (ET.ParseError,ValueError):raise NetworkError('STS не выдал полный действующий комплект реквизитов.')
            with self.lock:
                if token!=self.token:raise NetworkError('Bearer заменён во время подключения. Повторите.')
                self.credentials=creds;self.cred_expiry=expiry
                SECRETS.extend(creds.values())
            self.event('STS',200,'sts')
            return dict(creds)
    def open_object(self,uri,headers=None,cancel=None,region=REGION,s3_listing=False):
        if urlsplit(uri).scheme=='s3':uri=s3_url(*s3_parts(uri))
        headers=dict(headers or {})
        for _ in range(4):
            p=self.validate(uri)
            is_s3=p.hostname in S3_HOSTS
            presigned=bool(p.query) and not s3_listing
            key=(p.hostname,p.path.split('/')[1] if '/' in p.path else '')
            if presigned:modes=['presigned']
            elif is_s3:
                if self.mode=='public':modes=['public']
                elif self.mode=='signed':modes=['sts']
                else:
                    preferred=self.preferred.get(key,'public')
                    modes=[preferred]+[m for m in ('public','sts') if m!=preferred]
                    if not self.token:modes=[m for m in modes if m!='sts']
            elif p.hostname=='api.gptl.ru':modes=['bearer']
            else:modes=['public']
            attempts=[]; redirect=None; last_error=None
            for mode in modes:
                force=False
                for retry in range(2):
                    if cancel and cancel.is_set():raise Cancelled()
                    h=dict(headers)
                    try:
                        if mode=='sts':
                            creds=self.sts(cancel,force)
                            h=sign(uri,creds,region,headers,now()+dt.timedelta(seconds=self.clock_offset))
                        elif mode=='bearer' and self.token:h['Authorization']='Bearer '+self.token
                        r=self.open_retry(uri,h,cancel=cancel)
                    except NetworkError as e:
                        last_error=e;attempts.append({'mode':mode,'http':e.status,'code':e.code});break
                    if r.status in (200,206,412,416):
                        r.mode=mode;r.attempts=attempts
                        if r.status in (200,206):
                            with self.lock:self.preferred[key]=mode
                        self.event('S3_LIST' if s3_listing else 'OBJECT',r.status,mode)
                        return r
                    if r.status in (301,302,303,307,308) and r.headers.get('location'):
                        redirect=urljoin(uri,r.headers['location']);r.close();break
                    raw=r.read(32768);r.close()
                    code,_=error_code(raw)
                    attempts.append({'mode':mode,'http':r.status,'code':code})
                    self.event('S3_LIST' if s3_listing else 'OBJECT',r.status,mode,code)
                    last_error=NetworkError(error_hint(r.status,code,mode),r.status,code,attempts)
                    if mode=='sts' and retry==0:
                        if code in ('ExpiredToken','InvalidToken','InvalidAccessKeyId'):
                            force=True;continue
                        if code=='RequestTimeTooSkewed' and r.headers.get('date'):
                            try:
                                offset=parsedate_to_datetime(r.headers['date']).timestamp()-time.time()
                                if abs(offset)<24*3600:self.clock_offset=offset;continue
                            except (ValueError,TypeError):pass
                    # Fall back only for an authorization failure, never a different key/address.
                    if r.status not in (401,403):raise last_error
                    break
                if redirect:break
            if redirect:
                self.validate(redirect)
                uri=redirect;continue  # never reuse Authorization across redirects
            if last_error:
                last_error.attempts=attempts
                raise last_error
            raise NetworkError('Доступ к объекту не установлен.')
        raise NetworkError('Слишком много перенаправлений объекта.')
    def object(self,uri,headers=None,cancel=None,maximum=16*1024*1024,region=REGION):
        with self.open_object(uri,headers,cancel,region) as r:
            data=r.read(maximum+1)
            if len(data)>maximum:raise NetworkError('Сервер не поддержал диапазон или ответ превысил ограничение.',r.status)
            return {'status':r.status,'headers':r.headers,'body':data,'mode':r.mode}
    def probe(self,asset,cancel=None):
        # Read only the first 64 bytes even when the origin ignores Range.
        with self.open_object(asset['uri'],{'Range':'bytes=0-63'},cancel,asset.get('region') or REGION) as r:
            if r.status not in (200,206):self.fail(r,'PROBE',r.mode)
            data=r.read(64)
            if not data or data.lstrip().lower().startswith((b'<html',b'<!doctype',b'<?xml',b'<error')):
                raise NetworkError('Вместо данных получен пустой ответ/страница ошибки.',r.status)
            return {'http':r.status,'mode':r.mode,'content_type':r.headers.get('content-type',''),
                    'content_range':r.headers.get('content-range',''),'bytes':len(data),
                    'signature':data[:8].hex(),'note':'Подтверждено чтение фрагмента, не полное скачивание и не калибровка.'}
    def stac(self,platform,start,stop,consume,progress,cancel,max_pages=1500):
        params={'datetime':start+'/'+stop,'platforms':platform,'limit':100}
        url,method,body=qurl(STAC,params),'GET',None
        seen,pages=set(),set(); matched=None; reason='Лимит страниц';complete=False
        for page_num in range(max_pages):
            if cancel.is_set():raise Cancelled()
            try:obj=self.api_json(url,method,body,cancel)
            except NetworkError as e:
                if page_num==0 and method=='GET' and e.status in (400,405,422):
                    method,body,url='POST',dict(params,platforms=[platform]),STAC
                    obj=self.api_json(url,method,body,cancel)
                else:raise
            feats=obj.get('features')
            if not isinstance(feats,list):raise NetworkError('STAC: нет массива features.')
            count=obj.get('numberMatched',(obj.get('context') or {}).get('matched'))
            if isinstance(count,int):matched=count
            fp=hashlib.sha256(json.dumps(feats,sort_keys=True).encode()).hexdigest()
            if feats and fp in pages:reason='Повтор страницы';break
            pages.add(fp);records=[];assets=[];outside=0
            for feature in feats:
                rec,aa=normalize_item(feature)
                if not rec or rec['platform']!=platform:outside+=1;continue
                # Endpoint uses inclusive upper boundary; remove next-day midnight locally.
                if not rec['time'] or not start<=rec['time']<=stop:outside+=1;continue
                if rec['id'] not in seen:seen.add(rec['id']);records.append(rec);assets.extend(aa)
            consume(records,assets)
            progress('{}: страница {}, {} / {} записей'.format(platform,page_num+1,len(seen),matched if matched is not None else '?'))
            if outside:reason='Сервер не соблюдает фильтр';break
            if matched is not None and len(seen)==matched:complete=True;reason='Полная выдача';break
            link=next((x for x in obj.get('links',[]) if x.get('rel')=='next' and x.get('href')),None)
            if not link:
                complete=matched==len(seen) if matched is not None else len(feats)<100
                reason='Полная выдача' if complete else 'Счётчик и страницы не согласованы';break
            p=urlsplit(urljoin(url,link['href']))
            if p.hostname!='api.gptl.ru' or p.scheme not in ('http','https') or p.port not in (None,80,443):
                reason='Недопустимый next';break
            path=p.path
            if path.startswith('/catalogs/'):path='/stac/api/v1'+path
            url=urlunsplit(('https','api.gptl.ru',path,p.query,''))
            method=str(link.get('method') or 'GET').upper()
            if method not in ('GET','POST'):reason='Неподдерживаемый метод next';break
            nxt=link.get('body')
            if method=='POST' and link.get('merge'):
                merged=dict(body or params);merged.update(nxt or {});nxt=merged
            body=nxt if method=='POST' else None
        return {'complete':complete,'count':len(seen),'matched':matched,'reason':reason}
    def legacy(self,platform,start,stop,consume,progress,cancel,max_pages=1000):
        seen=set();offset=0
        params={'platform_identifiers':platform,'acquisition_date_after':start,'acquisition_date_before':stop,'limit':100}
        for page in range(max_pages):
            if cancel.is_set():raise Cancelled()
            obj=self.api_json(qurl(LEGACY,dict(params,offset=offset)),cancel=cancel)
            rows=obj.get('results')
            if not isinstance(rows,list):raise NetworkError('ЦБГД: нет массива results.')
            records=[];assets=[]
            for item in rows:
                rec,aa=normalize_item(item,'catalog')
                if rec and rec['platform']==platform and rec['id'] not in seen:
                    seen.add(rec['id']);records.append(rec);assets.extend(aa)
            consume(records,assets);offset+=len(rows)
            progress('ЦБГД {}: {} записей'.format(platform,len(seen)))
            if not rows or offset>=obj.get('count',offset+1):return len(seen)
            if not records:raise NetworkError('ЦБГД повторил страницу или не соблюдает фильтр. Выдача неполная.')
        raise NetworkError('Лимит страниц ЦБГД: выдача неполная.')
    def list_s3(self,bucket,prefix='',cancel=None,cursor=''):
        params={'list-type':2,'prefix':prefix,'delimiter':'/','max-keys':500}
        if cursor:params['continuation-token']=cursor
        url=qurl(s3_url(bucket),params)
        with self.open_object(url,cancel=cancel,s3_listing=True) as r:
            if r.status!=200:self.fail(r,'S3_LIST',r.mode)
            content=r.read(4*1024*1024+1)
        if len(content)>4*1024*1024:raise NetworkError('Список S3 слишком велик.')
        try:root=ET.fromstring(content)
        except ET.ParseError:raise NetworkError('S3 не вернул XML списка.')
        folders=[];objects=[]
        for el in root:
            name=el.tag.split('}')[-1]
            if name=='CommonPrefixes':folders.append(xml_value(el,'Prefix'))
            elif name=='Contents':
                key=xml_value(el,'Key')
                if key and key!=prefix:objects.append({'key':key,'size':int(xml_value(el,'Size') or 0)})
        truncated=xml_value(root,'IsTruncated').lower()=='true'
        return {'bucket':bucket,'prefix':prefix,'folders':folders,'objects':objects,
                'next':xml_value(root,'NextContinuationToken') if truncated else '', 'complete':not truncated}
