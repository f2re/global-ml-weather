"""OAuth code+PKCE и обновление токена. Нужен зарегистрированный клиент GPTL."""
import base64,hashlib,json,os,secrets,threading,time
from urllib.parse import urlencode,urlsplit
from .network import Client,NetworkError,SECRETS
ISSUER='https://auth.gptl.ru/auth/realms/etris/protocol/openid-connect'
class AuthClient(Client):
 def __init__(self,token='',oauth=None,**kw):
  self.oauth=oauth or {};self.refresh_lock=threading.RLock();self.refresh_token='';self.pending={};self.last_refresh=0.
  super().__init__(token,**kw)
 def set_token(self,token):
  # Manual Bearer replacement must not retain another account's refresh token.
  with self.refresh_lock:
   super().set_token(token);self.refresh_token='';self.pending={};self.last_refresh=0.
 def replace(self,token,refresh=''):
  with self.refresh_lock:
   self.set_token(token);self.refresh_token=str(refresh or '');self.last_refresh=0
   if self.refresh_token:SECRETS.append(self.refresh_token)
 def configure_oauth(self,cfg):
  if set(cfg)-{'client_id','redirect_uri','scope'}:raise ValueError('Допустимы client_id, redirect_uri, scope.')
  new=dict(self.oauth,**cfg);self.validate_oauth(new)
  with self.refresh_lock:
   if new!=self.oauth:self.replace('')
   self.oauth=new
 def validate_oauth(self,cfg):
  if not cfg.get('client_id'):raise ValueError('client_id должен быть выдан оператором GPTL.')
  p=urlsplit(cfg.get('redirect_uri',''))
  if p.scheme!='http' or p.hostname!='127.0.0.1' or p.path!='/oauth/callback' or p.query or p.fragment or p.username:raise ValueError('Нужен зарегистрированный redirect_uri http://127.0.0.1:ПОРТ/oauth/callback.')
 def begin(self):
  self.validate_oauth(self.oauth);state=secrets.token_urlsafe(32);verifier=secrets.token_urlsafe(48)
  challenge=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
  with self.refresh_lock:self.pending={state:(verifier,time.time()+300)}
  return ISSUER+'/auth?'+urlencode(dict(client_id=self.oauth['client_id'],redirect_uri=self.oauth['redirect_uri'],scope=self.oauth.get('scope','openid s3policy offline_access'),response_type='code',state=state,code_challenge=challenge,code_challenge_method='S256'))
 def exchange(self,fields):
  self.validate_oauth(self.oauth);fields=dict(fields,client_id=self.oauth['client_id'])
  secret=os.environ.get('GPTL_CLIENT_SECRET','')
  if secret:fields['client_secret']=secret;SECRETS.append(secret)
  r=self.open_retry(ISSUER+'/token',{'Content-Type':'application/x-www-form-urlencoded'},'POST',urlencode(fields))
  if r.status!=200:r.close();raise NetworkError('OAuth не выдал токены. Проверьте клиента, redirect_uri и refresh token.',r.status,'OAuthError')
  with r:obj=json.loads(r.read(131072))
  if not isinstance(obj.get('access_token'),str):raise NetworkError('В ответе OAuth нет access_token.')
  Client.set_token(self,obj['access_token']);self.refresh_token=obj.get('refresh_token') or self.refresh_token
  if self.refresh_token:SECRETS.append(self.refresh_token)
  return self.token_info()
 def callback(self,code,state):
  with self.refresh_lock:
   p=self.pending.pop(str(state),None)
   if not p or p[1]<time.time():raise ValueError('Неверный или истёкший OAuth state. Начните вход заново.')
   return self.exchange(dict(grant_type='authorization_code',code=str(code),redirect_uri=self.oauth['redirect_uri'],code_verifier=p[0]))
 def refresh(self,force=False):
  with self.refresh_lock:
   remaining=self.token_info().get('remaining')
   if not force and (remaining is None or remaining>120):return
   if not self.refresh_token:
    if force:raise ValueError('Для обновления нужен refresh token того же зарегистрированного OAuth-клиента. Один Bearer обновить невозможно.')
    return
   if not force and time.time()-self.last_refresh<30:return
   self.last_refresh=time.time();return self.exchange(dict(grant_type='refresh_token',refresh_token=self.refresh_token))
 def api_json(self,*args,**kw):self.refresh();return super().api_json(*args,**kw)
 def sts(self,*args,**kw):self.refresh();return super().sts(*args,**kw)
