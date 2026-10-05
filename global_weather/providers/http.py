"""Receipt-checked immutable downloads from fixed public providers."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from ..pipeline.io import atomic_json, sha256

HOSTS = {"www.ncei.noaa.gov", "noaa-ghcnh-pds.s3.amazonaws.com", "storage.googleapis.com"}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def download(url, output, *, max_bytes=100_000_000, network=False, cancelled=None):
    if type(max_bytes) is not int or max_bytes<=0:raise ValueError("Неверный предел загрузки.")
    p = urlsplit(url)
    if p.scheme != "https" or p.hostname not in HOSTS or p.port not in (None,443) or p.username or p.password or p.query or p.fragment:
        raise ValueError("Адрес вне разрешённых источников.")
    output = Path(output)
    if any(q.is_symlink() for q in (output, *output.parents)):
        raise ValueError("Ссылки в кэше запрещены.")
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists():
        if not receipt.is_file() or receipt.is_symlink():
            raise ValueError("Файл кэша не имеет паспорта. Он не будет использован или перезаписан.")
        row = json.loads(receipt.read_text())
        if row.get("url") != url or row.get("sha256") != sha256(output) or not 0 < output.stat().st_size <= max_bytes or row.get("bytes") != output.stat().st_size:
            raise ValueError("Кэш повреждён или относится к другому запросу.")
        return row
    if not network:
        raise ValueError("Для недостающих данных требуется разрешение сети при запуске.")
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        if cancelled and cancelled():
            raise InterruptedError("Загрузка отменена.")
        fd, name = tempfile.mkstemp(prefix=".download-", dir=output.parent)
        temporary = Path(name)
        try:
            count = 0
            with os.fdopen(fd, "wb") as stream, build_opener(NoRedirect()).open(
                    Request(url, headers={"User-Agent": "global-ml-weather/0.7", "Accept-Encoding":"identity"}), timeout=45) as response:
                while block := response.read(256*1024):
                    if cancelled and cancelled():
                        raise InterruptedError("Загрузка отменена.")
                    count += len(block)
                    if count > max_bytes:
                        raise ValueError("Источник превышает бюджет загрузки.")
                    stream.write(block)
                stream.flush(); os.fsync(stream.fileno())
            if not count:
                raise ValueError("Пустой ответ источника.")
            row = {"url":url, "sha256":sha256(temporary), "bytes":count,
                   "acquired_at":datetime.now(timezone.utc).isoformat(), "content_verified":False}
            os.link(temporary, output)
            atomic_json(receipt, row)
            return row
        except (URLError, OSError) as exc:
            if isinstance(exc, FileExistsError) or (isinstance(exc, HTTPError) and exc.code not in (429,500,502,503,504)) or attempt == 2:
                raise ValueError("Источник недоступен: проверьте сеть и повторите запуск.") from exc
            time.sleep(0.5 * 2**attempt)
        finally:
            temporary.unlink(missing_ok=True)
