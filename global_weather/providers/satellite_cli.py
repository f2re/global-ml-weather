"""Bounded satellite acquisition CLI; catalog access never grants model admission."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

from ..autonomous.service import Credentials
from ..pipeline.io import digest, read_json, sha256
from ..pipeline.dataset import utc
from ._gptl.network import SECRETS, redact
from .satellites import SatelliteCatalog

MAX_BYTES = 512 * 1024**2
PUBLIC_FIELDS = (
    "id",
    "platform",
    "time",
    "channel",
    "level",
    "category",
    "size",
    "filename",
)


def safe(value: Any) -> Any:
    """Only explicitly selected metadata is passed here, never raw provider assets."""
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, str):
        return redact(value)
    return value


def save_new(path: str | Path, value: Any) -> None:
    path = Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise FileExistsError("Нужен новый файл отчёта без символических ссылок.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".satellite-report-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def checked_request(value: Any) -> dict[str, Any]:
    required = {"start", "end", "platform", "collection", "max_pages"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Неверный состав паспорта поиска.")
    a, b = utc(value["start"]), utc(value["end"])
    if not a < b or b - a > timedelta(days=7):
        raise ValueError("Поиск ограничен семью сутками.")
    for key in ("platform", "collection"):
        field = value[key]
        if (
            not isinstance(field, str)
            or field
            and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", field)
        ):
            raise ValueError("Неверный идентификатор каталога.")
    if not value["platform"] and not value["collection"]:
        raise ValueError("Нужна платформа либо коллекция из каталога.")
    if type(value["max_pages"]) is not int or not 1 <= value["max_pages"] <= 10:
        raise ValueError("Предел страниц должен быть от 1 до 10.")
    return dict(value, start=a.isoformat(), end=b.isoformat())


def search_report(
    catalog: SatelliteCatalog, request: dict[str, Any], *, network: bool
) -> dict[str, Any]:
    result = catalog.search(**request, network=network)
    rows = [{key: item.get(key) for key in PUBLIC_FIELDS} for item in result["items"]]
    return {
        "schema": "satellite-search-cli-1",
        "request": request,
        "request_sha256": digest(request),
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "items": safe(rows),
        "truncated": result["truncated"],
        "period_completeness_verified": False,
        "model_ready": False,
        "note": "Выдача каталога не удостоверяет полноту периода и физический допуск.",
    }


def inspect_cache(cache: str | Path, identity: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise ValueError("Неверный идентификатор ресурса.")
    root = Path(cache).absolute() / identity
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Ссылка в каталоге кэша.")
    receipt = read_json(root / "receipt.json")
    path = root / "source.tif"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Нет обычного исходного файла.")
    if (
        receipt.get("id") != identity
        or receipt.get("bytes") != path.stat().st_size
        or receipt.get("sha256") != sha256(path)
    ):
        raise ValueError("Размер, идентичность или SHA256 кэша не совпали.")
    from ..connectors.ecosystem import gptl_asset

    asset = read_json(root / "asset.json")
    platform = asset.get("platform")
    source = "arktika_m" if platform in ("ARCM1", "ARCM2") else "electro_l"
    # The legacy bridge lacks a reviewed Electro platform registry. Inspection
    # must not label an arbitrary non-Arktika platform as an admitted sensor.
    report = {
        "schema": "satellite-cache-inspection-1",
        "id": identity,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "platform": safe(platform),
        "model_ready": False,
        "status": "transport_verified",
        "pending": [
            "view_geometry",
            "sensor_normalization",
            "physical_platform_review",
        ],
    }
    if source == "arktika_m":
        try:
            spec = gptl_asset(root / "asset.json", path, source=source)
        except (ValueError, KeyError, TypeError):
            report["status"] = "quarantined_physical_metadata"
        else:
            report["status"] = "physical_metadata_verified"
            report["quantity"] = spec["quantity"]
            report["units"] = spec["units"]
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--credentials",
        type=Path,
        help="Existing Credentials file; no token arguments.",
    )
    parser.add_argument("--arktika-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collections", "search", "download", "inspect-local", "import-local"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, required=True)
        if name in ("collections", "search", "download"):
            sub.add_argument("--network", action="store_true")
        if name == "search":
            sub.add_argument("--start", required=True)
            sub.add_argument("--end", required=True)
            sub.add_argument("--platform", default="")
            sub.add_argument("--collection", default="")
            sub.add_argument("--max-pages", type=int, default=3)
        if name == "download":
            sub.add_argument("--search-report", type=Path, required=True)
            sub.add_argument("--id", required=True)
            sub.add_argument("--max-bytes", type=int, default=MAX_BYTES)
        if name in ("inspect-local", "import-local"):
            sub.add_argument("--id", required=name == "import-local")
    args = parser.parse_args(argv)
    if args.output.exists() or any(
        p.is_symlink()
        for p in (args.output.absolute(), *args.output.absolute().parents)
    ):
        parser.error("Нужен новый файл отчёта без ссылок.")
    if args.command in ("collections", "search", "download") and not args.network:
        parser.error("Сеть требует явного --network.")
    credentials = Credentials(args.credentials) if args.credentials else None
    if credentials:
        SECRETS.extend(value for value in credentials.read().values() if value)
    catalog = SatelliteCatalog(
        args.cache,
        args.arktika_root,
        credentials.read if credentials else None,
        credentials.save if credentials else None,
    )
    if args.command == "collections":
        result = {
            "schema": "satellite-collections-cli-1",
            "collections": safe(catalog.collections(network=True)),
            "model_ready": False,
            "period_completeness_verified": False,
        }
    elif args.command == "search":
        request = checked_request(
            {
                key: getattr(args, key)
                for key in ("start", "end", "platform", "collection", "max_pages")
            }
        )
        result = search_report(catalog, request, network=True)
    elif args.command == "download":
        if (
            not re.fullmatch(r"[a-f0-9]{32}", args.id)
            or not 1 <= args.max_bytes <= MAX_BYTES
        ):
            parser.error("Неверный ID либо предел байт (1–536870912).")
        passport = read_json(args.search_report)
        request = checked_request(passport.get("request"))
        if passport.get("schema") != "satellite-search-cli-1" or passport.get(
            "request_sha256"
        ) != digest(request):
            raise ValueError("Паспорт поиска изменён.")
        old = [item for item in passport["items"] if item.get("id") == args.id]
        if len(old) != 1:
            raise ValueError("Выберите ровно один ресурс из сохранённой выдачи.")
        refreshed = search_report(catalog, request, network=True)
        current = [item for item in refreshed["items"] if item.get("id") == args.id]
        if current != old:
            raise ValueError("Ресурс исчез или его публичные метаданные изменились.")
        directory = args.cache / args.id
        if directory.exists() or directory.is_symlink():
            cached = inspect_cache(args.cache, args.id)
            if cached["bytes"] > args.max_bytes:
                raise ValueError("Готовый кэш превышает предел байт.")
        receipt = catalog.download(args.id, network=True, max_bytes=args.max_bytes)
        cached = inspect_cache(args.cache, args.id)
        if (
            receipt.get("id") != args.id
            or receipt.get("sha256") != cached["sha256"]
            or receipt.get("bytes") != cached["bytes"]
        ):
            raise ValueError("Результат загрузки не совпал с фактическим кэшем.")
        if cached["bytes"] > args.max_bytes:
            raise ValueError("Готовый кэш превышает предел байт.")
        result = {
            "schema": "satellite-download-cli-1",
            "request_sha256": digest(request),
            "asset": current[0],
            "receipt": safe(
                {
                    key: receipt.get(key)
                    for key in (
                        "id",
                        "bytes",
                        "sha256",
                        "acquired_at",
                        "status",
                        "level",
                        "channel",
                    )
                }
            ),
            "search_truncated": refreshed["truncated"],
            "period_completeness_verified": False,
            "model_ready": False,
        }
    elif args.command == "inspect-local":
        if args.id:
            result = inspect_cache(args.cache, args.id)
        else:
            result = safe(catalog.browse_local(limit=200))
    else:
        if not args.arktika_root:
            parser.error("Для импорта нужен --arktika-root.")
        catalog.browse_local(limit=200)
        result = catalog.import_local(args.id)
    save_new(args.output, result)
    return result


if __name__ == "__main__":
    main()
