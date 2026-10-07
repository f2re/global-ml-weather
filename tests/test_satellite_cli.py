"""Injected providers check the CLI contract; these are not real satellite tests."""

import json

import pytest

from global_weather.pipeline.io import digest
from global_weather.providers import satellite_cli as cli

ID = "a" * 32
ASSET = {
    "id": ID,
    "platform": "UNREVIEWED_PLATFORM",
    "time": "2020-01-01T00:00:00Z",
    "channel": 9,
    "level": "L2IR",
    "category": "channel",
    "size": 8,
    "filename": "channel.tif",
}


class Catalog:
    instances = []
    changed = False

    def __init__(self, cache, root=None, credentials=None, save_credentials=None):
        self.calls = []
        self.cache = cache
        self.credentials = credentials
        self.instances.append(self)

    def collections(self, *, network=False):
        assert network
        return [
            {
                "id": "fixture",
                "title": "https://example.invalid/?signature=SECRET",
                "platforms": [],
            }
        ]

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        row = dict(ASSET)
        if self.changed:
            row["size"] = 9
        row["uri"] = "https://example.invalid/?signature=SECRET"
        row["token"] = "SECRET"
        return {"items": [row], "truncated": True}

    def download(self, identity, **kwargs):
        self.calls.append(("download", identity, kwargs))
        root = self.cache / identity
        root.mkdir(parents=True)
        source = root / "source.tif"
        source.write_bytes(b"FAKE1234")
        receipt = {
            "id": identity,
            "bytes": 8,
            "sha256": cli.sha256(source),
            "status": "downloaded_not_physically_admitted",
            "model_ready": False,
            "uri": "https://example.invalid/?signature=SECRET",
        }
        (root / "receipt.json").write_text(json.dumps(receipt))
        (root / "asset.json").write_text(
            json.dumps({"platform": "UNREVIEWED_PLATFORM"})
        )
        return receipt


@pytest.fixture
def catalog(monkeypatch):
    Catalog.instances = []
    Catalog.changed = False
    monkeypatch.setattr(cli, "SatelliteCatalog", Catalog)
    return Catalog


def search(tmp_path):
    output = tmp_path / "search.json"
    cli.main(
        [
            "--cache",
            str(tmp_path / "cache"),
            "search",
            "--network",
            "--start",
            "2020-01-01T00:00:00Z",
            "--end",
            "2020-01-02T00:00:00Z",
            "--collection",
            "fixture",
            "--max-pages",
            "2",
            "--output",
            str(output),
        ]
    )
    return output


def test_search_records_bounds_and_never_claims_completeness(tmp_path, catalog):
    path = search(tmp_path)
    report = json.loads(path.read_text())
    assert report["truncated"] is True
    assert report["period_completeness_verified"] is False
    assert report["model_ready"] is False
    assert report["request"]["max_pages"] == 2
    assert report["request_sha256"] == digest(report["request"])
    assert report["items"][0]["platform"] == "UNREVIEWED_PLATFORM"
    assert "SECRET" not in path.read_text() and "uri" not in report["items"][0]


def test_explicit_network_required_before_client_creation(tmp_path, catalog):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--cache",
                str(tmp_path),
                "collections",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
    assert not catalog.instances


def test_download_repeats_frozen_request_and_passes_byte_budget(tmp_path, catalog):
    passport = search(tmp_path)
    result = cli.main(
        [
            "--cache",
            str(tmp_path / "cache"),
            "download",
            "--network",
            "--search-report",
            str(passport),
            "--id",
            ID,
            "--max-bytes",
            "8",
            "--output",
            str(tmp_path / "download.json"),
        ]
    )
    calls = catalog.instances[-1].calls
    assert calls[0][0] == "search" and calls[0][1]["max_pages"] == 2
    assert calls[1] == ("download", ID, {"network": True, "max_bytes": 8})
    assert result["search_truncated"] is True and result["model_ready"] is False
    assert "SECRET" not in (tmp_path / "download.json").read_text()


def test_changed_asset_blocks_download(tmp_path, catalog):
    passport = search(tmp_path)
    catalog.changed = True
    with pytest.raises(ValueError, match="метаданные"):
        cli.main(
            [
                "--cache",
                str(tmp_path / "cache"),
                "download",
                "--network",
                "--search-report",
                str(passport),
                "--id",
                ID,
                "--output",
                str(tmp_path / "download.json"),
            ]
        )
    assert [call[0] for call in catalog.instances[-1].calls] == ["search"]


def test_modified_request_blocks_network_search(tmp_path, catalog):
    passport = search(tmp_path)
    value = json.loads(passport.read_text())
    value["request"]["max_pages"] = 9
    passport.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="изменён"):
        cli.main(
            [
                "--cache",
                str(tmp_path / "cache"),
                "download",
                "--network",
                "--search-report",
                str(passport),
                "--id",
                ID,
                "--output",
                str(tmp_path / "download.json"),
            ]
        )
    assert catalog.instances[-1].calls == []


def test_credentials_only_from_existing_secure_file(tmp_path, catalog):
    credentials = tmp_path / "credentials.json"
    credentials.write_text(json.dumps({"gptl_token": "test-token"}))
    credentials.chmod(0o600)
    cli.main(
        [
            "--cache",
            str(tmp_path / "cache"),
            "--credentials",
            str(credentials),
            "collections",
            "--network",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert catalog.instances[-1].credentials() == {"gptl_token": "test-token"}
    assert "test-token" not in (tmp_path / "out.json").read_text()
    credentials.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        cli.main(
            [
                "--cache",
                str(tmp_path / "cache"),
                "--credentials",
                str(credentials),
                "collections",
                "--network",
                "--output",
                str(tmp_path / "other.json"),
            ]
        )


def test_existing_output_blocks_network_before_client(tmp_path, catalog):
    output = tmp_path / "out.json"
    output.write_text("preserved")
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--cache",
                str(tmp_path),
                "collections",
                "--network",
                "--output",
                str(output),
            ]
        )
    assert output.read_text() == "preserved" and catalog.instances == []


def test_inspection_keeps_unknown_platform_unadmitted(tmp_path):
    root = tmp_path / ID
    root.mkdir()
    source = root / "source.tif"
    source.write_bytes(b"TEST_NOT_A_REAL_RASTER")
    checksum = cli.sha256(source)
    (root / "receipt.json").write_text(
        json.dumps({"id": ID, "bytes": source.stat().st_size, "sha256": checksum})
    )
    (root / "asset.json").write_text(json.dumps({"platform": "UNREVIEWED_PLATFORM"}))
    report = cli.inspect_cache(tmp_path, ID)
    assert report["status"] == "transport_verified" and report["model_ready"] is False
    assert "physical_platform_review" in report["pending"]
    source.write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="SHA256"):
        cli.inspect_cache(tmp_path, ID)


def test_cache_receipt_cannot_understate_actual_download_size(tmp_path, catalog):
    passport = search(tmp_path)
    root = tmp_path / "cache" / ID
    root.mkdir(parents=True)
    source = root / "source.tif"
    source.write_bytes(b"123456789")
    (root / "receipt.json").write_text(
        json.dumps({"id": ID, "bytes": 1, "sha256": cli.sha256(source)})
    )
    (root / "asset.json").write_text(json.dumps({"platform": "UNREVIEWED_PLATFORM"}))
    with pytest.raises(ValueError, match="SHA256"):
        cli.main(
            [
                "--cache",
                str(tmp_path / "cache"),
                "download",
                "--network",
                "--search-report",
                str(passport),
                "--id",
                ID,
                "--max-bytes",
                "8",
                "--output",
                str(tmp_path / "download.json"),
            ]
        )
    assert [call[0] for call in catalog.instances[-1].calls] == ["search"]
