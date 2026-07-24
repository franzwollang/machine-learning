"""On-demand dataset download, caching, and checksum verification.

Manages a cache directory under tests/.data_cache/ (gitignored).
Downloads datasets on first use; verifies SHA-256 checksums; supports
offline mode that skips datasets not yet cached.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


_MANIFEST_PATH = Path(__file__).parent / "data_cache_manifest.toml"
_DEFAULT_CACHE = Path(__file__).parents[1] / ".data_cache"


def _get_cache_dir(request: Optional[Any] = None) -> Path:
    """Resolve the cache directory from CLI option or default."""
    if request is not None:
        custom = request.config.getoption("--real-data-cache", default=None)
        if custom:
            return Path(custom)
    env = os.environ.get("PROTEUS_DATA_CACHE")
    if env:
        return Path(env)
    return _DEFAULT_CACHE


def _load_manifest() -> dict[str, Any]:
    with open(_MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)


def _verify_checksum(path: Path, expected_sha256: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256


def download_dataset(name: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a dataset if not already cached; verify checksum."""
    manifest = _load_manifest()
    if name not in manifest:
        raise KeyError(f"Dataset {name!r} not in manifest.")
    entry = manifest[name]
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / entry["filename"]
    if dest.exists():
        if _verify_checksum(dest, entry["sha256"]):
            return dest
    offline = os.environ.get("PROTEUS_OFFLINE", "").lower() in ("1", "true", "yes")
    if offline:
        pytest.skip(f"Dataset {name!r} not cached and PROTEUS_OFFLINE is set.")
    try:
        import requests
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(
            "requests and tqdm are required for dataset downloads. "
            "Install with: pip install 'proteus[real_data]'"
        ) from e
    resp = requests.get(entry["url"], stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    if not _verify_checksum(dest, entry["sha256"]):
        dest.unlink()
        raise RuntimeError(f"Checksum mismatch for {name!r}.")
    return dest


@pytest.fixture
def cached_dataset(request):
    """Pytest fixture: returns a loader function ``cached_dataset(name)``."""
    cache_dir = _get_cache_dir(request)

    def _load(name: str) -> Path:
        return download_dataset(name, cache_dir)

    return _load


def pytest_addoption(parser):
    parser.addoption(
        "--real-data-cache",
        action="store",
        default=None,
        help="Path to a pre-populated real-data cache directory.",
    )
