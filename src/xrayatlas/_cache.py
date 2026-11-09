"""On-disk cache of fetched dataset payloads.

Entries live under ``$XDG_CACHE_HOME/xrayatlas`` (default ``~/.cache/xrayatlas``). Each entry is a
data file plus a metadata file holding the key and the SHA-256 of the data. A read verifies both; an
entry that fails verification is deleted and reads as a miss. Writes are atomic (temporary file then
rename), data first and metadata second, so an interrupted write is never mistaken for a valid
entry. Concurrent processes are not locked, so the worst case under contention is a temporary miss,
never a corrupt hit. The cache never stores tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def default_cache_root() -> Path:
    """Directory of the cache: ``$XDG_CACHE_HOME/xrayatlas`` or ``~/.cache/xrayatlas``."""
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "xrayatlas"


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


class Cache:
    """Verified key-value store of byte payloads."""

    def __init__(self, root: Path | None = None) -> None:
        """Create a cache.

        Args:
            root: Cache directory, or ``None`` for `default_cache_root`.
        """
        self.root = root if root is not None else default_cache_root()

    def _paths(self, key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        folder = self.root / digest[:2]
        return folder / f"{digest}.bin", folder / f"{digest}.json"

    def get(self, key: str) -> bytes | None:
        """Read an entry.

        Returns:
            The stored bytes, or ``None`` when the entry is missing, unreadable, belongs to another
            key, or fails its checksum (an unreadable or failing entry is deleted).
        """
        data_path, meta_path = self._paths(key)
        try:
            data = data_path.read_bytes()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            for path in (data_path, meta_path):
                path.unlink(missing_ok=True)
            return None
        valid = (
            isinstance(meta, dict)
            and meta.get("key") == key
            and meta.get("sha256") == hashlib.sha256(data).hexdigest()
        )
        if not valid:
            for path in (data_path, meta_path):
                path.unlink(missing_ok=True)
            return None
        return data

    def put(self, key: str, data: bytes) -> None:
        """Store an entry atomically."""
        data_path, meta_path = self._paths(key)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(
            {"key": key, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        )
        _atomic_write(data_path, data)
        _atomic_write(meta_path, meta.encode("utf-8"))

    def delete(self, key: str) -> None:
        """Delete an entry, if present."""
        for path in self._paths(key):
            path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Delete every entry."""
        shutil.rmtree(self.root, ignore_errors=True)
