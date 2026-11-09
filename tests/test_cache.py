from pathlib import Path

from xrayatlas._cache import Cache


def test_roundtrip(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    assert cache.get("k") is None
    cache.put("k", b"hello")
    assert cache.get("k") == b"hello"


def test_tampered_data_is_discarded(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("k", b"hello")
    data_path, _meta = cache._paths("k")
    data_path.write_bytes(b"HELLO")
    assert cache.get("k") is None
    assert not data_path.exists()


def test_entry_for_another_key_is_not_returned(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("a", b"1")
    data_a, meta_a = cache._paths("a")
    data_b, meta_b = cache._paths("b")
    data_b.parent.mkdir(parents=True, exist_ok=True)
    data_b.write_bytes(data_a.read_bytes())
    meta_b.write_bytes(meta_a.read_bytes())
    assert cache.get("b") is None


def test_missing_metadata_reads_as_miss(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("k", b"x")
    _data, meta = cache._paths("k")
    meta.unlink()
    assert cache.get("k") is None


def test_put_leaves_no_temporary_files(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("k", b"x")
    names = sorted(p.suffix for p in tmp_path.rglob("*") if p.is_file())
    assert names == [".bin", ".json"]
