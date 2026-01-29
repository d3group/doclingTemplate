"""Tests for the ingestion pipeline."""

import tempfile
from pathlib import Path

from docling_rag.ingest import get_file_hash, load_hashes, save_hashes


def test_get_file_hash():
    """Test that file hashing works correctly."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("test content")
        temp_path = Path(f.name)

    try:
        hash1 = get_file_hash(temp_path)
        hash2 = get_file_hash(temp_path)
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length
    finally:
        temp_path.unlink()


def test_hash_persistence(tmp_path, monkeypatch):
    """Test that hashes can be saved and loaded."""
    hash_file = tmp_path / ".docling_hashes.json"
    monkeypatch.setattr("docling_rag.ingest.HASH_FILE", hash_file)

    test_hashes = {"file1.pdf": "abc123", "file2.docx": "def456"}
    save_hashes(test_hashes)

    loaded = load_hashes()
    assert loaded == test_hashes


def test_load_hashes_empty(tmp_path, monkeypatch):
    """Test loading hashes when file doesn't exist."""
    hash_file = tmp_path / ".docling_hashes.json"
    monkeypatch.setattr("docling_rag.ingest.HASH_FILE", hash_file)

    loaded = load_hashes()
    assert loaded == {}
