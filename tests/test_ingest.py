"""Tests for the ingestion pipeline."""

from docling_rag.core.config import get_config
from docling_rag.core.ingest import (
    _get_cache_path,
    _read_markdown_cache,
    get_file_hash,
    ingest_documents,
    ingest_file,
    load_hashes,
    save_hashes,
)


def test_get_file_hash(sample_text_file):
    """Test that file hashing is deterministic."""
    hash1 = get_file_hash(sample_text_file)
    hash2 = get_file_hash(sample_text_file)
    assert hash1 == hash2
    assert len(hash1) == 32  # MD5 hex length


def test_get_file_hash_changes(sample_text_file):
    """Test that hash changes when file content changes."""
    hash1 = get_file_hash(sample_text_file)
    sample_text_file.write_text("different content")
    hash2 = get_file_hash(sample_text_file)
    assert hash1 != hash2


def test_hash_persistence(tmp_path):
    """Test that hashes can be saved and loaded."""
    test_hashes = {"file1.pdf": "abc123", "file2.docx": "def456"}
    save_hashes(test_hashes)
    loaded = load_hashes()
    assert loaded == test_hashes


def test_load_hashes_empty():
    """Test loading hashes when file doesn't exist."""
    loaded = load_hashes()
    assert loaded == {}


def test_ingest_text_file(sample_text_file):
    """Test ingestion of a text file."""
    stats = ingest_documents(verbose=False)
    assert stats["processed"] == 1
    assert stats["chunks_added"] >= 1
    assert stats["deleted"] == 0


def test_ingest_markdown_file(sample_markdown_file):
    """Test ingestion of a markdown file."""
    stats = ingest_documents(verbose=False)
    assert stats["processed"] == 1
    assert stats["chunks_added"] >= 1


def test_ingest_code_file(sample_code_file):
    """Test ingestion of a code file."""
    stats = ingest_documents(verbose=False)
    assert stats["processed"] == 1
    assert stats["chunks_added"] >= 1


def test_no_reprocess_on_second_run(sample_text_file):
    """Test that unchanged files are not reprocessed."""
    stats1 = ingest_documents(verbose=False)
    assert stats1["processed"] == 1

    stats2 = ingest_documents(verbose=False)
    assert stats2["processed"] == 0


def test_reprocess_on_change(sample_text_file):
    """Test that changed files are reprocessed."""
    ingest_documents(verbose=False)
    sample_text_file.write_text("completely new content about quantum computing")
    stats = ingest_documents(verbose=False)
    assert stats["processed"] == 1


def test_markdown_cache_created(sample_text_file):
    """Test that markdown cache files are created during ingestion."""
    ingest_documents(verbose=False)
    config = get_config()
    cache_dir = config.cache_dir
    assert cache_dir.exists()
    # Should have at least one cached markdown file
    md_files = list(cache_dir.rglob("*.md"))
    assert len(md_files) >= 1


def test_markdown_cache_reuse(sample_text_file):
    """Test that cached markdown is used on second ingestion."""
    # First ingest creates the cache
    ingest_documents(verbose=False)

    source_key = "sample.txt"
    cached = _read_markdown_cache(source_key)
    assert cached is not None
    assert len(cached) > 0


def test_deleted_file_cleanup(tmp_path):
    """Test that deleted files are cleaned up."""
    config = get_config()
    data_dir = config.data_dir
    file_path = data_dir / "temporary.txt"
    file_path.write_text("temporary content")

    # Ingest
    stats1 = ingest_documents(verbose=False)
    assert stats1["processed"] == 1

    # Delete the file
    file_path.unlink()

    # Re-ingest should detect deletion
    stats2 = ingest_documents(verbose=False)
    assert stats2["deleted"] == 1


def test_ingest_single_external_file(external_text_file):
    """Test ingesting a single file from outside the data directory."""
    stats = ingest_file(external_text_file, verbose=False)
    assert stats["processed"] == 1
    assert stats["chunks_added"] >= 1


def test_ingest_nonexistent_file():
    """Test that ingesting a nonexistent file returns zero stats."""
    stats = ingest_file("/nonexistent/file.txt", verbose=False)
    assert stats["processed"] == 0


def test_ingest_unsupported_format(tmp_path):
    """Test that unsupported file formats are rejected."""
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("content")
    stats = ingest_file(bad_file, verbose=False)
    assert stats["processed"] == 0


def test_cache_path_local():
    """Test cache path for local (relative) source keys."""
    path = _get_cache_path("document.pdf")
    assert "local" in str(path)
    assert path.name == "document.pdf.md"


def test_cache_path_external():
    """Test cache path for external (absolute) source keys."""
    path = _get_cache_path("/Users/test/docs/report.pdf")
    assert "external" in str(path)
    assert "report.pdf.md" in path.name
