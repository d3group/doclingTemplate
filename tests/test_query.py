"""Tests for the query module."""

from docling_rag.core.ingest import ingest_documents
from docling_rag.core.query import delete_source, format_results, get_stats, list_sources, query


def test_query_empty_db():
    """Test querying an empty database returns empty results."""
    results = query("anything")
    assert results == []


def test_query_round_trip(sample_text_file):
    """Test ingesting a document then querying it."""
    ingest_documents(verbose=False)
    results = query("artificial intelligence", n_results=3)
    assert len(results) >= 1
    assert results[0]["source"] == "sample.txt"
    assert results[0]["text"]
    assert results[0]["distance"] is not None


def test_query_with_source_filter(sample_text_file, sample_markdown_file):
    """Test filtering query results by source."""
    ingest_documents(verbose=False)

    # Query filtered to just the text file
    results = query("documentation", source_filter="sample.txt")
    for r in results:
        assert r["source"] == "sample.txt"


def test_get_stats_empty():
    """Test stats on empty database."""
    stats = get_stats()
    assert stats["total_chunks"] == 0
    assert stats["total_documents"] == 0
    assert stats["sources"] == []


def test_get_stats_after_ingest(sample_text_file):
    """Test stats after ingesting a document."""
    ingest_documents(verbose=False)
    stats = get_stats()
    assert stats["total_chunks"] >= 1
    assert stats["total_documents"] == 1
    assert len(stats["sources"]) == 1
    assert stats["sources"][0]["name"] == "sample.txt"


def test_list_sources(sample_text_file, sample_markdown_file):
    """Test listing indexed sources."""
    ingest_documents(verbose=False)
    sources = list_sources()
    assert "sample.txt" in sources
    assert "readme.md" in sources


def test_delete_source(sample_text_file):
    """Test deleting a source from the database."""
    ingest_documents(verbose=False)
    assert delete_source("sample.txt") is True
    # Should be empty now
    stats = get_stats()
    assert stats["total_chunks"] == 0


def test_delete_nonexistent_source(sample_text_file):
    """Test deleting a source that doesn't exist."""
    ingest_documents(verbose=False)
    assert delete_source("nonexistent.pdf") is False


def test_format_results_empty():
    """Test formatting empty results."""
    output = format_results([])
    assert "No results found" in output


def test_format_results():
    """Test formatting query results."""
    results = [
        {"text": "Some text about AI", "source": "doc.pdf", "distance": 0.123, "page": 5},
        {"text": "More content", "source": "notes.md", "distance": 0.456},
    ]
    output = format_results(results)
    assert "doc.pdf" in output
    assert "page 5" in output
    assert "0.123" in output
    assert "notes.md" in output


def test_query_metadata_present(sample_text_file):
    """Test that ingested chunks have the expected metadata fields."""
    ingest_documents(verbose=False)
    from docling_rag.core.db import get_table

    table = get_table(create=False)
    assert table is not None
    arrow_table = table.to_arrow()
    for i in range(arrow_table.num_rows):
        assert arrow_table.column("source")[i].as_py()
        assert arrow_table.column("chunk_index")[i].as_py() is not None
        assert arrow_table.column("ingested_at")[i].as_py()
        assert arrow_table.column("embed_model")[i].as_py()
