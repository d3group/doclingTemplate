"""Shared test fixtures for Docling RAG tests."""

import unittest.mock

import pytest

from docling_rag.core.config import RAGConfig, reset_config, set_config
from docling_rag.core.db import reset_clients
from docling_rag.core.embeddings import reset_embedding_cache
from docling_rag.core.ingest import reset_ingest_cache


@pytest.fixture(autouse=True)
def _force_cpu():
    """Force CPU device for all tests to avoid MPS OOM issues."""
    with unittest.mock.patch("docling_rag.core.embeddings.get_device", return_value="cpu"):
        yield


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path):
    """
    Ensure every test uses an isolated config with temp directories.

    This prevents tests from polluting each other or the real data/lancedb directories.
    """
    config = RAGConfig(
        data_dir=tmp_path / "data",
        lancedb_dir=tmp_path / "lancedb_data",
        cache_dir=tmp_path / ".docling_cache",
        hash_file=tmp_path / ".docling_hashes.json",
        enable_reranking=False,
        enable_hybrid_search=False,
    )
    config.data_dir.mkdir()
    set_config(config)
    yield config
    # Reset all caches so each test starts fresh
    reset_config()
    reset_clients()
    reset_embedding_cache()
    reset_ingest_cache()


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a simple text file for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "sample.txt"
    file_path.write_text(
        "This is a sample document about artificial intelligence and machine learning."
    )
    return file_path


@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create a markdown file for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "readme.md"
    file_path.write_text(
        "# Project Documentation\n\n"
        "## Overview\n\n"
        "This project implements a RAG system using Docling and LanceDB.\n\n"
        "## Features\n\n"
        "- Document parsing with Docling\n"
        "- Vector storage with LanceDB\n"
        "- Semantic search capabilities\n"
    )
    return file_path


@pytest.fixture
def sample_code_file(tmp_path):
    """Create a Python code file for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "example.py"
    file_path.write_text(
        "def hello(name: str) -> str:\n"
        '    """Greet someone by name."""\n'
        '    return f"Hello, {name}!"\n'
    )
    return file_path


@pytest.fixture
def external_text_file(tmp_path):
    """Create a text file outside the data directory for testing external ingestion."""
    external_dir = tmp_path / "external_docs"
    external_dir.mkdir()
    file_path = external_dir / "external.txt"
    file_path.write_text("This is an external document that lives outside the data directory.")
    return file_path
