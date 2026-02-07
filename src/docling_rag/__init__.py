"""Docling RAG - Local RAG system using Docling and LanceDB."""

from docling_rag.core import (
    RAGConfig,
    get_config,
    ingest_documents,
    ingest_file,
    query,
    set_config,
)

__all__ = [
    "ingest_documents",
    "ingest_file",
    "query",
    "RAGConfig",
    "get_config",
    "set_config",
]
__version__ = "0.1.0"
