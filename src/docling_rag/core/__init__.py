"""Core functionality for Docling RAG - no CLI/MCP dependencies."""

from docling_rag.core.config import RAGConfig, get_config, reset_config, set_config
from docling_rag.core.db import get_db, get_table, reset_clients
from docling_rag.core.embeddings import get_device, get_embedding_function, reset_embedding_cache
from docling_rag.core.ingest import (
    SUPPORTED_EXTENSIONS,
    ingest_documents,
    ingest_file,
    read_cached_document,
    reset_ingest_cache,
)
from docling_rag.core.query import (
    delete_source,
    format_results,
    get_stats,
    list_sources,
    query,
)

__all__ = [
    # Config
    "RAGConfig",
    "get_config",
    "set_config",
    "reset_config",
    # DB
    "get_db",
    "get_table",
    "reset_clients",
    # Embeddings
    "get_embedding_function",
    "get_device",
    "reset_embedding_cache",
    # Ingest
    "ingest_documents",
    "ingest_file",
    "read_cached_document",
    "reset_ingest_cache",
    "SUPPORTED_EXTENSIONS",
    # Query
    "query",
    "format_results",
    "get_stats",
    "list_sources",
    "delete_source",
]
