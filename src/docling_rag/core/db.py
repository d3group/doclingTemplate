"""Shared LanceDB connection and table helpers."""

import logging

import lancedb
import pyarrow as pa
from lancedb.embeddings.base import EmbeddingFunctionConfig

from docling_rag.core.config import get_config
from docling_rag.core.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

_db: lancedb.DBConnection | None = None

TABLE_NAME = "documents"


def get_db() -> lancedb.DBConnection:
    """Get or create the cached LanceDB connection."""
    global _db
    config = get_config()
    config.lancedb_dir.mkdir(parents=True, exist_ok=True)

    if _db is None:
        _db = lancedb.connect(str(config.lancedb_dir))
    return _db


def _build_arrow_schema() -> pa.Schema:
    """Build the PyArrow schema for the documents table."""
    embed_fn = get_embedding_function()
    ndims = embed_fn.ndims()

    return pa.schema(
        [
            pa.field("text", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), ndims)),
            pa.field("source", pa.utf8()),
            pa.field("chunk_index", pa.int32()),
            pa.field("ingested_at", pa.utf8()),
            pa.field("embed_model", pa.utf8()),
            pa.field("title", pa.utf8()),
            pa.field("section_heading", pa.utf8()),
            pa.field("page", pa.int32()),
        ]
    )


def _build_embedding_configs() -> list[EmbeddingFunctionConfig]:
    """Build the embedding function config for auto-embed on add/search."""
    embed_fn = get_embedding_function()
    return [
        EmbeddingFunctionConfig(
            source_column="text",
            vector_column="vector",
            function=embed_fn,
        )
    ]


def get_table(*, create: bool = True) -> lancedb.table.Table | None:
    """
    Get the documents table.

    Args:
        create: If True, create table if it doesn't exist.
                If False, return None when table doesn't exist.

    Returns:
        The LanceDB table, or None if create=False and table doesn't exist.
    """
    db = get_db()

    if create:
        # Use exist_ok=True to handle concurrent/repeated calls gracefully
        return db.create_table(
            TABLE_NAME,
            schema=_build_arrow_schema(),
            embedding_functions=_build_embedding_configs(),
            exist_ok=True,
        )

    # For read-only access, check if table exists first
    if TABLE_NAME in db.list_tables().tables:
        return db.open_table(TABLE_NAME)

    return None


def reset_clients() -> None:
    """Reset cached DB connection (useful for testing)."""
    global _db
    _db = None
