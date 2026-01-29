"""Docling RAG - Local RAG system using Docling and ChromaDB."""

from docling_rag.download import download_author_papers
from docling_rag.ingest import ingest_documents
from docling_rag.query import query

__all__ = ["ingest_documents", "query", "download_author_papers"]
__version__ = "0.1.0"
