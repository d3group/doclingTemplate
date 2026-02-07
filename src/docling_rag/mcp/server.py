"""FastMCP server for Docling RAG."""

import sys

from fastmcp import FastMCP

from docling_rag.core import (
    SUPPORTED_EXTENSIONS,
    delete_source,
    get_config,
    get_stats,
    ingest_documents,
    ingest_file,
    list_sources,
    query,
    read_cached_document,
)

mcp = FastMCP(
    name="docling-rag",
    instructions="Local RAG system for document retrieval and Q&A using Docling and LanceDB. "
    "Use query_knowledge to search documents, ingest_documents_tool to add documents, "
    "read_document to get full document context, "
    "and get_database_stats to check the knowledge base status.",
)


def _log(msg: str) -> None:
    """Log to stderr so MCP stdio transport isn't corrupted."""
    print(msg, file=sys.stderr, flush=True)


# ============================================================================
# Tools (Actions)
# ============================================================================


@mcp.tool(description="Search the knowledge base for relevant information")
def query_knowledge(question: str, n_results: int = 5, source: str | None = None) -> list[dict]:
    """
    Query the knowledge base with a natural language question.

    Args:
        question: The search query or question
        n_results: Number of results to return (default: 5)
        source: Optional source file to restrict search to (as shown in list_indexed_sources)

    Returns:
        List of relevant chunks with source, page (if available), and relevance score
    """
    return query(question, n_results=n_results, source_filter=source)


@mcp.tool(description="Ingest documents from a directory into the knowledge base")
def ingest_documents_tool(path: str = "data", verbose: bool = False) -> dict:
    """
    Process and index documents from the specified directory.

    Supports: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, and code files.
    Uses hash-based change detection to only process new or modified files.

    Args:
        path: Directory containing documents to ingest (default: "data")
        verbose: Print progress information (default: False)

    Returns:
        Statistics: {"processed": int, "deleted": int, "chunks_added": int}
    """
    if verbose:
        _log(f"Starting ingestion from: {path}")
    result = ingest_documents(data_dir=path, verbose=verbose)
    if verbose:
        _log(f"Ingestion complete: {result}")
    return result


@mcp.tool(description="Ingest a single file from any location")
def ingest_file_tool(file_path: str, verbose: bool = False) -> dict:
    """
    Process and index a single file from any location.

    Useful for adding files outside the default data directory.
    The file's absolute path is tracked for change detection.

    Args:
        file_path: Path to the file to ingest
        verbose: Print progress information (default: False)

    Returns:
        Statistics: {"processed": int, "chunks_added": int}
    """
    if verbose:
        _log(f"Ingesting file: {file_path}")
    result = ingest_file(file_path=file_path, verbose=verbose)
    if verbose:
        _log(f"File ingestion complete: {result}")
    return result


@mcp.tool(description="Read the full parsed content of an indexed document")
def read_document(source: str) -> dict:
    """
    Read the full parsed markdown content of an indexed document.

    This returns the complete document content (cached from parsing),
    useful when search chunks don't provide enough context.

    Args:
        source: The source file path (as shown in list_indexed_sources)

    Returns:
        {"source": str, "content": str} or {"source": str, "error": str}
    """
    content = read_cached_document(source)
    if content is None:
        return {
            "source": source,
            "error": "No cached content found. The document may not have been ingested yet, "
            "or the cache may have been cleared.",
        }
    return {"source": source, "content": content}


@mcp.tool(description="Get database statistics")
def get_database_stats() -> dict:
    """
    Get statistics about the knowledge base.

    Returns:
        Dict with total_chunks, total_documents, and list of sources with chunk counts
    """
    return get_stats()


@mcp.tool(description="List all indexed documents")
def list_indexed_sources() -> list[str]:
    """
    List all documents that have been indexed in the knowledge base.

    Returns:
        List of source file paths
    """
    return list_sources()


@mcp.tool(description="Remove a document from the knowledge base")
def delete_document(source: str) -> dict:
    """
    Delete all chunks for a specific source document.

    Args:
        source: The source file path to remove (as shown in list_indexed_sources)

    Returns:
        {"deleted": True/False, "source": str}
    """
    success = delete_source(source)
    return {"deleted": success, "source": source}


# ============================================================================
# Resources (Read-only data)
# ============================================================================


@mcp.resource("rag://stats")
def resource_stats() -> str:
    """Current database statistics."""
    stats = get_stats()
    lines = [
        "# Knowledge Base Statistics",
        "",
        f"- Total chunks: {stats['total_chunks']}",
        f"- Total documents: {stats['total_documents']}",
    ]
    if stats["sources"]:
        lines.extend(["", "## Documents:"])
        for src in stats["sources"]:
            lines.append(f"- {src['name']} ({src['chunks']} chunks)")
    return "\n".join(lines)


@mcp.resource("rag://sources")
def resource_sources() -> str:
    """List of indexed documents."""
    sources = list_sources()
    if not sources:
        return "No documents indexed yet."
    return "\n".join(f"- {s}" for s in sources)


@mcp.resource("rag://config")
def resource_config() -> str:
    """Current configuration."""
    config = get_config()
    return f"""# RAG Configuration

- Data directory: {config.data_dir}
- LanceDB directory: {config.lancedb_dir}
- Cache directory: {config.cache_dir}
- Hash file: {config.hash_file}
- Embedding model: {config.embed_model}
- Max tokens per chunk: {config.max_tokens}
- Default results: {config.default_results}
- Hybrid search: {config.enable_hybrid_search}
- Reranking: {config.enable_reranking}
"""


@mcp.resource("rag://supported-formats")
def resource_formats() -> str:
    """Supported file formats for ingestion."""
    return f"""# Supported File Formats

The following file extensions are supported for ingestion:

{", ".join(sorted(SUPPORTED_EXTENSIONS))}

## Categories:
- **Documents**: PDF, DOCX, PPTX, XLSX, HTML
- **Images**: PNG, JPG, JPEG (OCR processed)
- **Text/Markdown**: MD, TXT
- **Code**: PY, JS, TS, JSON, YAML, TOML, SH, CSS
"""


# ============================================================================
# Prompts (Agent guidance)
# ============================================================================


@mcp.prompt()
def rag_usage() -> str:
    """Comprehensive guide for AI agents on using this RAG system."""
    return """# Docling RAG System - Agent Guide

## Overview
This is a local document retrieval system that indexes documents using Docling
and stores embeddings in LanceDB with hybrid search (vector + BM25) and
cross-encoder reranking for high-quality retrieval.

## Available Tools

### query_knowledge(question, n_results=5)
Search the indexed documents for relevant information.
Uses hybrid search (semantic + keyword) with cross-encoder reranking.

**Returns:**
- `text`: The relevant chunk content
- `source`: Source file path
- `page`: Page number (if available, for PDFs)
- `distance`: Relevance score

**Best practices:**
- Use specific, detailed questions for better results
- Request more results (n_results=10) for comprehensive research
- Check the source to verify information context

### read_document(source)
Read the full parsed markdown content of an indexed document.
Use this when chunks alone don't provide enough context.

### ingest_documents_tool(path="data", verbose=False)
Index documents from a directory into the knowledge base.

**Supports:**
- Documents: PDF, DOCX, PPTX, XLSX
- Web: HTML
- Text: MD, TXT
- Images: PNG, JPG (OCR)
- Code: PY, JS, TS, JSON, YAML, TOML, SH, CSS

Only processes new or modified files (hash-based change detection).

### ingest_file_tool(file_path, verbose=False)
Index a single file from any location.

### get_database_stats()
View database statistics (total chunks, documents).

### list_indexed_sources()
List all indexed documents.

### delete_document(source)
Remove a document from the index.

## Resources

- `rag://stats` - Database statistics
- `rag://sources` - Indexed document list
- `rag://config` - Current configuration
- `rag://supported-formats` - List of supported file formats

## Typical Workflow

1. **Check status**: Use `get_database_stats()` to see what's indexed
2. **Ingest if needed**: Use `ingest_documents_tool()` to add new documents
3. **Query**: Use `query_knowledge()` to search for information
4. **Expand context**: Use `read_document(source)` when chunks aren't enough
"""


@mcp.prompt()
def query_tips() -> str:
    """Tips for formulating effective RAG queries."""
    return """# Query Formulation Tips

## Good Queries
- "What are the key features of X?"
- "How does the authentication system work?"
- "What parameters does function Y accept?"
- "Explain the architecture of the payment module"
- "What are the requirements for deploying this application?"

## Less Effective Queries
- "X" (too vague - add context)
- "Tell me everything" (too broad - be specific)
- Single keywords (add context about what you're looking for)

## Query Strategies

### For Code Documentation
- "How is [function/class] implemented?"
- "What does [module] depend on?"
- "What are the error handling patterns in [file]?"

### For Research Papers
- "What methodology was used for [topic]?"
- "What are the main findings about [subject]?"
- "How does [paper] compare to previous work?"

### For Technical Documentation
- "How do I configure [feature]?"
- "What are the API endpoints for [service]?"
- "What are the system requirements?"

## Interpreting Results

- **Relevance scores**: Higher is better when reranking is enabled
- **Multiple chunks from same source**: Strong relevance indicator
- **Check page numbers**: Navigate to original for full context
- **Consider source types**: PDFs may have OCR artifacts, code files are exact

## When Results Are Poor

1. Rephrase with different terminology
2. Check if relevant documents are indexed (`list_indexed_sources()`)
3. Try broader or narrower query scope
4. Verify documents were processed successfully (`get_database_stats()`)
5. Use `read_document(source)` to check the full document content
"""


def run_server():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run_server()
