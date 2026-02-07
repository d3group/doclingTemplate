"""Query interface for the RAG system."""

import logging

from docling_rag.core.config import get_config
from docling_rag.core.db import get_table

logger = logging.getLogger(__name__)


def _get_reranker():
    """Get the cross-encoder reranker for hybrid search."""
    from lancedb.rerankers import CrossEncoderReranker

    return CrossEncoderReranker(column="text")


def query(
    question: str,
    n_results: int | None = None,
    source_filter: str | None = None,
    where: dict | None = None,
) -> list[dict]:
    """
    Query the knowledge base and return relevant chunks with sources.

    Args:
        question: The query string
        n_results: Number of results to return (defaults to config value)
        source_filter: Optional source file path to restrict results to
        where: Optional where clause dict (unused, kept for API compat)

    Returns:
        List of dicts with keys: text, source, page (if available), distance
    """
    config = get_config()

    if n_results is None:
        n_results = config.default_results

    table = get_table(create=False)
    if table is None:
        return []

    if table.count_rows() == 0:
        return []

    # Choose query type based on config
    query_type = "hybrid" if config.enable_hybrid_search else "vector"

    # For hybrid search, we need an FTS index. Fall back to vector if not available.
    if query_type == "hybrid":
        try:
            search = table.search(question, query_type="hybrid")
        except Exception:
            # FTS index may not exist yet, fall back to vector search
            logger.debug("FTS index not available, falling back to vector search")
            search = table.search(question, query_type="vector")
    else:
        search = table.search(question, query_type="vector")

    # Apply source filter
    if source_filter:
        search = search.where(f"source = '{source_filter}'")

    # Determine how many candidates to fetch
    fetch_limit = config.rerank_candidates if config.enable_reranking else n_results

    # Apply reranking if enabled
    if config.enable_reranking:
        try:
            reranker = _get_reranker()
            search = search.rerank(reranker=reranker)
        except Exception:
            logger.debug("Reranking failed, returning raw results")

    results = search.limit(fetch_limit).to_list()

    # Trim to requested number (reranker may have fetched more candidates)
    results = results[:n_results]

    # Format results
    formatted = []
    for row in results:
        result = {
            "text": row.get("text", ""),
            "source": row.get("source", "unknown"),
            "distance": row.get("_distance") or row.get("_relevance_score"),
        }
        page = row.get("page")
        if page is not None and page >= 0:
            result["page"] = page
        formatted.append(result)

    return formatted


def format_results(results: list[dict]) -> str:
    """Format query results for display."""
    if not results:
        return "No results found. Have you run 'ingest' yet?"

    output = []
    for i, r in enumerate(results, 1):
        source_info = r["source"]
        if "page" in r:
            source_info += f", page {r['page']}"

        distance_str = f" (distance: {r['distance']:.3f})" if r["distance"] else ""

        output.append(f"[{i}] Source: {source_info}{distance_str}")
        output.append(f"    {r['text'][:500]}...")
        output.append("")

    return "\n".join(output)


def get_stats() -> dict:
    """
    Get statistics about the knowledge base.

    Returns:
        Dict with total_chunks, total_documents, and list of sources
    """
    table = get_table(create=False)
    if table is None:
        return {"total_chunks": 0, "total_documents": 0, "sources": []}

    count = table.count_rows()
    if count == 0:
        return {"total_chunks": 0, "total_documents": 0, "sources": []}

    # Get unique sources and their chunk counts
    source_col = table.to_arrow().column("source").to_pylist()
    sources = {}
    for s in source_col:
        sources[s] = sources.get(s, 0) + 1

    return {
        "total_chunks": count,
        "total_documents": len(sources),
        "sources": [{"name": name, "chunks": cnt} for name, cnt in sorted(sources.items())],
    }


def list_sources() -> list[str]:
    """
    List all indexed document sources.

    Returns:
        List of source file paths
    """
    stats = get_stats()
    return [s["name"] for s in stats["sources"]]


def delete_source(source: str) -> bool:
    """
    Delete all chunks for a specific source document.

    Args:
        source: The source file path to delete

    Returns:
        True if deleted, False if source not found
    """
    table = get_table(create=False)
    if table is None:
        return False

    # Check if source exists
    count_before = table.count_rows()
    table.delete(f"source = '{source}'")
    count_after = table.count_rows()

    return count_after < count_before
