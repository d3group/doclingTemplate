"""Query interface for the RAG system."""

from pathlib import Path

import chromadb

from docling_rag.embeddings import get_embedding_function

CHROMA_DIR = Path("chroma_db")


def query(question: str, n_results: int = 5) -> list[dict]:
    """
    Query the knowledge base and return relevant chunks with sources.

    Args:
        question: The query string
        n_results: Number of results to return

    Returns:
        List of dicts with keys: text, source, page (if available), distance
    """
    if not CHROMA_DIR.exists():
        return []

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(
            "documents",
            embedding_function=get_embedding_function(),
        )
    except ValueError:
        # Collection doesn't exist
        return []

    # Query the collection
    results = collection.query(query_texts=[question], n_results=n_results)

    # Format results
    formatted = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            result = {
                "text": doc,
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "distance": results["distances"][0][i] if results["distances"] else None,
            }
            # Add page if available
            if "page" in results["metadatas"][0][i]:
                result["page"] = results["metadatas"][0][i]["page"]
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
