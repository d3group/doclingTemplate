"""LLM-based chunk contextualization for improved retrieval quality.

Uses a local Ollama model to generate a short context prefix for each chunk,
explaining where it fits in the overall document. This is done once at
ingestion time, not per query.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from docling_rag.core.config import get_config

logger = logging.getLogger(__name__)

_CONTEXT_PROMPT = """\
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short succinct context (1-2 sentences) to situate this chunk within \
the overall document for the purposes of improving search retrieval. \
Answer only with the context, nothing else."""


def _call_ollama(prompt: str) -> str | None:
    """Call the Ollama API and return the response text, or None on failure."""
    config = get_config()
    url = f"{config.ollama_base_url}/api/generate"

    try:
        response = httpx.post(
            url,
            json={
                "model": config.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except httpx.ConnectError:
        logger.warning("Ollama not reachable at %s -- skipping contextualization", url)
        return None
    except (httpx.HTTPStatusError, httpx.TimeoutException):
        logger.debug("Ollama call failed", exc_info=True)
        return None


def contextualize_chunk(chunk_text: str, full_document_md: str) -> str:
    """
    Generate a contextualized version of a chunk by prepending LLM-generated context.

    If Ollama is not available, returns the original chunk text unchanged.

    Args:
        chunk_text: The raw chunk text
        full_document_md: The full document markdown for context

    Returns:
        Chunk text with context prefix prepended, or original text on failure
    """
    prompt = _CONTEXT_PROMPT.format(
        document=full_document_md[:8000],  # Truncate long documents
        chunk=chunk_text,
    )

    context = _call_ollama(prompt)
    if context:
        return f"{context}\n\n{chunk_text}"
    return chunk_text


def contextualize_chunks(
    chunk_texts: list[str],
    full_document_md: str,
    max_workers: int = 4,
) -> list[str]:
    """
    Generate contextualized versions of multiple chunks from the same document.

    Ollama HTTP calls are I/O-bound, so threads parallelize them effectively.

    Args:
        chunk_texts: List of raw chunk texts
        full_document_md: The full document markdown for context
        max_workers: Number of parallel Ollama requests

    Returns:
        List of contextualized chunk texts (same length as input)
    """
    results: list[str | None] = [None] * len(chunk_texts)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(contextualize_chunk, text, full_document_md): i
            for i, text in enumerate(chunk_texts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            if results[idx] != chunk_texts[idx]:
                logger.debug("Contextualized chunk %d/%d", idx + 1, len(chunk_texts))

    return results  # type: ignore[return-value]


def is_ollama_available() -> bool:
    """Check if Ollama is running and reachable."""
    config = get_config()
    try:
        response = httpx.get(f"{config.ollama_base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False
