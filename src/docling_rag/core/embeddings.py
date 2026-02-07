"""Shared embedding configuration for LanceDB."""

import torch
from lancedb.embeddings import register
from lancedb.embeddings.sentence_transformers import SentenceTransformerEmbeddings

from docling_rag.core.config import get_config

_embed_fn = None
_cached_model_name: str | None = None


def get_device() -> str:
    """Detect the best available device for inference."""
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    return "cpu"


@register("docling-rag-embeddings")
class DoclingEmbeddings(SentenceTransformerEmbeddings):
    """Custom embedding function that properly handles models needing config overrides."""

    def get_embedding_model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.name,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            config_kwargs={"use_memory_efficient_attention": False},
        )


def get_embedding_function():
    """Get the cached LanceDB embedding function."""
    from lancedb.embeddings import get_registry

    global _embed_fn, _cached_model_name
    config = get_config()

    # Invalidate cache if model changed
    if _embed_fn is not None and _cached_model_name == config.embed_model:
        return _embed_fn

    _embed_fn = (
        get_registry()
        .get("docling-rag-embeddings")
        .create(
            name=config.embed_model,
            device=get_device(),
        )
    )
    _cached_model_name = config.embed_model
    return _embed_fn


def reset_embedding_cache() -> None:
    """Reset the cached embedding function (useful for testing)."""
    global _embed_fn, _cached_model_name
    _embed_fn = None
    _cached_model_name = None
