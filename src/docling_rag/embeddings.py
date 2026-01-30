"""Shared embedding configuration for ChromaDB."""

import torch
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"
MAX_TOKENS = 500  # BGE also has 512 limit, stay under with safety margin


def get_device() -> str:
    """Detect the best available device for inference."""
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    return "cpu"


def get_embedding_function():
    """Get the embedding function for ChromaDB."""
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_ID,
        device=get_device(),
    )
