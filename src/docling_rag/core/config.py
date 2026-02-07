"""Configuration management for the RAG system."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""

    data_dir: Path = field(default_factory=lambda: Path("data"))
    lancedb_dir: Path = field(default_factory=lambda: Path("lancedb_data"))
    cache_dir: Path = field(default_factory=lambda: Path(".docling_cache"))
    hash_file: Path = field(default_factory=lambda: Path(".docling_hashes.json"))
    embed_model: str = "Snowflake/snowflake-arctic-embed-m-v2.0"
    max_tokens: int = 1500
    default_results: int = 5
    # Reranking
    enable_reranking: bool = True
    rerank_candidates: int = 20
    # Hybrid search
    enable_hybrid_search: bool = True
    # Contextual retrieval (requires Ollama)
    enable_contextual_retrieval: bool = False
    ollama_model: str = "qwen2.5:1.5b"
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def load(cls, path: Path | None = None) -> "RAGConfig":
        """Load config from TOML file, falling back to defaults."""
        if path is None:
            path = Path("rag.toml")
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            # Convert string paths to Path objects
            for key in ("data_dir", "lancedb_dir", "cache_dir", "hash_file"):
                if key in data:
                    data[key] = Path(data[key])
            return cls(**data)
        return cls()


# Global default config (can be overridden)
_config: RAGConfig | None = None


def get_config() -> RAGConfig:
    """Get the current configuration (loads from rag.toml if exists)."""
    global _config
    if _config is None:
        _config = RAGConfig.load()
    return _config


def set_config(config: RAGConfig) -> None:
    """Override the global configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to reload from file on next access."""
    global _config
    _config = None
