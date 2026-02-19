"""Tests for the configuration module."""

from pathlib import Path

from docling_rag.core.config import RAGConfig, get_config, reset_config, set_config


def test_default_config():
    """Test that default config values are sensible."""
    config = RAGConfig()
    assert config.data_dir == Path("data")
    assert config.lancedb_dir == Path("lancedb_data")
    assert config.cache_dir == Path(".docling_cache")
    assert config.hash_file == Path(".docling_hashes.json")
    assert config.embed_model == "Snowflake/snowflake-arctic-embed-m-v2.0"
    assert config.max_tokens == 1500
    assert config.default_results == 5
    assert config.enable_reranking is True
    assert config.rerank_candidates == 20
    assert config.enable_hybrid_search is True
    assert config.enable_contextual_retrieval is True
    assert config.ollama_model == "qwen2.5:1.5b"
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.contextual_workers == 4


def test_config_from_toml(tmp_path):
    """Test loading config from a TOML file."""
    toml_path = tmp_path / "rag.toml"
    toml_path.write_text(
        'data_dir = "my_docs"\nlancedb_dir = "my_vectors"\nmax_tokens = 300\ndefault_results = 10\n'
    )
    config = RAGConfig.load(toml_path)
    assert config.data_dir == Path("my_docs")
    assert config.lancedb_dir == Path("my_vectors")
    assert config.max_tokens == 300
    assert config.default_results == 10
    # Unset fields should keep defaults
    assert config.embed_model == "Snowflake/snowflake-arctic-embed-m-v2.0"


def test_config_missing_file():
    """Test that missing config file returns defaults."""
    config = RAGConfig.load(Path("/nonexistent/rag.toml"))
    assert config.data_dir == Path("data")


def test_get_set_reset_config():
    """Test the global config get/set/reset cycle."""
    custom = RAGConfig(max_tokens=999)
    set_config(custom)
    assert get_config().max_tokens == 999

    reset_config()
    # After reset, get_config will reload from file (or defaults)
    # Since we're in a test with _isolated_config fixture, the fixture handles this
