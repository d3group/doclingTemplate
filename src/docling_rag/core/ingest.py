"""Document ingestion pipeline using Docling and LanceDB."""

import hashlib
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from transformers import logging as tf_logging

from docling_rag.core.config import get_config
from docling_rag.core.db import get_table

logger = logging.getLogger(__name__)

# Suppress transformers token length warnings (we handle chunk sizing ourselves)
tf_logging.set_verbosity_error()

# Document formats (native Docling support)
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".png", ".jpg", ".jpeg"}

# Text/code formats (converted to markdown for processing)
TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
    ".css",
}

# Language hints for code syntax highlighting in markdown
LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".css": "css",
    ".html": "html",
    ".htm": "html",
}

# All supported extensions
SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | TEXT_EXTENSIONS

# Module-level caches for heavy objects
_converter: DocumentConverter | None = None
_chunker: HybridChunker | None = None
_chunker_model: str | None = None


def get_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def convert_text_to_markdown(file_path: Path) -> str:
    """Convert a text/code file to markdown format for Docling processing."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    ext = file_path.suffix.lower()
    lang = LANG_MAP.get(ext, "")

    # For markdown files, return as-is
    if ext == ".md":
        return content

    # For plain text, just return content
    if ext == ".txt":
        return content

    # For code files, wrap in a code block with filename header
    return f"# {file_path.name}\n\n```{lang}\n{content}\n```"


def load_hashes() -> dict[str, str]:
    """Load previously computed file hashes."""
    config = get_config()
    if config.hash_file.exists():
        return json.loads(config.hash_file.read_text())
    return {}


def save_hashes(hashes: dict[str, str]) -> None:
    """Save file hashes to disk."""
    config = get_config()
    config.hash_file.write_text(json.dumps(hashes, indent=2))


def _get_cache_path(source_key: str) -> Path:
    """
    Get the markdown cache file path for a source key.

    For relative source keys (from data/): .docling_cache/local/<source_key>.md
    For absolute source keys (external files): .docling_cache/external/<md5>_<filename>.md
    """
    config = get_config()
    cache_dir = config.cache_dir

    if source_key.startswith("/"):
        # External file -- use hash of path to avoid filesystem issues
        path_hash = hashlib.md5(source_key.encode()).hexdigest()[:12]
        filename = Path(source_key).name
        return cache_dir / "external" / f"{path_hash}_{filename}.md"
    else:
        return cache_dir / "local" / f"{source_key}.md"


def _load_cache_index() -> dict[str, str]:
    """Load the cache index mapping source_key -> cache filename."""
    config = get_config()
    index_path = config.cache_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def _save_cache_index(index: dict[str, str]) -> None:
    """Save the cache index."""
    config = get_config()
    index_path = config.cache_dir / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2))


def _write_markdown_cache(source_key: str, md_content: str) -> None:
    """Write parsed markdown to cache."""
    cache_path = _get_cache_path(source_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(md_content, encoding="utf-8")

    # Update index
    index = _load_cache_index()
    index[source_key] = str(cache_path)
    _save_cache_index(index)


def _read_markdown_cache(source_key: str) -> str | None:
    """Read cached markdown for a source key, or None if not cached."""
    cache_path = _get_cache_path(source_key)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


def _delete_markdown_cache(source_key: str) -> None:
    """Delete cached markdown for a source key."""
    cache_path = _get_cache_path(source_key)
    if cache_path.exists():
        cache_path.unlink()

    # Update index
    index = _load_cache_index()
    index.pop(source_key, None)
    _save_cache_index(index)


def read_cached_document(source_key: str) -> str | None:
    """
    Read the full cached markdown for a document.

    This is the public API for retrieving cached parsed documents,
    used by the MCP read_document tool.

    Args:
        source_key: The source key as shown in list_sources()

    Returns:
        Full markdown content, or None if not cached
    """
    return _read_markdown_cache(source_key)


def _is_hidden_path(file_path: Path, base_dir: Path) -> bool:
    """
    Check if a file is hidden or inside a hidden directory.

    Hidden means the name starts with '.' (e.g., .venv, .git, .claude).
    """
    # Check if the file itself is hidden
    if file_path.name.startswith("."):
        return True

    # Check if any parent directory (up to base_dir) is hidden
    try:
        relative = file_path.relative_to(base_dir)
        for part in relative.parts[:-1]:  # Exclude the filename itself
            if part.startswith("."):
                return True
    except ValueError:
        pass

    return False


def _get_source_key(file_path: Path, data_dir: Path | None) -> str:
    """
    Get the source key for a file (used in metadata and hash tracking).

    For files within data_dir, uses relative path.
    For external files, uses absolute path.
    """
    if data_dir is not None:
        try:
            return str(file_path.relative_to(data_dir))
        except ValueError:
            pass
    return str(file_path.resolve())


def get_documents_to_process(
    data_dir: Path,
    use_absolute_paths: bool = False,
) -> tuple[list[Path], list[str], dict[str, str]]:
    """
    Determine which documents in a directory need processing.

    Args:
        data_dir: Directory to scan for documents
        use_absolute_paths: If True, use absolute paths as source keys (for external directories)

    Returns:
        - new_or_changed: Files that need to be (re)processed
        - deleted: File paths that were removed and need cleanup
        - current_hashes: Hash mapping for all current files
    """
    old_hashes = load_hashes()
    current_hashes: dict[str, str] = {}
    new_or_changed: list[Path] = []

    # Carry forward hashes for external files (absolute paths) since they're
    # not scanned by this directory walk
    for path, file_hash in old_hashes.items():
        if path.startswith("/"):
            current_hashes[path] = file_hash

    # Find all supported files in directory (excluding hidden files/folders)
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in data_dir.rglob(f"*{ext}"):
            # Skip hidden files and files in hidden directories
            if _is_hidden_path(file_path, data_dir):
                continue

            if use_absolute_paths:
                source_key = str(file_path.resolve())
            else:
                source_key = str(file_path.relative_to(data_dir))

            file_hash = get_file_hash(file_path)
            current_hashes[source_key] = file_hash

            # Check if new or changed
            if source_key not in old_hashes or old_hashes[source_key] != file_hash:
                new_or_changed.append(file_path)

    # Only detect deletions for the default data/ folder (relative paths)
    # External directories use absolute paths and don't trigger deletions
    if not use_absolute_paths:
        deleted = [
            path for path in old_hashes if not path.startswith("/") and path not in current_hashes
        ]
    else:
        deleted = []

    return new_or_changed, deleted, current_hashes


def _get_converter() -> DocumentConverter:
    """Get or create the cached document converter."""
    global _converter
    if _converter is None:
        _converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.XLSX,
                InputFormat.HTML,
                InputFormat.IMAGE,
                InputFormat.MD,
            ]
        )
    return _converter


def _get_chunker() -> HybridChunker:
    """Get or create the cached document chunker."""
    global _chunker, _chunker_model
    config = get_config()

    # Invalidate if model changed
    if _chunker is not None and _chunker_model == config.embed_model:
        return _chunker

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(config.embed_model),
        max_tokens=config.max_tokens,
    )
    _chunker = HybridChunker(tokenizer=tokenizer)
    _chunker_model = config.embed_model
    return _chunker


def reset_ingest_cache() -> None:
    """Reset cached converter and chunker (useful for testing)."""
    global _converter, _chunker, _chunker_model
    _converter = None
    _chunker = None
    _chunker_model = None


def _check_embed_model_consistency(table, config) -> None:
    """Warn if any stored chunks use a different embed model than current config."""
    if table.count_rows() == 0:
        return

    # Sample the first row to check embed_model
    sample = table.head(1).to_pydict()
    if sample.get("embed_model") and sample["embed_model"][0]:
        stored_model = sample["embed_model"][0]
        if stored_model != config.embed_model:
            logger.warning(
                "Embed model mismatch: stored chunks use '%s' but config specifies '%s'. "
                "Consider re-ingesting all documents for consistent embeddings.",
                stored_model,
                config.embed_model,
            )


def _build_fts_index(table) -> None:
    """Create or rebuild the full-text search index on the text column."""
    try:
        table.create_fts_index("text", replace=True)
    except Exception:
        logger.debug("Failed to create FTS index", exc_info=True)


def _process_single_file(
    file_path: Path,
    source_key: str,
    table,
    converter,
    chunker,
    verbose: bool = True,
    use_cache: bool = True,
) -> int:
    """
    Process a single file and add chunks to the table.

    If a markdown cache exists and use_cache is True, the expensive Docling
    parsing step is skipped and cached markdown is used for chunking instead.

    Returns:
        Number of chunks added
    """
    # Remove old chunks for this file (if re-processing)
    table.delete(f"source = '{source_key}'")

    try:
        cached_md = _read_markdown_cache(source_key) if use_cache else None

        if cached_md is not None:
            # Cache hit: skip Docling parsing, convert cached markdown via Docling
            if verbose:
                logger.info("  Using cached markdown")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(cached_md)
                tmp_path = Path(tmp.name)
            try:
                result = converter.convert(tmp_path)
            finally:
                tmp_path.unlink()
            doc = result.document
        else:
            # Cache miss: full Docling parse
            ext = file_path.suffix.lower()
            if ext in TEXT_EXTENSIONS:
                md_content = convert_text_to_markdown(file_path)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, encoding="utf-8"
                ) as tmp:
                    tmp.write(md_content)
                    tmp_path = Path(tmp.name)
                try:
                    result = converter.convert(tmp_path)
                finally:
                    tmp_path.unlink()
            else:
                result = converter.convert(file_path)
            doc = result.document

            # Cache the parsed markdown for future use
            md_export = result.document.export_to_markdown()
            _write_markdown_cache(source_key, md_export)

        # Chunk the document
        chunks = list(chunker.chunk(doc))

        if not chunks:
            if verbose:
                logger.warning("  No chunks extracted from %s", source_key)
            return 0

        # Extract document-level metadata
        config = get_config()
        now = datetime.now(UTC).isoformat()
        doc_title = ""
        if hasattr(doc, "title") and doc.title:
            doc_title = str(doc.title)

        # Apply contextual retrieval if enabled
        chunk_texts = [chunk.text for chunk in chunks]
        if config.enable_contextual_retrieval:
            from docling_rag.core.contextualize import contextualize_chunks

            # Use cached markdown or export from parsed doc
            doc_md = cached_md or result.document.export_to_markdown()
            chunk_texts = contextualize_chunks(chunk_texts, doc_md)
            if verbose:
                logger.info("  Applied contextual retrieval to %d chunks", len(chunks))

        # Prepare data for LanceDB (list of dicts, vector auto-generated)
        rows = []
        for i, chunk in enumerate(chunks):
            row = {
                "text": chunk_texts[i],
                "source": source_key,
                "chunk_index": i,
                "ingested_at": now,
                "embed_model": config.embed_model,
                "title": doc_title,
                "section_heading": "",
                "page": -1,
            }

            # Add page info and section heading if available
            if (
                hasattr(chunk, "meta")
                and chunk.meta
                and hasattr(chunk.meta, "doc_items")
                and chunk.meta.doc_items
            ):
                first_item = chunk.meta.doc_items[0]
                if hasattr(first_item, "prov") and first_item.prov:
                    prov = first_item.prov[0]
                    if hasattr(prov, "page_no"):
                        row["page"] = prov.page_no

            # Extract section heading from chunk headings
            if hasattr(chunk, "meta") and chunk.meta and hasattr(chunk.meta, "headings"):
                headings = chunk.meta.headings
                if headings:
                    row["section_heading"] = " > ".join(headings)

            rows.append(row)

        # Add to table (LanceDB generates embeddings automatically via schema config)
        table.add(rows)

        if verbose:
            logger.info("  Added %d chunks", len(chunks))

        return len(chunks)

    except Exception:
        logger.exception("  Error processing %s", source_key)
        return 0


def ingest_file(file_path: Path | str, verbose: bool = True) -> dict[str, int]:
    """
    Ingest a single file from any location into LanceDB.

    This is useful for adding files from outside the default data directory.
    The file's absolute path is used as the source key.

    Args:
        file_path: Path to the file to ingest
        verbose: Print progress information

    Returns:
        Statistics about the ingestion process
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return {"processed": 0, "chunks_added": 0}

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported file type: %s", ext)
        return {"processed": 0, "chunks_added": 0}

    # Use absolute path as source key for external files
    source_key = str(file_path)

    if verbose:
        logger.info("Processing: %s", source_key)

    # Initialize LanceDB table and get cached heavy objects
    table = get_table()
    converter = _get_converter()
    chunker = _get_chunker()

    # Process the file
    chunks_added = _process_single_file(file_path, source_key, table, converter, chunker, verbose)

    # Build FTS index after ingestion
    if chunks_added > 0:
        _build_fts_index(table)

    # Update hash file to track this external file
    hashes = load_hashes()
    hashes[source_key] = get_file_hash(file_path)
    save_hashes(hashes)

    stats = {"processed": 1, "chunks_added": chunks_added}

    if verbose:
        logger.info("Ingestion complete: 1 file processed, %d chunks added", chunks_added)

    return stats


def ingest_documents(data_dir: Path | str | None = None, verbose: bool = True) -> dict[str, int]:
    """
    Ingest documents from data directory into LanceDB.

    Args:
        data_dir: Directory containing documents to ingest (defaults to config.data_dir)
        verbose: Print progress information

    Returns:
        Statistics about the ingestion process
    """
    config = get_config()

    # Determine if using default data_dir or custom directory
    is_custom_dir = data_dir is not None
    data_dir = config.data_dir if data_dir is None else Path(data_dir).resolve()

    # Use absolute paths for custom directories to avoid collisions
    use_absolute_paths = is_custom_dir

    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    # Get documents to process
    new_or_changed, deleted, current_hashes = get_documents_to_process(
        data_dir, use_absolute_paths=use_absolute_paths
    )

    if not new_or_changed and not deleted:
        if verbose:
            logger.info("No changes detected. All documents are up to date.")
        return {"processed": 0, "deleted": 0, "chunks_added": 0}

    # Initialize LanceDB table
    table = get_table()

    # Warn if stored embed model differs from current config
    _check_embed_model_consistency(table, config)

    # Remove chunks and cache for deleted files
    for source_path in deleted:
        if verbose:
            logger.info("Removing chunks for deleted file: %s", source_path)
        table.delete(f"source = '{source_path}'")
        _delete_markdown_cache(source_path)

    converter = _get_converter()
    chunker = _get_chunker()
    chunks_added = 0

    for file_path in new_or_changed:
        if use_absolute_paths:
            source_key = str(file_path.resolve())
        else:
            source_key = str(file_path.relative_to(data_dir))

        if verbose:
            logger.info("Processing: %s", source_key)

        chunks_added += _process_single_file(
            file_path, source_key, table, converter, chunker, verbose
        )

        # Save hashes incrementally after each file to prevent data loss on crash
        save_hashes(current_hashes)

    # Build FTS index after all files are processed
    if chunks_added > 0 or deleted:
        _build_fts_index(table)

    stats = {
        "processed": len(new_or_changed),
        "deleted": len(deleted),
        "chunks_added": chunks_added,
    }

    if verbose:
        logger.info(
            "Ingestion complete: %d files processed, %d removed, %d chunks added",
            stats["processed"],
            stats["deleted"],
            stats["chunks_added"],
        )

    return stats
