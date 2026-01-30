"""Document ingestion pipeline using Docling and ChromaDB."""

import hashlib
import json
import tempfile
from pathlib import Path

import chromadb
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from transformers import logging as tf_logging

from docling_rag.embeddings import EMBED_MODEL_ID, MAX_TOKENS, get_embedding_function

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

# Paths
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
HASH_FILE = Path(".docling_hashes.json")


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
    if HASH_FILE.exists():
        return json.loads(HASH_FILE.read_text())
    return {}


def save_hashes(hashes: dict[str, str]) -> None:
    """Save file hashes to disk."""
    HASH_FILE.write_text(json.dumps(hashes, indent=2))


def get_documents_to_process(data_dir: Path) -> tuple[list[Path], list[str], dict[str, str]]:
    """
    Determine which documents need processing.

    Returns:
        - new_or_changed: Files that need to be (re)processed
        - deleted: File paths that were removed and need cleanup
        - current_hashes: Hash mapping for all current files
    """
    old_hashes = load_hashes()
    current_hashes: dict[str, str] = {}
    new_or_changed: list[Path] = []

    # Find all supported files
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in data_dir.rglob(f"*{ext}"):
            rel_path = str(file_path.relative_to(data_dir))
            file_hash = get_file_hash(file_path)
            current_hashes[rel_path] = file_hash

            # Check if new or changed
            if rel_path not in old_hashes or old_hashes[rel_path] != file_hash:
                new_or_changed.append(file_path)

    # Find deleted files
    deleted = [path for path in old_hashes if path not in current_hashes]

    return new_or_changed, deleted, current_hashes


def ingest_documents(data_dir: Path = DATA_DIR, verbose: bool = True) -> dict[str, int]:
    """
    Ingest documents from data directory into ChromaDB.

    Args:
        data_dir: Directory containing documents to ingest
        verbose: Print progress information

    Returns:
        Statistics about the ingestion process
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    # Get documents to process
    new_or_changed, deleted, current_hashes = get_documents_to_process(data_dir)

    if not new_or_changed and not deleted:
        if verbose:
            print("No changes detected. All documents are up to date.")
        return {"processed": 0, "deleted": 0, "chunks_added": 0}

    # Initialize ChromaDB
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
        embedding_function=get_embedding_function(),
    )

    # Remove chunks from deleted files
    for rel_path in deleted:
        if verbose:
            print(f"Removing chunks for deleted file: {rel_path}")
        # Delete all chunks with this source
        existing = collection.get(where={"source": rel_path})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    # Process new/changed files
    converter = DocumentConverter(
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
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
        max_tokens=MAX_TOKENS,
    )
    chunker = HybridChunker(tokenizer=tokenizer)
    chunks_added = 0

    for file_path in new_or_changed:
        rel_path = str(file_path.relative_to(data_dir))
        if verbose:
            print(f"Processing: {rel_path}")

        # Remove old chunks for this file (if re-processing)
        existing = collection.get(where={"source": rel_path})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        try:
            # Handle text/code files by converting to markdown first
            ext = file_path.suffix.lower()
            if ext in TEXT_EXTENSIONS:
                md_content = convert_text_to_markdown(file_path)
                # Create temp markdown file for Docling
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, encoding="utf-8"
                ) as tmp:
                    tmp.write(md_content)
                    tmp_path = Path(tmp.name)
                try:
                    result = converter.convert(tmp_path)
                finally:
                    tmp_path.unlink()  # Clean up temp file
            else:
                result = converter.convert(file_path)
            doc = result.document

            # Chunk the document
            chunks = list(chunker.chunk(doc))

            if not chunks:
                if verbose:
                    print(f"  Warning: No chunks extracted from {rel_path}")
                continue

            # Prepare data for ChromaDB
            ids = [f"{rel_path}_{i}" for i in range(len(chunks))]
            documents = [chunk.text for chunk in chunks]
            metadatas = []

            for i, chunk in enumerate(chunks):
                meta = {
                    "source": rel_path,
                    "chunk_index": i,
                }
                # Add page info if available
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
                            meta["page"] = prov.page_no
                metadatas.append(meta)

            # Add to collection (ChromaDB generates embeddings automatically)
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            chunks_added += len(chunks)

            if verbose:
                print(f"  Added {len(chunks)} chunks")

        except Exception as e:
            print(f"  Error processing {rel_path}: {e}")

    # Save updated hashes
    save_hashes(current_hashes)

    stats = {
        "processed": len(new_or_changed),
        "deleted": len(deleted),
        "chunks_added": chunks_added,
    }

    if verbose:
        print(
            f"\nIngestion complete: {stats['processed']} files processed, "
            f"{stats['deleted']} removed, {stats['chunks_added']} chunks added"
        )

    return stats
