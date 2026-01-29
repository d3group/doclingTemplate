"""Command-line interface for Docling RAG."""

import argparse
import shutil
import sys
from pathlib import Path

from docling_rag.download import download_author_papers
from docling_rag.ingest import ingest_documents
from docling_rag.query import format_results, query


def init_project(target_dir: Path) -> None:
    """Initialize a new RAG project at the target directory."""
    target = Path(target_dir).resolve()

    if target.exists() and any(target.iterdir()):
        print(f"Error: {target} already exists and is not empty.")
        sys.exit(1)

    # Find the template (this package's root)
    template_dir = Path(__file__).parent.parent.parent

    # Create target directory
    target.mkdir(parents=True, exist_ok=True)

    # Copy essential files
    files_to_copy = [
        "pyproject.toml",
        "README.md",
        "uv.lock",
        ".ruff.toml",
        ".gitignore",
        ".python-version",
    ]

    for filename in files_to_copy:
        src = template_dir / filename
        if src.exists():
            shutil.copy(src, target / filename)

    # Copy src directory
    shutil.copytree(template_dir / "src", target / "src")

    # Copy tests directory
    if (template_dir / "tests").exists():
        shutil.copytree(
            template_dir / "tests",
            target / "tests",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    else:
        (target / "tests").mkdir(exist_ok=True)
        (target / "tests" / "__init__.py").touch()

    # Create data directory (users can organize however they want)
    (target / "data").mkdir(parents=True)

    print(f"Created new RAG project at: {target}")
    print()
    print("Next steps:")
    print(f"  cd {target}")
    print("  uv sync")
    print("  # Add documents to data/")
    print("  uv run -m docling_rag ingest")
    print("  uv run -m docling_rag query 'your question'")


def show_stats() -> None:
    """Show statistics about the current knowledge base."""
    import chromadb

    chroma_dir = Path("chroma_db")

    if not chroma_dir.exists():
        print("No knowledge base found. Run 'ingest' first.")
        return

    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        collection = client.get_collection("documents")
        count = collection.count()

        # Get unique sources
        all_docs = collection.get(include=["metadatas"])
        sources = set()
        for meta in all_docs["metadatas"]:
            if "source" in meta:
                sources.add(meta["source"])

        print("Knowledge Base Statistics")
        print("─────────────────────────")
        print(f"Total chunks:    {count}")
        print(f"Total documents: {len(sources)}")
        print()
        if sources:
            print("Documents:")
            for src in sorted(sources):
                # Count chunks per source
                src_chunks = collection.get(where={"source": src})
                print(f"  • {src} ({len(src_chunks['ids'])} chunks)")

    except ValueError:
        print("No documents ingested yet.")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="docling-rag",
        description="Local RAG system using Docling and ChromaDB",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Init command
    init_parser = subparsers.add_parser("init", help="Create a new RAG project")
    init_parser.add_argument("path", help="Path for the new project")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from data/ directory")
    ingest_parser.add_argument(
        "--data-dir", default="data", help="Directory containing documents (default: data)"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.add_argument(
        "-n", "--num-results", type=int, default=5, help="Number of results (default: 5)"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show knowledge base statistics")

    # Download command
    download_parser = subparsers.add_parser(
        "download", help="Download papers from Google Scholar author page"
    )
    download_parser.add_argument("url", help="Google Scholar author profile URL")
    download_parser.add_argument(
        "--max", type=int, default=None, help="Maximum number of papers to download"
    )
    download_parser.add_argument(
        "--output", default="data", help="Output directory (default: data)"
    )
    download_parser.add_argument(
        "--no-scihub",
        action="store_true",
        help="Disable Sci-Hub fallback (only download open access papers)",
    )

    args = parser.parse_args()

    if args.command == "init":
        init_project(args.path)
    elif args.command == "ingest":
        ingest_documents(data_dir=args.data_dir)
    elif args.command == "query":
        results = query(args.question, n_results=args.num_results)
        print(format_results(results))
    elif args.command == "stats":
        show_stats()
    elif args.command == "download":
        download_author_papers(
            scholar_url=args.url,
            output_dir=args.output,
            max_papers=args.max,
            use_scihub=not args.no_scihub,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
