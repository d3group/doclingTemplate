"""Command-line interface for Docling RAG."""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from docling_rag.core import format_results, get_stats, ingest_documents, ingest_file, query

# Configure logging for CLI usage (show INFO+ from our package)
logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger("docling_rag").setLevel(logging.INFO)
# Silence noisy third-party loggers
for _name in ("lancedb", "httpx", "sentence_transformers", "transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)


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
        ".mcp.json",
        "rag.toml.example",
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
    print("  uv run docling-rag ingest")
    print("  uv run docling-rag query 'your question'")


def show_stats() -> None:
    """Show statistics about the current knowledge base."""
    stats = get_stats()

    if stats["total_chunks"] == 0:
        print("No knowledge base found. Run 'ingest' first.")
        return

    print("Knowledge Base Statistics")
    print("-------------------------")
    print(f"Total chunks:    {stats['total_chunks']}")
    print(f"Total documents: {stats['total_documents']}")
    print()
    if stats["sources"]:
        print("Documents:")
        for src in stats["sources"]:
            print(f"  - {src['name']} ({src['chunks']} chunks)")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="docling-rag",
        description="Local RAG system using Docling and LanceDB",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Init command
    init_parser = subparsers.add_parser("init", help="Create a new RAG project")
    init_parser.add_argument("path", help="Path for the new project")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from data/ directory")
    ingest_parser.add_argument(
        "--data-dir", default=None, help="Directory containing documents (default: data)"
    )
    ingest_parser.add_argument("--file", "-f", help="Ingest a single file from any path")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.add_argument(
        "-n", "--num-results", type=int, default=5, help="Number of results (default: 5)"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show knowledge base statistics")

    args = parser.parse_args()

    if args.command == "init":
        init_project(args.path)
    elif args.command == "ingest":
        if args.file:
            ingest_file(file_path=args.file)
        else:
            ingest_documents(data_dir=args.data_dir)
    elif args.command == "query":
        results = query(args.question, n_results=args.num_results)
        print(format_results(results))
    elif args.command == "stats":
        show_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
