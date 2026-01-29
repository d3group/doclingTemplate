# Docling RAG Template

Local RAG (Retrieval-Augmented Generation) system that parses documents and enables semantic search. **100% free** - all models run locally.

## Features

- **Document parsing**: PDF, DOCX, PPTX, XLSX, HTML, images (via [Docling](https://github.com/DS4SD/docling))
- **Vector storage**: Local ChromaDB database
- **Embeddings**: sentence-transformers (runs locally, no API costs)
- **Change detection**: Only re-processes new/modified files
- **Scholar download**: Bulk download papers from Google Scholar author pages

## Quick Start

```bash
# Install dependencies
uv sync

# Add documents to data/ (any structure)
cp your-document.pdf data/

# Ingest documents
uv run -m docling_rag ingest

# Query
uv run -m docling_rag query "What is this document about?"

# View statistics
uv run -m docling_rag stats
```

## Creating a New Project

Use this as a template for topic-specific knowledge bases:

```bash
# From this directory, create a new project
uv run -m docling_rag init ~/projects/rag-my-topic

# Set up the new project
cd ~/projects/rag-my-topic
uv sync

# Add your documents and start using it
cp ~/Documents/relevant-files/* data/
uv run -m docling_rag ingest
```

## Commands

| Command | Description |
|---------|-------------|
| `uv run -m docling_rag ingest` | Process documents in `data/` |
| `uv run -m docling_rag query "..."` | Search the knowledge base |
| `uv run -m docling_rag query "..." -n 10` | Return more results (default: 5) |
| `uv run -m docling_rag stats` | Show database statistics |
| `uv run -m docling_rag init <path>` | Create a new project from template |
| `uv run -m docling_rag download <url>` | Download papers from Google Scholar |
| `uv run -m docling_rag download <url> --max 10` | Limit number of papers |

## Project Structure

```
project/
├── data/                  # Your documents (any structure)
├── chroma_db/             # Vector database (auto-created, gitignored)
├── .docling_hashes.json   # Change tracking (auto-created, gitignored)
└── src/docling_rag/       # Source code
```

## Supported File Types

- PDF (`.pdf`)
- Word (`.docx`)
- PowerPoint (`.pptx`)
- Excel (`.xlsx`)
- HTML (`.html`, `.htm`)
- Images (`.png`, `.jpg`, `.jpeg`)

## How It Works

1. **Ingest**: Documents are parsed with Docling, chunked semantically, and stored as embeddings in ChromaDB
2. **Query**: Your question is embedded and compared against stored chunks using cosine similarity
3. **Results**: Returns the most relevant chunks with source file and page references

## Downloading Papers from Google Scholar

Bulk download an author's papers directly into your `data/` folder:

```bash
# Download all available papers from an author
uv run -m docling_rag download "https://scholar.google.com/citations?user=AUTHOR_ID"

# Limit to first 20 papers
uv run -m docling_rag download "https://scholar.google.com/citations?user=AUTHOR_ID" --max 20

# Only open-access (no Sci-Hub)
uv run -m docling_rag download "https://scholar.google.com/citations?user=AUTHOR_ID" --no-scihub

# Then ingest them
uv run -m docling_rag ingest
```

Downloads try open-access sources first (arXiv, preprints), then falls back to Sci-Hub for paywalled papers.

### Setting Up Tor (Recommended)

Google Scholar blocks automated requests. Install and start Tor to bypass this:

```bash
# Install Tor (macOS)
brew install tor

# Start Tor
brew services start tor

# Now downloads will automatically use Tor
uv run -m docling_rag download "https://scholar.google.com/citations?user=..."
```

The download command auto-detects Tor on port 9050 and uses it with browser TLS fingerprint impersonation (via curl_cffi) to avoid detection.

If you get blocked:
- Make sure Tor is running: `brew services start tor`
- Wait a few minutes and retry
- Use a VPN as alternative

## Usage with Claude Code

Just ask questions about your documents:

```
You: "What does the paper say about performance bounds?"
Claude: [Runs query, synthesizes answer with citations]
```

## Tips

- **Separate projects by topic** - Create different RAG projects for different domains to keep results focused
- **Re-run ingest after changes** - The system detects modified files automatically
- **Use specific queries** - "What are the three main contributions?" works better than "summarize"
