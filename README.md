# Docling RAG

Local RAG (Retrieval-Augmented Generation) system using [Docling](https://github.com/DS4SD/docling) for document parsing, [LanceDB](https://lancedb.com/) for vector storage, and [Snowflake Arctic Embed](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) for embeddings.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/d3group/doclingTemplate.git
cd doclingTemplate
uv sync
```

### Configuration

Defaults work out of the box. Copy `rag.toml.example` to `rag.toml` to customize:

```toml
data_dir = "data"
lancedb_dir = "lancedb_data"
embed_model = "Snowflake/snowflake-arctic-embed-m-v2.0"
max_tokens = 1500
enable_reranking = true
enable_hybrid_search = true
```

### Contextual Retrieval (Optional)

Contextual retrieval uses a local LLM to enrich each chunk with document context before embedding. This significantly improves search quality ([Anthropic reports](https://www.anthropic.com/news/contextual-retrieval) 35-67% fewer retrieval failures).

Requires [Ollama](https://ollama.com/) running locally:

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:1.5b
```

Then enable in `rag.toml`:

```toml
enable_contextual_retrieval = true
ollama_model = "qwen2.5:1.5b"
```

Re-ingest your documents after enabling. If Ollama isn't running, ingestion falls back to raw chunks automatically.

### First Run

```bash
# Add your documents
cp your-documents/* data/

# Ingest them into the knowledge base
uv run docling-rag ingest

# Query
uv run docling-rag query "What is this document about?"

# Check what's indexed
uv run docling-rag stats
```

## MCP Server

The MCP server lets Claude Code, Claude Desktop, or any MCP-compatible tool query your knowledge base directly.

### Claude Code (same project)

The included `.mcp.json` auto-configures the server. Run `/mcp` in Claude Code or restart the session.

### Claude Code (different project)

Create a `.mcp.json` in the other project pointing back to this one:

```json
{
  "mcpServers": {
    "docling-rag": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/doclingTemplate", "run", "docling-rag-mcp"]
    }
  }
}
```

### Remote Server (via SSH)

If the MCP server is running on a remote machine, you can connect to it over SSH. The MCP protocol uses stdio, so SSH transparently tunnels the connection.

**1. Set up SSH key authentication** (so no password prompt blocks the connection):

```bash
ssh-keygen -t ed25519          # skip if you already have a key
ssh-copy-id user@your-server
```

**2. Create `.mcp.json`** in your local project:

```json
{
  "mcpServers": {
    "docling-rag": {
      "command": "ssh",
      "args": [
        "user@your-server",
        "cd /path/to/doclingTemplate && uv run docling-rag-mcp"
      ]
    }
  }
}
```

Replace `user@your-server` and `/path/to/doclingTemplate` with your values. If `uv` isn't in the remote PATH, use its full path (e.g. `~/.local/bin/uv`).

**3. Verify** the SSH command works before starting Claude Code:

```bash
ssh user@your-server "cd /path/to/doclingTemplate && uv run docling-rag-mcp"
```

If it hangs waiting for input, the server is running correctly (it's waiting for JSON-RPC commands over stdin).

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "docling-rag": {
      "command": "/opt/homebrew/bin/uv",
      "args": ["--directory", "/absolute/path/to/doclingTemplate", "run", "docling-rag-mcp"]
    }
  }
}
```

Use the **full path to `uv`** (find with `which uv`) — GUI apps don't share your terminal's PATH.

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `query_knowledge` | Search the knowledge base |
| `ingest_documents_tool` | Ingest documents from a directory |
| `ingest_file_tool` | Ingest a single file from any path |
| `get_database_stats` | Get database statistics |
| `list_indexed_sources` | List all indexed documents |
| `delete_document` | Remove a document from the index |

## CLI Reference

```bash
uv run docling-rag ingest                      # Ingest data/ directory
uv run docling-rag ingest --file /path/to/doc  # Ingest a single file from anywhere
uv run docling-rag query "your question"       # Search the knowledge base
uv run docling-rag query "..." -n 10           # Return more results (default: 5)
uv run docling-rag stats                       # Show statistics
uv run docling-rag init <path>                 # Create new project from template
```

## Creating a New Project

Create separate knowledge bases for different topics:

```bash
uv run docling-rag init ~/projects/my-topic
cd ~/projects/my-topic
uv sync
cp ~/Documents/relevant-files/* data/
uv run docling-rag ingest
```

Each project gets its own vector database, config, and document store.

## How It Works

1. **Ingest** — Documents are parsed with Docling, chunked semantically, and stored as embeddings in LanceDB
2. **Search** — Queries use hybrid search (vector similarity + BM25 keyword matching) with cross-encoder reranking
3. **Results** — Returns the most relevant chunks with source file and page references

## Supported File Types

PDF, DOCX, PPTX, XLSX, HTML/HTM, images (PNG, JPG, JPEG — OCR), Markdown, LaTeX, plain text, and code files (`.py`, `.js`, `.ts`, `.json`, `.yaml`, `.yml`, `.toml`, `.sh`, `.css`).
