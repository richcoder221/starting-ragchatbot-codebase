# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Manager

Always use `uv` — never `pip` or bare `python` commands.

```bash
uv sync                          # install dependencies
uv add <package>                 # add a new package
uv run python script.py          # run a script
uv run uvicorn app:app --reload  # run the server
```

## Setup

```bash
# Install dependencies
uv sync

# Configure API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...
```

## Running

```bash
# Start the server (from repo root)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

Web UI: `http://localhost:8000` — API docs: `http://localhost:8000/docs`

Windows requires Git Bash to run the shell script.

## Architecture

**Full-stack RAG app.** FastAPI backend serves the vanilla JS frontend as static files from the same origin (`/`), so the frontend uses relative `/api` paths with no CORS friction in production.

### Request lifecycle

1. `frontend/script.js` POSTs `{ query, session_id }` to `POST /api/query`
2. `backend/app.py` delegates to `RAGSystem.query()`
3. `RAGSystem` (Facade) pulls conversation history from `SessionManager`, then calls `AIGenerator.generate_response()` with tool definitions
4. **Claude API call #1** — Claude decides whether to call the `search_course_content` tool
5. If tool use: `ToolManager` dispatches to `CourseSearchTool` → `VectorStore.search()` → ChromaDB query using Sentence-Transformers embeddings → results returned as formatted string to Claude
6. **Claude API call #2** — Claude synthesizes the tool result into a final answer (skipped when Claude answers from general knowledge, making it a single-call path)
7. Sources are pulled from `CourseSearchTool.last_sources`, session history is updated, JSON response returned

### Startup indexing

On boot, `@app.on_event("startup")` calls `RAGSystem.add_course_folder("../docs/")`. Each `.txt` file is parsed by `DocumentProcessor` into a `Course` + `[CourseChunk]` and stored in two ChromaDB collections:
- `course_catalog` — course-level metadata, used for fuzzy course name resolution
- `course_content` — chunked lesson text (800 chars, 100 char overlap), used for semantic search

Already-indexed courses are skipped (deduped by title).

### Key files

| File | Role |
|------|------|
| `backend/config.py` | All tunable constants (model, chunk size, max results, history length) |
| `backend/rag_system.py` | Facade — the only entry point the API layer uses |
| `backend/ai_generator.py` | Claude API calls; two-step tool-use loop in `_handle_tool_execution()` |
| `backend/vector_store.py` | ChromaDB wrapper; `search()` does a two-step resolve-then-query when `course_name` is provided |
| `backend/search_tools.py` | `Tool` ABC + `CourseSearchTool` (Strategy); `ToolManager` registry routes Claude tool calls to Python |
| `backend/document_processor.py` | Parses course `.txt` format; `chunk_text()` does sentence-aware chunking |
| `backend/session_manager.py` | In-memory session store; history is injected into the Claude **system prompt**, not the messages array |
| `backend/models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |

### Course document format

Files in `docs/` must follow this structure for the parser to extract metadata correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```

### Adding a new search tool

Implement the `Tool` ABC in `backend/search_tools.py` (define `get_tool_definition()` returning an Anthropic tool schema and `execute(**kwargs)`), then register it with `tool_manager.register_tool(your_tool)` in `RAGSystem.__init__()`. Claude will automatically be able to call it.