# MyRAG

A personal RAG system (Retrieval-Augmented Generation): ingest a folder, index
its files, and then ask an AI questions about them.

> **Status:** v0.2.0. `ingest`, `ask`, `search`, `chat`, `status` and `reset` work.

> **Upgrading from v0.1:** the default embedding model changed (see below).
> Re-embed your index once with `myrag ingest --rebuild <folder>`; all folders
> in the index are re-ingested.

## Quick start

Requirement: Python 3.10 or newer.

```bash
git clone https://github.com/d4sw4r/MyRAG.git
cd MyRAG
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # takes a while the first time (PyTorch is large)

export ANTHROPIC_API_KEY=sk-ant-...  # get a key: https://platform.claude.com/settings/keys

myrag ingest ~/Documents/notes       # 1. ingest a folder
myrag status                         # 2. check what is in the index
myrag ask "What do my notes say about Docker?"   # 3. ask
myrag chat                           # 4. or have a conversation with follow-up questions
```

Tip: Without a GPU, the CPU build of PyTorch is enough and saves a large download:
`pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu`

On the first run, the embedding model (about 470 MB) and the reranker (about
470 MB) are downloaded from huggingface.co. After that, everything except the
Claude API works offline.

## Commands

```bash
myrag ingest <folder>        # ingest a folder; run again = pick up changes only
myrag ask "<question>"       # ask a question, answer with sources
myrag ask "<question>" -k 3  # use only the 3 best matching passages
myrag search "<text>"        # show the best matching passages with scores, no AI call
myrag chat                   # interactive chat with history and streamed answers
myrag ingest --rebuild <folder>  # delete the index and re-ingest all its folders
myrag status                 # folders, files, chunks, model, and whether an API key is set
myrag reset [--yes]          # delete the entire index
```

Several folders can live in the same index. Running `ingest` on one folder only
changes that folder's entries; the others are kept.

Without installing: `python -m myrag ...` (with `PYTHONPATH=src`).

## Ingesting (`myrag ingest`)

- Reads the folder recursively. Hidden folders (e.g. `.git`) are skipped. Supported:
  - text and Markdown: `.txt`, `.md`, `.rst`, `.tex`, `.log`, ...
  - documents: `.pdf`, `.docx`, `.pptx` (incl. speaker notes), `.xlsx`, `.csv`, `.epub`
  - web and mail: `.html`, `.htm`, `.eml` (subject, sender, body, attachment names)
  - source code and config: `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.c`, `.sql`,
    `.sh`, `.json`, `.yaml`, `.toml`, ...
  - Table rows (CSV, Excel) are stored as `column: value; ...`, so every chunk keeps
    the column names.
- Splits the texts into chunks of about 1000 characters with 200 characters of
  overlap (change with `MYRAG_CHUNK_SIZE` / `MYRAG_CHUNK_OVERLAP`).
- Stores a SHA-256 hash and the chunk settings per file: on the next run only new or
  changed files (or all files, if the chunk settings changed) are processed again,
  and chunks of deleted files are removed.
- Embeddings are computed locally with sentence-transformers
  (`intfloat/multilingual-e5-small`, works well for German and other languages).
  It reads up to 512 tokens, so each chunk is embedded completely. (The v0.1
  model only read 128 tokens and ignored about half of every chunk.)
- The index remembers which embedding model built it (`myrag_index.json` next to
  the database). If the model changes, `ingest`, `ask`, `search` and `chat` stop
  with a hint instead of silently returning wrong results.
- Data is stored in ChromaDB (collection `myrag`), each chunk with the metadata
  `source` (relative path), `root` (ingested folder), `chunk_index`, `file_hash`
  and `chunking`.
- A broken file does not abort the run; it is reported at the end.
- Files without text (e.g. scanned PDFs without OCR) are reported as
  "without text". Run OCR on such PDFs first.

## Asking questions (`myrag ask`)

- Retrieval in four steps:
  1. **Vector search:** the question is embedded with the same model used for
     ingesting and compared with all chunks (finds passages with the same meaning).
  2. **Keyword search (BM25):** finds exact terms such as names, invoice numbers or
     error codes that the vector search often misses.
  3. Both lists are merged (reciprocal rank fusion) into about 30 candidates.
  4. **Reranking:** a cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
     multilingual) reads question and passage together and picks the best 5
     (change with `-k`).
- Passages below the relevance threshold (`MYRAG_MIN_SCORE`) are dropped. If
  nothing relevant is left, MyRAG says so and does not call the language model.
- The chunks are passed to the language model as numbered excerpts. It answers
  only from these excerpts and cites statements with `[1]`, `[2]` and so on.
- The answer is streamed while it is generated. Below it is the list of sources
  with file path and section.
- The default is the Claude API (set `ANTHROPIC_API_KEY`). To run locally, use
  `MYRAG_LLM=ollama` (first e.g. `ollama pull llama3.1`).

```bash
export ANTHROPIC_API_KEY=...
myrag ask "How long do cats sleep?"
MYRAG_LLM=ollama MYRAG_MODEL=qwen2.5 myrag ask "How long do cats sleep?"
```

## Searching without AI (`myrag search`)

Shows the best matching passages with file path and scores (relevance from the
reranker, similarity from the vector search, keyword score from BM25). Fast and
free; useful to check what `ask` would get to see.

## Chat (`myrag chat`)

An interactive session: ask follow-up questions like "and when was that?".
Each follow-up is first rewritten into a standalone search query using the
conversation, then answered with the last 6 turns as context. Commands:
`/sources` (sources of the last answer), `/clear` (forget the conversation),
`/help`, `/exit` (or Ctrl+D).

## Configuration (environment variables)

| Variable            | Default                   | Meaning                            |
|---------------------|---------------------------|------------------------------------|
| `MYRAG_DB_PATH`     | `./.myrag/chroma`         | Location of the index              |
| `MYRAG_EMBED_MODEL` | `intfloat/multilingual-e5-small` | Embedding model (ingest and ask); `intfloat/multilingual-e5-base` is better but slower |
| `MYRAG_CHUNK_SIZE` / `MYRAG_CHUNK_OVERLAP` | `1000` / `200` | Chunk length and overlap in characters |
| `MYRAG_HYBRID`      | `1`                       | `0` turns the keyword search (BM25) off |
| `MYRAG_RERANK`      | `1`                       | `0` turns reranking off (faster, less accurate) |
| `MYRAG_RERANK_MODEL`| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranker, e.g. `BAAI/bge-reranker-v2-m3` (better, larger) |
| `MYRAG_CANDIDATES`  | `30`                      | Candidates passed to the reranker |
| `MYRAG_MIN_SCORE`   | `0.1`                     | Minimum reranker score (0 to 1); lower it if good passages are dropped |
| `MYRAG_MIN_SIMILARITY` | `0` (off)              | Minimum cosine similarity when reranking is off |
| `MYRAG_LLM`         | `claude`                  | `claude` or `ollama`               |
| `MYRAG_MODEL`       | `claude-opus-5-5` or `llama3.1` | Model name                   |
| `ANTHROPIC_API_KEY` | –                         | API key for Claude                 |
| `OLLAMA_HOST`       | `http://localhost:11434`  | Address of a local Ollama          |

## Layout

```
src/myrag/
  cli.py        # command line (argparse): ingest, ask, search, chat, status, reset
  status.py     # overview of index and configuration
  config.py     # configuration from environment variables
  ingest.py     # folder -> index (only new/changed files)
  loaders.py    # find files, read text from all supported formats, file hash
  chunking.py   # split text into overlapping chunks
  embeddings.py # shared embedding model for ingest and ask
  store.py      # ChromaDB access, index info (embedding model)
  retrieval.py  # hybrid search (vector + BM25), reranking, relevance threshold
  ask.py        # question -> passages -> streamed LLM answer with sources
  search.py     # passages with scores, no LLM
  chat.py       # interactive chat with history and follow-up rewriting
  llm.py        # language models: Claude API and Ollama (incl. streaming)
tests/
```

## Common errors

| Message | Fix |
|---|---|
| `No Claude API key found` | `export ANTHROPIC_API_KEY=...` or `MYRAG_LLM=ollama` |
| `The index is empty` / `No index at ...` | run `myrag ingest <folder>` first; work in the same directory or set `MYRAG_DB_PATH` |
| `Could not load the embedding model` | the first run needs internet access to huggingface.co |
| `Ollama is not reachable at ...` | start `ollama serve` or check `OLLAMA_HOST` |
| `The index was built with the embedding model ...` | `myrag ingest --rebuild <folder>` or set `MYRAG_EMBED_MODEL` back |
| `Nothing relevant was found ...` | check with `myrag search`; lower `MYRAG_MIN_SCORE` if good passages are dropped |
| `Warning: reranker ... could not be loaded` | needs internet on first use; or `MYRAG_RERANK=0` |

## Assumptions

- By default the index lives in `./.myrag/chroma` in the current directory.
  If you use `myrag` from different directories, it is best to set
  `export MYRAG_DB_PATH=~/.myrag/chroma`.

## Tests

```bash
pytest
```
