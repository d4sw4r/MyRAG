# MyRAG

A personal RAG system (Retrieval-Augmented Generation): ingest a folder, index
its files, and then ask an AI questions about them.

> **Status:** v0.1.0. `ingest`, `ask`, `status` and `reset` work.

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
```

Tip: Without a GPU, the CPU build of PyTorch is enough and saves a large download:
`pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu`

On the first `ingest`, the embedding model (about 470 MB) is downloaded from
huggingface.co. After that, ingesting works offline.

## Commands

```bash
myrag ingest <folder>        # ingest a folder; run again = pick up changes only
myrag ask "<question>"       # ask a question, answer with sources
myrag ask "<question>" -k 3  # use only the 3 best matching passages
myrag status                 # folders, files, chunks, model, and whether an API key is set
myrag reset [--yes]          # delete the entire index
```

Several folders can live in the same index. Running `ingest` on one folder only
changes that folder's entries; the others are kept.

Without installing: `python -m myrag ...` (with `PYTHONPATH=src`).

## Ingesting (`myrag ingest`)

- Reads the folder recursively: `.txt`, `.md`, `.pdf`, `.docx`. Hidden folders
  (e.g. `.git`) are skipped.
- Splits the texts into chunks of about 1000 characters with 200 characters of overlap.
- Stores a SHA-256 hash per file: on the next run only new or changed files are
  processed again, and chunks of deleted files are removed.
- Embeddings are computed locally with sentence-transformers
  (`paraphrase-multilingual-MiniLM-L12-v2`, works well for German and other
  languages). The model is downloaded once on the first run.
- Data is stored in ChromaDB (collection `myrag`), each chunk with the metadata
  `source` (relative path), `root` (ingested folder), `chunk_index` and `file_hash`.
- A broken file does not abort the run; it is reported at the end.
- Files without text (e.g. scanned PDFs without OCR) are reported as
  "without text". Run OCR on such PDFs first.

## Asking questions (`myrag ask`)

- The question is embedded with the same embedding model used for ingesting,
  then `ask` fetches the best matching chunks from ChromaDB (5 by default,
  change with `-k`).
- The chunks are passed to the language model as numbered excerpts. It answers
  only from these excerpts and cites statements with `[1]`, `[2]` and so on.
- Below the answer is the list of sources with file path and section.
- The default is the Claude API (set `ANTHROPIC_API_KEY`). To run locally, use
  `MYRAG_LLM=ollama` (first e.g. `ollama pull llama3.1`).

```bash
export ANTHROPIC_API_KEY=...
myrag ask "How long do cats sleep?"
MYRAG_LLM=ollama MYRAG_MODEL=qwen2.5 myrag ask "How long do cats sleep?"
```

## Configuration (environment variables)

| Variable            | Default                   | Meaning                            |
|---------------------|---------------------------|------------------------------------|
| `MYRAG_DB_PATH`     | `./.myrag/chroma`         | Location of the index              |
| `MYRAG_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (ingest and ask) |
| `MYRAG_LLM`         | `claude`                  | `claude` or `ollama`               |
| `MYRAG_MODEL`       | `claude-opus-5-5` or `llama3.1` | Model name                   |
| `ANTHROPIC_API_KEY` | –                         | API key for Claude                 |
| `OLLAMA_HOST`       | `http://localhost:11434`  | Address of a local Ollama          |

## Layout

```
src/myrag/
  cli.py        # command line (argparse): ingest, ask, status, reset
  status.py     # overview of index and configuration
  config.py     # configuration from environment variables
  ingest.py     # folder -> index (only new/changed files)
  loaders.py    # find files, read text from txt/md/pdf/docx, file hash
  chunking.py   # split text into overlapping chunks
  embeddings.py # shared embedding model for ingest and ask
  store.py      # ChromaDB access (get_collection)
  ask.py        # question -> passages -> LLM answer with sources
  llm.py        # language models: Claude API and Ollama
tests/
```

## Common errors

| Message | Fix |
|---|---|
| `No Claude API key found` | `export ANTHROPIC_API_KEY=...` or `MYRAG_LLM=ollama` |
| `The index is empty` / `No index at ...` | run `myrag ingest <folder>` first; work in the same directory or set `MYRAG_DB_PATH` |
| `Could not load the embedding model` | the first run needs internet access to huggingface.co |
| `Ollama is not reachable at ...` | start `ollama serve` or check `OLLAMA_HOST` |

## Assumptions

- By default the index lives in `./.myrag/chroma` in the current directory.
  If you use `myrag` from different directories, it is best to set
  `export MYRAG_DB_PATH=~/.myrag/chroma`.

## Tests

```bash
pytest
```
