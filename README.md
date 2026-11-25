# MyRAG

Eigenes RAG-System (Retrieval-Augmented Generation): Ein Ordner wird eingelesen,
die Dateien werden indexiert und können anschließend mit einer KI befragt werden.

> **Status:** Grundgerüst (v0.1.0). Die Befehle existieren, `ingest` und `ask`
> sind noch Platzhalter und werden in den nächsten Schritten umgesetzt.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Benutzung

```bash
myrag ingest ~/Dokumente/notizen      # Ordner einlesen und indexieren
myrag ask "Was steht in meinen Notizen über Docker?"
myrag ask "..." -k 3                   # nur die 3 besten Fundstellen nutzen
```

Alternativ ohne Installation: `python -m myrag ...` (mit `PYTHONPATH=src`).

## Konfiguration (Umgebungsvariablen)

| Variable            | Standard                  | Bedeutung                          |
|---------------------|---------------------------|------------------------------------|
| `MYRAG_DB_PATH`     | `~/.myrag/db`             | Speicherort des Index              |
| `MYRAG_LLM`         | `claude`                  | `claude` oder `ollama`             |
| `MYRAG_MODEL`       | (je nach LLM)             | Modellname                         |
| `ANTHROPIC_API_KEY` | –                         | API-Schlüssel für Claude           |
| `OLLAMA_HOST`       | `http://localhost:11434`  | Adresse eines lokalen Ollama       |

## Aufbau

```
src/myrag/
  cli.py      # Kommandozeile (argparse): ingest, ask
  config.py   # Konfiguration aus Umgebungsvariablen
  ingest.py   # Ordner einlesen -> Index   (Platzhalter)
  ask.py      # Frage -> Fundstellen -> LLM-Antwort (Platzhalter)
tests/
```

## Geplant / Annahmen

- **Ingest:** Text- und Markdown-Dateien (später PDF u. a.), Aufteilung in Chunks,
  Hash-Cache, damit unveränderte Dateien nicht neu eingelesen werden.
- **Suche:** lokale Embeddings (sentence-transformers) mit ChromaDB.
- **Antworten:** Claude-API oder lokal per Ollama, umschaltbar über `MYRAG_LLM`.
- Der Index liegt pro Benutzer unter `~/.myrag/`, nicht im eingelesenen Ordner.

## Tests

```bash
pytest
```
