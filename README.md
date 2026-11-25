# MyRAG

Eigenes RAG-System (Retrieval-Augmented Generation): Ein Ordner wird eingelesen,
die Dateien werden indexiert und können anschließend mit einer KI befragt werden.

> **Status:** v0.1.0. `ingest` funktioniert, `ask` ist noch ein Platzhalter.

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

## Einlesen (`myrag ingest`)

- Liest den Ordner rekursiv ein: `.txt`, `.md`, `.pdf`, `.docx`. Versteckte Ordner
  (z. B. `.git`) werden übersprungen.
- Teilt die Texte in Chunks von ca. 1000 Zeichen mit 200 Zeichen Überlappung.
- Merkt sich pro Datei einen SHA-256-Hash: Beim nächsten Lauf werden nur neue oder
  geänderte Dateien neu verarbeitet, Chunks gelöschter Dateien werden entfernt.
- Embeddings lokal mit sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`,
  gut für Deutsch). Beim ersten Lauf wird das Modell einmalig heruntergeladen.
- Gespeichert wird in ChromaDB (Collection `myrag`), pro Chunk mit den Metadaten
  `source` (relativer Pfad), `chunk_index` und `file_hash`.
- Eine fehlerhafte Datei bricht den Lauf nicht ab, sie wird am Ende gemeldet.

## Konfiguration (Umgebungsvariablen)

| Variable            | Standard                  | Bedeutung                          |
|---------------------|---------------------------|------------------------------------|
| `MYRAG_DB_PATH`     | `./.myrag/chroma`         | Speicherort des Index              |
| `MYRAG_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding-Modell (ingest und ask) |
| `MYRAG_LLM`         | `claude`                  | `claude` oder `ollama`             |
| `MYRAG_MODEL`       | (je nach LLM)             | Modellname                         |
| `ANTHROPIC_API_KEY` | –                         | API-Schlüssel für Claude           |
| `OLLAMA_HOST`       | `http://localhost:11434`  | Adresse eines lokalen Ollama       |

## Aufbau

```
src/myrag/
  cli.py      # Kommandozeile (argparse): ingest, ask
  config.py   # Konfiguration aus Umgebungsvariablen
  ingest.py   # Ordner einlesen -> Index (nur neue/geänderte Dateien)
  loaders.py  # Dateien finden, Text aus txt/md/pdf/docx lesen, Datei-Hash
  chunking.py # Text in überlappende Chunks teilen
  embeddings.py # gemeinsames Embedding-Modell für ingest und ask
  store.py    # ChromaDB-Zugriff (get_collection)
  ask.py      # Frage -> Fundstellen -> LLM-Antwort (Platzhalter)
tests/
```

## Geplant / Annahmen

- **Antworten:** Claude-API oder lokal per Ollama, umschaltbar über `MYRAG_LLM`.
- Der Index liegt standardmäßig unter `./.myrag/chroma` im aktuellen Verzeichnis.

## Tests

```bash
pytest
```
