# MyRAG

Eigenes RAG-System (Retrieval-Augmented Generation): Ein Ordner wird eingelesen,
die Dateien werden indexiert und können anschließend mit einer KI befragt werden.

> **Status:** v0.1.0. `ingest`, `ask`, `status` und `reset` funktionieren.

## Schnellstart

Voraussetzung: Python 3.10 oder neuer.

```bash
git clone https://github.com/d4sw4r/MyRAG.git
cd MyRAG
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # dauert beim ersten Mal (PyTorch ist groß)

export ANTHROPIC_API_KEY=sk-ant-...  # Schlüssel: https://platform.claude.com/settings/keys

myrag ingest ~/Dokumente/notizen     # 1. Ordner einlesen
myrag status                         # 2. prüfen, was im Index liegt
myrag ask "Was steht in meinen Notizen über Docker?"   # 3. fragen
```

Tipp: Ohne Grafikkarte reicht die CPU-Version von PyTorch und spart viel Download:
`pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu`

Beim ersten `ingest` wird das Embedding-Modell (ca. 470 MB) von huggingface.co
geladen, danach läuft das Einlesen offline.

## Befehle

```bash
myrag ingest <ordner>      # Ordner einlesen; erneut ausführen = nur Änderungen übernehmen
myrag ask "<frage>"        # Frage stellen, Antwort mit Quellen
myrag ask "<frage>" -k 3   # nur die 3 besten Fundstellen nutzen
myrag status               # Ordner, Dateien, Chunks, Modell und ob ein API-Schlüssel da ist
myrag reset [--yes]        # kompletten Index löschen
```

Mehrere Ordner können im selben Index liegen. `ingest` auf einen Ordner ändert
nur dessen Einträge, die anderen bleiben erhalten.

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
  `source` (relativer Pfad), `root` (eingelesener Ordner), `chunk_index` und `file_hash`.
- Eine fehlerhafte Datei bricht den Lauf nicht ab, sie wird am Ende gemeldet.
- Dateien ohne Text (z. B. eingescannte PDFs ohne Texterkennung) werden als
  „ohne Text“ gemeldet. Für solche PDFs vorher eine OCR laufen lassen.

## Fragen stellen (`myrag ask`)

- Die Frage wird mit demselben Embedding-Modell eingebettet wie beim Einlesen,
  dann holt `ask` die passendsten Chunks (Standard 5, änderbar mit `-k`) aus ChromaDB.
- Die Chunks gehen nummeriert als Auszüge an das Sprachmodell. Es antwortet nur
  anhand dieser Auszüge und belegt Aussagen mit `[1]`, `[2]` usw.
- Unter der Antwort steht die Quellenliste mit Dateipfad und Abschnitt.
- Standard ist die Claude-API (`ANTHROPIC_API_KEY` setzen). Lokal geht es mit
  `MYRAG_LLM=ollama` (vorher z. B. `ollama pull llama3.1`).

```bash
export ANTHROPIC_API_KEY=...
myrag ask "Wie lange schlafen Katzen?"
MYRAG_LLM=ollama MYRAG_MODEL=qwen2.5 myrag ask "Wie lange schlafen Katzen?"
```

## Konfiguration (Umgebungsvariablen)

| Variable            | Standard                  | Bedeutung                          |
|---------------------|---------------------------|------------------------------------|
| `MYRAG_DB_PATH`     | `./.myrag/chroma`         | Speicherort des Index              |
| `MYRAG_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding-Modell (ingest und ask) |
| `MYRAG_LLM`         | `claude`                  | `claude` oder `ollama`             |
| `MYRAG_MODEL`       | `claude-opus-5-5` bzw. `llama3.1` | Modellname                 |
| `ANTHROPIC_API_KEY` | –                         | API-Schlüssel für Claude           |
| `OLLAMA_HOST`       | `http://localhost:11434`  | Adresse eines lokalen Ollama       |

## Aufbau

```
src/myrag/
  cli.py      # Kommandozeile (argparse): ingest, ask, status, reset
  status.py   # Überblick über Index und Konfiguration
  config.py   # Konfiguration aus Umgebungsvariablen
  ingest.py   # Ordner einlesen -> Index (nur neue/geänderte Dateien)
  loaders.py  # Dateien finden, Text aus txt/md/pdf/docx lesen, Datei-Hash
  chunking.py # Text in überlappende Chunks teilen
  embeddings.py # gemeinsames Embedding-Modell für ingest und ask
  store.py    # ChromaDB-Zugriff (get_collection)
  ask.py      # Frage -> Fundstellen -> LLM-Antwort mit Quellen
  llm.py      # Sprachmodelle: Claude-API und Ollama
tests/
```

## Häufige Fehler

| Meldung | Lösung |
|---|---|
| `Kein Claude-API-Schlüssel gefunden` | `export ANTHROPIC_API_KEY=...` oder `MYRAG_LLM=ollama` |
| `Der Index ist leer` / `Kein Index unter ...` | erst `myrag ingest <ordner>`; im selben Verzeichnis arbeiten oder `MYRAG_DB_PATH` setzen |
| `Embedding-Modell konnte nicht geladen werden` | beim ersten Lauf Internet zu huggingface.co nötig |
| `Ollama ist unter ... nicht erreichbar` | `ollama serve` starten bzw. `OLLAMA_HOST` prüfen |

## Annahmen

- Der Index liegt standardmäßig unter `./.myrag/chroma` im aktuellen Verzeichnis.
  Wer `myrag` aus verschiedenen Verzeichnissen nutzt, setzt am besten
  `export MYRAG_DB_PATH=~/.myrag/chroma`.

## Tests

```bash
pytest
```
