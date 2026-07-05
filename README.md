# second-brain

A fully local, privacy-first Second Brain system powered by LangChain, Ollama (Gemma 3), and ChromaDB for semantic search over personal notes and bookmarks. Nothing leaves your machine.

## Architecture

```
second-brain/
├── data/
│   ├── notes/          # your .md files (recursive)
│   └── bookmarks/      # browser bookmark exports (.html, Netscape format)
├── storage/chroma/     # persistent vector DB (auto-created, gitignored)
├── src/
│   ├── config.py       # all knobs in one place
│   ├── loaders.py      # markdown + bookmark loaders
│   ├── ingest.py       # load -> chunk -> embed -> persist
│   └── query.py        # retrieve -> prompt -> stream answer + cite sources
└── requirements.txt
```

Pipeline: markdown is split by headers (keeps sections semantically whole), then size-capped at 1000 chars / 150 overlap. Embeddings: `bge-m3` (multilingual). Retrieval: MMR (k=5 from a pool of 20) for relevant *and* diverse context. LLM: `gemma3:12b` — fits a 16GB RTX 4070 Ti SUPER with headroom.

## Setup

```powershell
# 1. Python env
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Pull models (Ollama must be running)
ollama pull gemma3:12b
ollama pull bge-m3
```

## Usage

```powershell
# Put .md files in data/notes/, bookmark HTML exports in data/bookmarks/

# Index everything (use --reset to rebuild from scratch)
python -m src.ingest

# Chat with your brain
python -m src.query
# or one-shot:
python -m src.query "what did I write about chunking strategies?"
```

## Extending

- New source type (PDF, Obsidian vault, RSS): add a `load_*` function in `src/loaders.py`, register it in `ingest()`.
- Different model: change one line in `src/config.py`.
- Better retrieval: swap the retriever in `src/query.py` (hybrid BM25, reranking, etc.).
