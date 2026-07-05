# Welcome to Second Brain

This is a sample note so the first ingest run has something to index.

## What is this project?

Second Brain is a fully local, privacy-first RAG system. Your notes and
bookmarks never leave this machine. Embeddings come from `bge-m3`, answers
come from `gemma3` — both served by Ollama on your own GPU.

## How to use it

- Drop your Markdown notes into `data/notes/` (subfolders are fine).
- Export your browser bookmarks as HTML into `data/bookmarks/`.
- Run `python -m src.ingest` to index everything.
- Run `python -m src.query` and start asking questions.
