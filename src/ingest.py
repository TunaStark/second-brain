"""Ingestion pipeline: load -> chunk -> embed -> persist.

Run:  python -m src.ingest [--reset]
"""

import argparse
import json
import logging
import shutil
import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src import config
from src.loaders import load_bookmarks, load_markdown_notes

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Header-aware markdown splitting, then size-capped recursive splitting.

    Header split keeps semantic units (a section stays one chunk);
    recursive split guards against giant sections blowing the size cap.
    Bookmarks are tiny — they pass through untouched.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=config.MD_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata.get("source_type") == "bookmark":
            chunks.append(doc)
            continue
        sections = header_splitter.split_text(doc.page_content)
        for section in sections:
            section.metadata.update(doc.metadata)  # keep source/title
        chunks.extend(size_splitter.split_documents(sections))

    # Chroma rejects non-scalar metadata; flatten defensively.
    for chunk in chunks:
        chunk.metadata = {
            k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))
        }
    return chunks


def ingest(reset: bool = False) -> None:
    if reset and config.STORAGE_DIR.exists():
        logger.info("Resetting vector store at %s", config.STORAGE_DIR)
        shutil.rmtree(config.STORAGE_DIR)
    if reset:
        config.DOCSTORE_PATH.unlink(missing_ok=True)

    # Ensure data dirs exist so first run tells the user where to put files.
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    config.BOOKMARKS_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_markdown_notes(config.NOTES_DIR) + load_bookmarks(config.BOOKMARKS_DIR)
    if not docs:
        logger.error(
            "No documents found. Put .md files in %s and/or bookmark exports (.html) in %s",
            config.NOTES_DIR,
            config.BOOKMARKS_DIR,
        )
        sys.exit(1)

    chunks = chunk_documents(docs)
    logger.info("Chunked %d documents into %d chunks", len(docs), len(chunks))

    # Persist raw chunks as JSONL: BM25 rebuilds its sparse index from this at query time.
    config.DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DOCSTORE_PATH, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {"page_content": chunk.page_content, "metadata": chunk.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("Docstore for BM25 written to %s", config.DOCSTORE_PATH)

    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    try:
        embeddings.embed_query("connectivity check")
    except Exception as exc:
        logger.error(
            "Cannot reach Ollama at %s (%s). Is it running? Did you `ollama pull %s`?",
            config.OLLAMA_BASE_URL,
            exc,
            config.EMBEDDING_MODEL,
        )
        sys.exit(1)

    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.STORAGE_DIR),
    )

    # Batch inserts: steady progress, bounded memory.
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        logger.info("Embedded %d / %d chunks", min(i + batch_size, len(chunks)), len(chunks))

    logger.info("Done. Vector store persisted at %s", config.STORAGE_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest notes and bookmarks into the vector store.")
    parser.add_argument("--reset", action="store_true", help="Wipe the vector store before ingesting.")
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
