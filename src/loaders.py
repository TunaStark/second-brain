"""Document loaders: markdown notes + browser bookmark exports.

Each loader returns list[Document] with rich metadata so answers can cite sources.
Extensible: add a new load_* function, register it in ingest.py.
"""

import logging
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

ENCODINGS = ("utf-8", "utf-8-sig", "cp1254", "latin-1")  # tried in order


def _read_text(path: Path) -> str | None:
    """Read text file, surviving encoding chaos. Returns None if unreadable."""
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return None
    logger.warning("Undecodable file skipped: %s", path)
    return None


def load_markdown_notes(notes_dir: Path) -> list[Document]:
    """Load every .md file under notes_dir (recursive)."""
    docs: list[Document] = []
    if not notes_dir.is_dir():
        logger.warning("Notes directory missing: %s", notes_dir)
        return docs

    for path in sorted(notes_dir.rglob("*.md")):
        text = _read_text(path)
        if not text or not text.strip():
            logger.info("Empty/unreadable note skipped: %s", path.name)
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path.relative_to(notes_dir)),
                    "source_type": "note",
                    "title": path.stem,
                },
            )
        )
    logger.info("Loaded %d markdown notes from %s", len(docs), notes_dir)
    return docs


def load_bookmarks(bookmarks_dir: Path) -> list[Document]:
    """Parse Netscape-format bookmark exports (.html) from Chrome/Firefox/Edge.

    Each bookmark becomes one small Document: title + URL + folder path.
    """
    docs: list[Document] = []
    if not bookmarks_dir.is_dir():
        logger.warning("Bookmarks directory missing: %s", bookmarks_dir)
        return docs

    for path in sorted(bookmarks_dir.glob("*.html")):
        text = _read_text(path)
        if not text:
            continue
        soup = BeautifulSoup(text, "lxml")
        for anchor in soup.find_all("a"):
            url = anchor.get("href", "").strip()
            title = anchor.get_text(strip=True)
            if not url or not url.startswith(("http://", "https://")):
                continue
            # folder = nearest preceding H3 (Netscape format nests folders as H3)
            folder_tag = anchor.find_previous("h3")
            folder = folder_tag.get_text(strip=True) if folder_tag else ""
            content = f"Bookmark: {title}\nURL: {url}"
            if folder:
                content += f"\nFolder: {folder}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "source_type": "bookmark",
                        "title": title or url,
                    },
                )
            )
    logger.info("Loaded %d bookmarks from %s", len(docs), bookmarks_dir)
    return docs
