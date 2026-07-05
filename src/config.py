"""Central config. One place. Change here, everything follows."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"          # your .md files live here
BOOKMARKS_DIR = DATA_DIR / "bookmarks"  # browser bookmark exports (.html) live here
STORAGE_DIR = PROJECT_ROOT / "storage" / "chroma"

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma3:12b"        # 12B fits comfortably in 16GB VRAM with room for context
EMBEDDING_MODEL = "bge-m3"      # strong multilingual embeddings, 1024-dim
LLM_TEMPERATURE = 0.1           # low temp = factual RAG answers
LLM_NUM_CTX = 8192              # context window; bge-m3 chunks fit easily

# --- Vector store ---
COLLECTION_NAME = "second_brain"

# --- Chunking ---
CHUNK_SIZE = 1000               # chars, not tokens; ~250 tokens per chunk
CHUNK_OVERLAP = 150
MD_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# --- Retrieval ---
RETRIEVAL_K = 5                 # final chunks fed to LLM
RETRIEVAL_FETCH_K = 20          # MMR candidate pool
RETRIEVAL_LAMBDA = 0.5          # MMR relevance/diversity balance
