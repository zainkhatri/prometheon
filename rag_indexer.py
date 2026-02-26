#!/usr/bin/env python3
"""
PROMETHEUS RAG Indexer
Crawls the mergerfs pool, embeds text/code files with bge-m3 via Ollama,
stores vectors in Qdrant. Runs incrementally — only reprocesses changed files.

Usage:
  python3 rag_indexer.py              # index everything
  python3 rag_indexer.py --wipe       # wipe collection and reindex from scratch
  python3 rag_indexer.py --query "find all Python files using psutil"
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

# ─── Config ──────────────────────────────────────────────────────────────────

OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL    = "bge-m3"          # pull with: ollama pull bge-m3
EMBED_DIM      = 1024              # bge-m3 output dimension

QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION     = "prometheus_pool"

POOL_ROOT      = os.getenv("POOL_ROOT", "/srv/mergerfs/PROMETHEUS")
# Note: PROMETHEON app lives at /srv/mergerfs/PROMETHEUS/PROMETHEON
STATE_FILE     = os.getenv("INDEX_STATE", "/srv/qdrant-data/index_state.json")

CHUNK_SIZE     = 1500    # characters per chunk
CHUNK_OVERLAP  = 200     # overlap between chunks

# File types to index (everything else is skipped)
INDEXABLE_EXTS = {
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".sh", ".bash", ".zsh",
    ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".java", ".rb",
    ".php", ".swift", ".kt", ".cs", ".lua", ".r", ".m",
    # Config / data
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".env", ".env.example",
    # Docs
    ".md", ".txt", ".rst", ".org",
    # Web
    ".html", ".css", ".scss", ".xml",
    # Scripts / misc
    ".sql", ".dockerfile", "dockerfile",
    ".makefile", "makefile",
}

# Directories to always skip
SKIP_DIRS = {
    "journal-detection", "NASA", "mcc-genai-guild", "take-homes", "URAP BERKELEY",
    "journal-detection", "NASA", "mcc-genai-guild", "take-homes", "URAP BERKELEY",
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".DS_Store", "thumbs", "thumbs_hq",
    "PHOTOS", "Videos", "Movies",   # skip media — too large, not useful for RAG
}

MAX_FILE_BYTES = 500_000   # skip files over 500KB (likely minified/generated)

# ─── Embedding ───────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """Get embedding vector from Ollama bge-m3."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─── File crawling ───────────────────────────────────────────────────────────

def should_index(path: Path) -> bool:
    """True if this file should be embedded."""
    if path.name.startswith(".") and path.suffix not in {".env", ".env.example"}:
        return False
    ext = path.suffix.lower() or path.name.lower()
    if ext not in INDEXABLE_EXTS:
        return False
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
    except OSError:
        return False
    return True


def crawl(root: str) -> Iterator[Path]:
    """Walk the pool root, yielding indexable files."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place so os.walk doesn't descend
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            p = Path(dirpath) / fname
            if should_index(p):
                yield p


# ─── State tracking ──────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def file_key(path: Path) -> str:
    return str(path)


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime


# ─── Qdrant helpers ──────────────────────────────────────────────────────────

def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient, wipe: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if wipe and COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"  Wiped collection '{COLLECTION}'")
        existing.remove(COLLECTION)
    if COLLECTION not in existing:
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"  Created collection '{COLLECTION}'")


def delete_file_points(client: QdrantClient, filepath: str):
    """Remove all existing points for a file (before re-indexing)."""
    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="filepath", match=MatchValue(value=filepath))]
        ),
    )


def upsert_chunks(client: QdrantClient, filepath: str, chunks: list[str]):
    """Embed and upsert all chunks for a file."""
    points = []
    for i, chunk in enumerate(chunks):
        vec = embed(chunk)
        point_id = int(hashlib.md5(f"{filepath}:{i}".encode()).hexdigest()[:8], 16)
        points.append(PointStruct(
            id=point_id,
            vector=vec,
            payload={
                "filepath": filepath,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "snippet": chunk[:300],
            },
        ))
    if points:
        client.upsert(collection_name=COLLECTION, points=points)


# ─── Main indexing loop ───────────────────────────────────────────────────────

def run_index(wipe: bool = False):
    print(f"\nPROMETHEUS RAG Indexer")
    print(f"  Pool:    {POOL_ROOT}")
    print(f"  Ollama:  {OLLAMA_HOST}  model={EMBED_MODEL}")
    print(f"  Qdrant:  {QDRANT_HOST}:{QDRANT_PORT}  collection={COLLECTION}")
    print()

    # Verify Ollama is reachable and model exists
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(EMBED_MODEL in m for m in models):
            print(f"⚠  Model '{EMBED_MODEL}' not found in Ollama. Pull it:")
            print(f"   ollama pull {EMBED_MODEL}")
            sys.exit(1)
    except requests.RequestException as e:
        print(f"❌ Ollama unreachable: {e}")
        sys.exit(1)

    client = get_client()
    ensure_collection(client, wipe=wipe)

    state = {} if wipe else load_state()
    new_state = {}

    indexed = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for path in crawl(POOL_ROOT):
        key = file_key(path)
        try:
            mtime = file_mtime(path)
        except OSError:
            continue

        # Skip if unchanged
        if not wipe and state.get(key) == mtime:
            new_state[key] = mtime
            skipped += 1
            continue

        try:
            text = path.read_text(errors="ignore")
        except OSError as e:
            print(f"  ⚠ read error: {path}: {e}")
            errors += 1
            continue

        if not text.strip():
            new_state[key] = mtime
            skipped += 1
            continue

        chunks = chunk_text(text)
        try:
            # Remove old points for this file, then insert fresh
            delete_file_points(client, key)
            upsert_chunks(client, key, chunks)
        except Exception as e:
            print(f"  ❌ upsert failed: {path}: {e}")
            errors += 1
            continue

        new_state[key] = mtime
        indexed += 1
        if indexed % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{indexed} indexed | {skipped} skipped | {errors} errors | {elapsed:.0f}s]  {path.name}")

    save_state(new_state)
    elapsed = time.time() - t0
    print(f"\n✅ Done — {indexed} indexed, {skipped} unchanged, {errors} errors — {elapsed:.1f}s")


# ─── Query mode ──────────────────────────────────────────────────────────────

def run_query(query: str, top_k: int = 5):
    print(f"\nQuerying: {query!r}\n")
    vec = embed(query)
    client = get_client()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
    )
    if not results:
        print("No results found.")
        return
    for i, hit in enumerate(results, 1):
        p = hit.payload
        print(f"[{i}] score={hit.score:.3f}  {p['filepath']}  (chunk {p['chunk_index']+1}/{p['total_chunks']})")
        print(f"    {p['snippet'][:200].replace(chr(10), ' ')}")
        print()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PROMETHEUS RAG Indexer")
    parser.add_argument("--wipe", action="store_true", help="Wipe collection and reindex from scratch")
    parser.add_argument("--query", metavar="TEXT", help="Query mode: search the index")
    parser.add_argument("--top-k", type=int, default=5, help="Results to return in query mode")
    args = parser.parse_args()

    if args.query:
        run_query(args.query, top_k=args.top_k)
    else:
        run_index(wipe=args.wipe)
