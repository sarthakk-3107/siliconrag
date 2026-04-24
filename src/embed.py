"""
Embedding pipeline with OpenAI text-embedding-3-small and ChromaDB persistence.

Design decisions:
- text-embedding-3-small: good quality/cost tradeoff ($0.02 / 1M tokens).
  For ~500 chunks averaging 600 tokens, total cost ~$0.006. Essentially free.
- Persistent ChromaDB: avoids re-embedding on every run. Critical for iteration speed.
- Batch embedding (100 at a time): OpenAI API supports batches, 10x faster than
  one-at-a-time calls.
- Metadata stored alongside vectors: enables filtered retrieval (e.g., "only search
  ARM Cortex docs").
"""

import json
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100
COLLECTION_NAME = "silicon_docs"


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to a .env file in project root."
        )
    return OpenAI(api_key=api_key)


def get_chroma_client(persist_dir: Optional[Path] = None) -> chromadb.ClientAPI:
    if persist_dir is None:
        persist_dir = Path(__file__).parent.parent / "data" / "chroma_db"
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def embed_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Embed a batch of texts via OpenAI API."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # Response data is in same order as input
    return [item.embedding for item in resp.data]


def load_chunks(chunks_path: Path) -> list[dict]:
    with open(chunks_path) as f:
        return json.load(f)


def build_collection(
    chunks: list[dict],
    chroma_client: chromadb.ClientAPI,
    openai_client: OpenAI,
    reset: bool = False,
) -> chromadb.Collection:
    """Embed chunks and load into ChromaDB."""
    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    # Check what's already embedded (resume-friendly)
    existing_ids = set(collection.get()["ids"])
    to_embed = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not to_embed:
        print(f"All {len(chunks)} chunks already embedded. Skipping.")
        return collection

    print(f"Embedding {len(to_embed)} new chunks "
          f"({len(existing_ids)} already done)...")

    for i in tqdm(range(0, len(to_embed), BATCH_SIZE), desc="Embedding"):
        batch = to_embed[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        try:
            embeddings = embed_batch(texts, openai_client)
        except Exception as e:
            print(f"  Batch {i} failed: {e}. Skipping.")
            continue

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "doc_name": c["doc_name"],
                    "section": c["section"],
                    "page_num": c["page_num"],
                    "chip_family": c.get("chip_family") or "unknown",
                    "token_count": c["token_count"],
                }
                for c in batch
            ],
        )

    total = collection.count()
    print(f"Collection '{COLLECTION_NAME}' now has {total} chunks")
    return collection


def embed_query(query: str, client: OpenAI) -> list[float]:
    """Embed a single query string."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return resp.data[0].embedding


if __name__ == "__main__":
    chunks_path = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"No chunks found at {chunks_path}. Run ingest.py first."
        )

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    openai_client = get_openai_client()
    chroma_client = get_chroma_client()

    collection = build_collection(chunks, chroma_client, openai_client)
    print(f"\nDone. Sample chunk IDs:")
    sample = collection.get(limit=3)
    for cid in sample["ids"]:
        print(f"  {cid}")
