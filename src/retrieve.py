"""
Hybrid retrieval combining dense embeddings (semantic) + BM25 (lexical).

Design decisions:
- Why hybrid? Pure dense retrieval misses exact part numbers ("LM358AN") and
  specific pin names. Pure BM25 misses conceptual queries ("how does the cache
  coherence protocol work"). Hybrid catches both.
- Reciprocal Rank Fusion (RRF): simple, parameter-free way to merge rankings.
  Works better than weighted score combination because scores from different
  retrievers aren't comparable.
- Cross-encoder reranker: re-scores top candidates with a model that sees the
  query and document together. Slower but much more precise than bi-encoder alone.
- k=20 initial retrieval, rerank to top-5: standard pipeline. Cast a wide net,
  then precision-filter.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from embed import (
    COLLECTION_NAME,
    embed_query,
    get_chroma_client,
    get_openai_client,
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_name: str
    section: str
    page_num: int
    chip_family: str
    score: float  # final score (higher is better)
    source: str   # "dense", "bm25", "hybrid", or "reranked"


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple tokenization for BM25. Lowercase, split on whitespace + basic punctuation."""
    import re
    text = text.lower()
    # Preserve part numbers like "LM358AN" as single tokens
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    return [t for t in tokens if len(t) > 1]


class HybridRetriever:
    """Combines ChromaDB dense retrieval with BM25 lexical retrieval."""

    def __init__(
        self,
        chroma_client: Optional[chromadb.ClientAPI] = None,
        use_reranker: bool = True,
    ):
        self.openai = get_openai_client()
        self.chroma = chroma_client or get_chroma_client()
        self.collection = self.chroma.get_collection(COLLECTION_NAME)

        # Load all documents into memory for BM25
        # For 500-2000 chunks this is fine. At 100K+ you'd use Elasticsearch.
        print("Loading corpus for BM25...")
        all_docs = self.collection.get()
        self.all_ids = all_docs["ids"]
        self.all_texts = all_docs["documents"]
        self.all_metadatas = all_docs["metadatas"]

        tokenized_corpus = [tokenize_for_bm25(t) for t in self.all_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.use_reranker = use_reranker
        self.reranker: Optional[CrossEncoder] = None
        if use_reranker:
            print("Loading cross-encoder reranker...")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print(f"Retriever ready. Corpus size: {len(self.all_ids)} chunks")

    def _dense_search(
        self, query: str, k: int, chip_filter: Optional[str] = None
    ) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) from dense search."""
        query_emb = embed_query(query, self.openai)
        where = {"chip_family": chip_filter} if chip_filter else None
        result = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            where=where,
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Convert cosine distance -> similarity (0-1)
        scores = [1.0 - d for d in distances]
        return list(zip(ids, scores))

    def _bm25_search(
        self, query: str, k: int, chip_filter: Optional[str] = None
    ) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) from BM25 search."""
        query_tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(query_tokens)

        # Apply chip filter if requested
        indexed = list(enumerate(scores))
        if chip_filter:
            indexed = [
                (i, s) for i, s in indexed
                if self.all_metadatas[i].get("chip_family") == chip_filter
            ]

        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:k]
        return [(self.all_ids[i], float(s)) for i, s in top]

    def _reciprocal_rank_fusion(
        self,
        rankings: list[list[tuple[str, float]]],
        k_constant: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.
        Formula: RRF_score(d) = sum over rankers of 1 / (k + rank(d))
        k=60 is standard (Cormack et al., 2009).
        """
        rrf_scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, (doc_id, _score) in enumerate(ranking):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                    k_constant + rank + 1
                )
        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return merged

    def _rerank(
        self, query: str, candidates: list[tuple[str, float]], top_k: int
    ) -> list[tuple[str, float]]:
        """Cross-encoder reranking for precision."""
        if not self.reranker or not candidates:
            return candidates[:top_k]

        # Fetch texts for candidates
        id_to_text = {cid: self.all_texts[self.all_ids.index(cid)]
                      for cid, _ in candidates}

        pairs = [[query, id_to_text[cid]] for cid, _ in candidates]
        scores = self.reranker.predict(pairs)

        reranked = [
            (cid, float(score))
            for (cid, _), score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def _build_retrieved(
        self, ranked: list[tuple[str, float]], source: str
    ) -> list[RetrievedChunk]:
        """Hydrate chunk IDs into RetrievedChunk objects."""
        results = []
        for chunk_id, score in ranked:
            try:
                idx = self.all_ids.index(chunk_id)
            except ValueError:
                continue
            meta = self.all_metadatas[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=self.all_texts[idx],
                    doc_name=meta.get("doc_name", ""),
                    section=meta.get("section", ""),
                    page_num=int(meta.get("page_num", 0)),
                    chip_family=meta.get("chip_family", "unknown"),
                    score=score,
                    source=source,
                )
            )
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",  # "dense", "bm25", "hybrid"
        chip_filter: Optional[str] = None,
        initial_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Main retrieval entry point."""
        if mode == "dense":
            ranked = self._dense_search(query, top_k, chip_filter)
            return self._build_retrieved(ranked, "dense")

        if mode == "bm25":
            ranked = self._bm25_search(query, top_k, chip_filter)
            return self._build_retrieved(ranked, "bm25")

        # Hybrid: retrieve more candidates, merge with RRF, optionally rerank
        dense = self._dense_search(query, initial_k, chip_filter)
        bm25 = self._bm25_search(query, initial_k, chip_filter)
        fused = self._reciprocal_rank_fusion([dense, bm25])

        if self.use_reranker:
            reranked = self._rerank(query, fused[: initial_k], top_k)
            return self._build_retrieved(reranked, "reranked")
        return self._build_retrieved(fused[:top_k], "hybrid")


if __name__ == "__main__":
    # Smoke test
    retriever = HybridRetriever(use_reranker=False)
    test_queries = [
        "What is the maximum operating frequency of the Cortex-M4?",
        "RISC-V vector extension",
        "thermal design power",
    ]
    for q in test_queries:
        print(f"\n--- Query: {q}")
        results = retriever.retrieve(q, top_k=3, mode="hybrid")
        for r in results:
            print(f"  [{r.score:.3f}] {r.doc_name} / {r.section[:50]} (p.{r.page_num})")
            print(f"    {r.text[:150]}...")
