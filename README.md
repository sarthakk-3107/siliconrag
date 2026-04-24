# SiliconRAG

**AI assistant for semiconductor design & validation documentation.** A domain-specialized RAG + agentic system that helps silicon engineers query chip datasheets, architecture manuals, and validation specs through natural language.

## Motivation

Silicon engineers spend significant time searching datasheets, TRMs, and whitepapers for specific parameters, doing side-by-side chip comparisons, and performing timing calculations. SiliconRAG reduces that friction with a tool-calling agent specialized for four common query patterns: spec lookup, chip comparison, timing calculation, and open-ended conceptual search.

## Architecture

```
User query
    │
    ▼
┌───────────────────────┐
│  LangGraph Agent      │  ← gpt-4o-mini, 0 temperature
│  (ReAct loop, max 5)  │
└───────┬───────────────┘
        │ tool call
        ▼
┌──────────────────────────────────────────────────────┐
│  Tools                                               │
│  • spec_lookup(param, chip)     ─┐                   │
│  • compare_chips(a, b, param)   ─┤──► Hybrid         │
│  • search_docs(query)            ─┘    Retriever     │
│  • calculate_timing(...)        ──► pure math        │
└──────────────────────────────────────────────────────┘
                                        │
                                        ▼
        ┌───────────────────────────────────────────┐
        │  Hybrid Retrieval                         │
        │  ┌─────────────┐  ┌──────────────┐        │
        │  │ Dense       │  │ BM25         │        │
        │  │ (OpenAI     │  │ (rank_bm25)  │        │
        │  │  text-      │  └──────────────┘        │
        │  │  embedding- │         │                │
        │  │  3-small)   │         │                │
        │  └─────────────┘         │                │
        │         │                │                │
        │         └────┬───────────┘                │
        │              ▼                            │
        │     Reciprocal Rank Fusion (k=60)         │
        │              │                            │
        │              ▼                            │
        │     Cross-encoder reranker                │
        │     (ms-marco-MiniLM-L-6-v2)              │
        └───────────────────────────────────────────┘
                       │
                       ▼
                  Top-5 chunks
                       │
                       ▼
               Agent synthesizes answer
```

## Key design decisions

- **Section-aware chunking** over fixed-size chunking: preserves the spec hierarchy in datasheets.
- **Hybrid retrieval (dense + BM25 + RRF + reranker)**: dense alone misses exact part numbers; BM25 alone misses conceptual queries.
- **Tool-calling over naive RAG**: specialized tools produce better structured answers for spec/comparison/timing queries.
- **LLM-as-Judge with gpt-4o** for evaluation; stronger judge than generator is standard practice.
- **Paired bootstrap + permutation tests** for statistically rigorous A/B comparison.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Download PDFs into data/raw/
# See "Data sources" below for suggestions

# 4. Ingest, embed, and run
python src/ingest.py       # PDF -> chunks
python src/embed.py         # chunks -> ChromaDB
python src/retrieve.py      # smoke test retrieval
python src/agent.py         # smoke test agent

# 5. Evaluate
python src/evaluate.py      # runs A/B with CIs

# 6. Serve
python src/api.py           # FastAPI on :8000
streamlit run app.py        # demo UI on :8501
```

## Data sources

Tested with public documentation from:
- **RISC-V specs** — https://riscv.org/technical/specifications/
- **ARM Cortex-M/A Technical Reference Manuals** — developer.arm.com
- **NVIDIA architecture whitepapers** — H100, A100, Blackwell
- **TI / ADI datasheets** — op-amps, ADCs, DACs
- **IEEE papers on signal integrity / timing analysis**

Target corpus: 15-25 documents, ~500-2000 pages total.

## Evaluation

Run `python src/evaluate.py` to produce a comparison table like:

```
======================================================================
  Agentic  vs  Naive   (n=100)
======================================================================
Metric              A        B     Diff              95% CI        p
----------------------------------------------------------------------
correctness      0.XXX    0.XXX   +0.XXX   [+0.XXX, +0.XXX]   0.XXXX*
faithfulness     0.XXX    0.XXX   +0.XXX   [+0.XXX, +0.XXX]   0.XXXX*
completeness     0.XXX    0.XXX   +0.XXX   [+0.XXX, +0.XXX]   0.XXXX
overall          0.XXX    0.XXX   +0.XXX   [+0.XXX, +0.XXX]   0.XXXX*
  * = 95% CI excludes 0 (statistically significant)
```

Per-category breakdown lives in `eval/results/`.

## Example queries

| Query | Category | Tool used |
|-------|----------|-----------|
| "What is the max operating frequency of the Cortex-M4?" | spec_lookup | `spec_lookup` |
| "Compare H100 vs A100 on memory bandwidth" | comparison | `compare_chips` |
| "If setup=2ns, hold=0.5ns, clock=5ns, does timing meet?" | calculation | `calculate_timing` |
| "Explain cache coherence in multi-core systems" | conceptual | `search_docs` |

## Failure modes observed

[Fill in after running eval. Examples:]
- Agent occasionally hallucinates specs when documents have conflicting values across datasheet revisions
- BM25 tokenization sometimes splits part numbers awkwardly (addressed with custom regex)
- Cross-encoder reranker can demote relevant table excerpts that were parsed into awkward formats

## Cost & latency

- Per query: ~$0.001-0.005 depending on agent iterations (gpt-4o-mini) + $0.001 embedding
- Per eval run (100 queries, 2 systems, LLM-as-Judge): ~$1-2
- Full corpus embedding: ~$0.01 for ~2000 chunks
- P95 latency (agentic mode, cold reranker load): ~2-4s; warm: ~800ms

## Future work

- Fine-tune a query classifier (DistilBERT + LoRA) to route queries to tool subsets before invoking the agent
- Schematic/diagram understanding via vision models
- Continuous eval pipeline triggered on new document ingestion
- Export answers with formatted source citations (markdown/LaTeX)

## License

MIT
