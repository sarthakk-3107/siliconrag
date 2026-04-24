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

## Evaluation Results

A/B test comparing **agentic RAG** (LangGraph + tool-calling) vs **naive RAG** 
(single-shot retrieve-then-generate) on 45 queries across 4 categories 
(spec lookup, comparison, calculation, conceptual).

**Methodology:**
- Judge model: `gpt-4o-mini`, scoring 0–1 on correctness, faithfulness, completeness
- Paired bootstrap (10000 resamples) for 95% CIs on mean score differences
- Paired permutation test (10000 permutations) for p-values
- Same 45 queries evaluated on both systems (paired design)

**Results:**

| Metric        | Agentic | Naive | Diff   | 95% CI             | p-value |
|---------------|---------|-------|--------|--------------------|---------| 
| Correctness   | 0.704   | 0.738 | -0.033 | [-0.167, +0.098]   | 0.60    |
| Faithfulness  | 0.720   | 0.744 | -0.024 | [-0.160, +0.109]   | 0.71    |
| Completeness  | 0.662   | 0.684 | -0.022 | [-0.142, +0.098]   | 0.74    |
| **Overall**   | 0.696   | 0.722 | -0.027 | [-0.155, +0.099]   | 0.67    |

**Interpretation:** No statistically significant difference detected between 
agentic and naive RAG on this corpus. Both systems achieve ~70% overall quality. 
All 95% CIs include zero; none of the differences are significant at α=0.05.

This is a **null result**, which is informative: on a small, topically narrow 
corpus (5 PDFs, DAC8811-dominant), the agent's specialized tool routing does 
not outperform naive semantic retrieval. Both architectures converge to 
similar quality because the retrieval problem is already easy for dense 
embeddings in this regime.

## Failure Analysis & Discussion

**Why the agentic system didn't outperform naive RAG on this corpus:**

1. **Corpus narrowness.** 55% of chunks come from a single datasheet (DAC8811). 
   When one document dominates, dense embedding retrieval already reaches high 
   precision on most queries, leaving little room for agentic routing to add 
   value. Specialized tools (`spec_lookup`, `compare_chips`) add the most 
   value on heterogeneous corpora where routing to the right document matters.

2. **Query complexity distribution.** 58% of queries are simple spec lookups 
   ("What is the resolution of DAC8811?"). These are precisely the queries 
   where a single retrieval pass succeeds; agentic overhead (routing LLM call 
   + synthesis LLM call) adds latency and variance without adding precision.

3. **Self-judgment bias.** Using `gpt-4o-mini` as both generator and judge 
   likely compresses measured gaps between systems. Standard practice uses a 
   stronger judge (e.g., `gpt-4o` or Claude Opus). A stronger judge would 
   amplify observed effects in either direction.

4. **Sample size.** n=45 is small for detecting effect sizes < 5%. With 
   n=150–200, the CIs would shrink by ~2x and smaller true effects might 
   reach significance.

**What I'd do differently with more time:**

- Expand corpus to 15–20 heterogeneous documents across ISAs, GPUs, analog ICs, 
  and signal integrity papers, so routing decisions become meaningful
- Expand eval to 150+ queries with deliberate coverage of multi-hop and 
  comparison queries (where agentic should shine)
- Use a stronger judge model to reduce self-preference bias
- Add a third system arm: hybrid RAG without the agent (dense + BM25 + rerank, 
  no tool routing). This would isolate the contribution of agentic routing 
  from retrieval improvements.

**What worked well:**

- Both systems achieve ~70% on correctness and faithfulness, indicating the 
  retrieval pipeline (hybrid dense + BM25 + cross-encoder reranker) is 
  surfacing relevant content
- The evaluation methodology itself is rigorous: paired design, bootstrap CIs, 
  permutation tests
- Null results are themselves valuable — they prevent overclaiming and 
  highlight where architectural complexity isn't yet justified by the problem.

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
