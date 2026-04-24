"""
Tool definitions for the SiliconRAG agent.

Design: 4 specialized tools, each wrapping retrieval with a domain-specific prompt.
The agent decides which to call based on query intent. Tool-based routing beats
naive RAG on structured queries (spec lookup, comparison) because:
1. Each tool has a purpose-built prompt that structures the output
2. Metadata filters can be applied per tool type
3. Multi-tool queries ("compare A and B on power") become natural
"""

from typing import Optional

from langchain_core.tools import tool

from retrieve import HybridRetriever, RetrievedChunk


# Singleton retriever - initialized on first use
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(use_reranker=True)
    return _retriever


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Format chunks as context for the LLM."""
    if not chunks:
        return "No relevant documentation found."
    formatted = []
    for i, c in enumerate(chunks, 1):
        formatted.append(
            f"[Source {i}] {c.doc_name} / {c.section} (page {c.page_num})\n"
            f"{c.text}"
        )
    return "\n\n---\n\n".join(formatted)


@tool
def spec_lookup(parameter: str, chip: str) -> str:
    """
    Look up a specific electrical, timing, or performance specification for a chip.

    Use this for queries like:
      - "What is the max operating frequency of Cortex-M4?"
      - "VDD range for the LM358"
      - "setup time for DDR4 at 3200MT/s"

    Args:
        parameter: The spec being looked up (e.g., "max frequency", "VDD", "TDP")
        chip: The chip or component name (e.g., "Cortex-M4", "H100", "LM358")
    """
    retriever = get_retriever()
    query = f"{parameter} {chip}"
    chunks = retriever.retrieve(query, top_k=5, mode="hybrid")
    return format_chunks(chunks)


@tool
def compare_chips(chip_a: str, chip_b: str, parameter: str) -> str:
    """
    Compare two chips on a specific parameter by retrieving specs from both.

    Use this for queries like:
      - "Compare power consumption of A100 vs H100"
      - "RISC-V vs Cortex-M on cache architecture"

    Args:
        chip_a: First chip/component
        chip_b: Second chip/component
        parameter: What to compare on (e.g., "power", "frequency", "cache size")
    """
    retriever = get_retriever()
    chunks_a = retriever.retrieve(f"{parameter} {chip_a}", top_k=3, mode="hybrid")
    chunks_b = retriever.retrieve(f"{parameter} {chip_b}", top_k=3, mode="hybrid")
    return (
        f"=== {chip_a} ({parameter}) ===\n{format_chunks(chunks_a)}\n\n"
        f"=== {chip_b} ({parameter}) ===\n{format_chunks(chunks_b)}"
    )


@tool
def calculate_timing(
    setup_ns: float,
    hold_ns: float,
    clock_period_ns: float,
    propagation_delay_ns: float = 0.0,
) -> str:
    """
    Calculate timing margins and determine if timing is met.

    Use this for queries involving explicit timing math:
      - "If setup is 2ns, hold is 0.5ns, clock period is 5ns, does it meet timing?"
      - "What's the slack with 1ns prop delay and 3ns clock?"

    Args:
        setup_ns: Required setup time in nanoseconds
        hold_ns: Required hold time in nanoseconds
        clock_period_ns: Clock period in nanoseconds
        propagation_delay_ns: Combinational path delay in nanoseconds (optional)
    """
    max_freq_mhz = 1000.0 / clock_period_ns
    setup_slack = clock_period_ns - setup_ns - propagation_delay_ns
    hold_slack = propagation_delay_ns - hold_ns
    timing_met = setup_slack >= 0 and hold_slack >= 0

    return (
        f"Timing Analysis:\n"
        f"  Clock period: {clock_period_ns} ns ({max_freq_mhz:.1f} MHz max)\n"
        f"  Setup requirement: {setup_ns} ns\n"
        f"  Hold requirement: {hold_ns} ns\n"
        f"  Propagation delay: {propagation_delay_ns} ns\n"
        f"  Setup slack: {setup_slack:.3f} ns {'(MET)' if setup_slack >= 0 else '(VIOLATED)'}\n"
        f"  Hold slack: {hold_slack:.3f} ns {'(MET)' if hold_slack >= 0 else '(VIOLATED)'}\n"
        f"  Overall: {'TIMING MET' if timing_met else 'TIMING VIOLATED'}"
    )


@tool
def search_docs(query: str) -> str:
    """
    Free-form semantic search over all semiconductor documentation.

    Use this as a fallback for conceptual or open-ended queries:
      - "How does cache coherence work in multi-core systems?"
      - "What is signal integrity?"
      - "Explain the difference between SRAM and DRAM"

    Args:
        query: The natural language query
    """
    retriever = get_retriever()
    chunks = retriever.retrieve(query, top_k=5, mode="hybrid")
    return format_chunks(chunks)


ALL_TOOLS = [spec_lookup, compare_chips, calculate_timing, search_docs]
