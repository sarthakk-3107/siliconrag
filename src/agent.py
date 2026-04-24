"""
LangGraph agent for SiliconRAG.

Architecture: ReAct-style agent loop.
  1. LLM receives query + system prompt describing tools
  2. LLM either calls a tool or produces a final answer
  3. Tool results are appended to messages, loop continues
  4. Max 5 iterations to prevent runaway loops

Design decisions:
- gpt-4o-mini for the agent: fast, cheap, and smart enough for tool routing.
  Upgrade to gpt-4o only if routing quality becomes a problem.
- System prompt emphasizes domain expertise and source attribution.
- Temperature 0 for reproducible tool routing. The creativity should come
  from the knowledge in the docs, not the model.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a silicon design and validation assistant helping engineers \
query chip documentation (datasheets, architecture manuals, validation specs).

You have access to these tools:
1. spec_lookup(parameter, chip) - Get specific electrical/timing/performance specs
2. compare_chips(chip_a, chip_b, parameter) - Side-by-side spec comparison
3. calculate_timing(setup, hold, clock_period, prop_delay) - Timing math
4. search_docs(query) - Free-form semantic search for conceptual questions

Rules:
- Always cite sources using [Source N] notation from the tool output.
- For spec queries, prefer spec_lookup over search_docs.
- For comparison queries, use compare_chips.
- For timing calculations with concrete numbers, call calculate_timing.
- If the documentation does not contain the answer, say so clearly. Do not invent specs.
- Be concise. Engineers want the number/answer first, then context.
"""

MAX_ITERATIONS = 5


class AgentState(TypedDict):
    messages: list
    iterations: int


@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_iterations: int = MAX_ITERATIONS


@dataclass
class AgentResponse:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    iterations: int = 0
    sources: list[str] = field(default_factory=list)


def build_agent(config: Optional[AgentConfig] = None):
    """Build and compile the LangGraph agent."""
    config = config or AgentConfig()

    llm = ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        """LLM decides what to do next."""
        response = llm_with_tools.invoke(state["messages"])
        return {
            "messages": state["messages"] + [response],
            "iterations": state["iterations"] + 1,
        }

    def should_continue(state: AgentState) -> str:
        """Decide whether to call tools or end."""
        if state["iterations"] >= config.max_iterations:
            return "end"
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    def tools_node(state: AgentState) -> dict:
        """Execute tool calls and append results."""
        result = tool_node.invoke({"messages": state["messages"]})
        return {
            "messages": state["messages"] + result["messages"],
            "iterations": state["iterations"],
        }

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_agent(query: str, agent=None) -> AgentResponse:
    """Run a single query through the agent."""
    if agent is None:
        agent = build_agent()

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ],
        "iterations": 0,
    }
    final_state = agent.invoke(initial_state)

    # Extract final answer
    final_answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    # Extract tool calls made
    tool_calls = []
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc["name"], "args": tc["args"]})

    # Extract sources mentioned in tool outputs
    sources = []
    for msg in final_state["messages"]:
        if isinstance(msg, ToolMessage):
            # Parse "[Source N] doc_name / section" patterns
            import re
            for match in re.finditer(
                r"\[Source \d+\]\s+([^\n/]+)\s*/\s*([^\n(]+)", msg.content
            ):
                src = f"{match.group(1).strip()} / {match.group(2).strip()}"
                if src not in sources:
                    sources.append(src)

    return AgentResponse(
        answer=final_answer,
        tool_calls=tool_calls,
        iterations=final_state["iterations"],
        sources=sources,
    )


# Naive RAG baseline for A/B comparison
def run_naive_rag(query: str) -> AgentResponse:
    """Baseline: single retrieval + answer, no agent, no tool routing."""
    from retrieve import HybridRetriever

    # Reuse shared retriever if possible
    from tools import get_retriever
    retriever = get_retriever()

    chunks = retriever.retrieve(query, top_k=5, mode="dense")  # dense-only, no hybrid
    context = "\n\n".join(
        f"[Source {i+1}] {c.doc_name}\n{c.text}" for i, c in enumerate(chunks)
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"Cite sources as [Source N].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return AgentResponse(
        answer=response.content,
        tool_calls=[],
        iterations=1,
        sources=[c.doc_name for c in chunks],
    )


if __name__ == "__main__":
    agent = build_agent()
    test_queries = [
        "What is the maximum operating frequency of the Cortex-M4?",
        "Compare the H100 and A100 on memory bandwidth",
        "If setup is 2ns, hold is 0.5ns, and clock period is 5ns with 1ns prop delay, does timing meet?",
        "Explain how cache coherence protocols work",
    ]
    for q in test_queries:
        print(f"\n{'='*60}\nQuery: {q}\n{'='*60}")
        response = run_agent(q, agent)
        print(f"Answer: {response.answer}")
        print(f"Tools called: {[tc['name'] for tc in response.tool_calls]}")
        print(f"Iterations: {response.iterations}")
