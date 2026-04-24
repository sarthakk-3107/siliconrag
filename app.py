"""
Streamlit demo UI for SiliconRAG.
Chat interface with sources panel and mode toggle.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import build_agent, run_agent, run_naive_rag

st.set_page_config(page_title="SiliconRAG", page_icon="⚡", layout="wide")
st.title("⚡ SiliconRAG")
st.caption("AI assistant for semiconductor design & validation docs")

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Retrieval mode",
        options=["Agentic (tool-calling)", "Naive RAG (baseline)"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Example queries:**")
    examples = [
        "What is the max operating frequency of Cortex-M4?",
        "Compare H100 and A100 on memory bandwidth",
        "If setup=2ns, hold=0.5ns, clock=5ns, does timing meet?",
        "Explain cache coherence in multi-core systems",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["pending_query"] = ex

# Build agent once
if "agent" not in st.session_state:
    with st.spinner("Loading agent..."):
        st.session_state["agent"] = build_agent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")
        if msg.get("tool_calls"):
            with st.expander(f"Tool calls ({len(msg['tool_calls'])})"):
                for tc in msg["tool_calls"]:
                    st.code(f"{tc['name']}({tc['args']})", language="python")

# Input
user_input = st.chat_input("Ask about chip specs, comparisons, or concepts...")
if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if mode.startswith("Naive"):
                response = run_naive_rag(user_input)
            else:
                response = run_agent(user_input, st.session_state["agent"])
        st.markdown(response.answer)
        if response.sources:
            with st.expander(f"Sources ({len(response.sources)})"):
                for src in response.sources:
                    st.markdown(f"- {src}")
        if response.tool_calls:
            with st.expander(f"Tool calls ({len(response.tool_calls)})"):
                for tc in response.tool_calls:
                    st.code(f"{tc['name']}({tc['args']})", language="python")

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "tool_calls": response.tool_calls,
        }
    )
