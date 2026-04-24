"""
FastAPI REST endpoint for SiliconRAG.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from agent import build_agent, run_agent, run_naive_rag


class QueryRequest(BaseModel):
    query: str
    mode: str = "agentic"  # "agentic" or "naive"


class QueryResponse(BaseModel):
    answer: str
    tool_calls: list[dict]
    sources: list[str]
    iterations: int
    mode: str


agent_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: build agent once, reuse across requests
    print("Building agent...")
    agent_holder["agent"] = build_agent()
    print("Agent ready.")
    yield
    agent_holder.clear()


app = FastAPI(
    title="SiliconRAG",
    description="AI assistant for semiconductor documentation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if req.mode == "naive":
        response = run_naive_rag(req.query)
    else:
        response = run_agent(req.query, agent_holder.get("agent"))

    return QueryResponse(
        answer=response.answer,
        tool_calls=response.tool_calls,
        sources=response.sources,
        iterations=response.iterations,
        mode=req.mode,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
