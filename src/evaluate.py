"""
Evaluation harness with LLM-as-Judge and bootstrap confidence intervals.

Design decisions:
- LLM-as-Judge with gpt-4o (stronger than generator): standard practice.
  Judge is given query, answer, and ground truth, scores 0-1 on three axes:
  correctness, faithfulness, completeness.
- Paired bootstrap with 10000 resamples: gives 95% CIs on mean score differences.
  "Paired" because we evaluate both systems on the same queries.
- Permutation test for p-values: non-parametric, no distributional assumptions.
- Report per-category breakdowns: reveals where each system wins/loses.
"""

import json
import os
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


@dataclass
class EvalQuery:
    query_id: str
    category: str  # "spec_lookup" | "comparison" | "calculation" | "conceptual"
    query: str
    ground_truth: str  # key facts the answer must contain
    ground_truth_sources: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    query_id: str
    category: str
    query: str
    answer: str
    correctness: float
    faithfulness: float
    completeness: float
    tool_calls: list[str] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def overall(self) -> float:
        return (self.correctness + self.faithfulness + self.completeness) / 3


JUDGE_SYSTEM_PROMPT = """You are an expert semiconductor engineer evaluating answers \
from an AI assistant.

Score the answer on three dimensions, each 0.0-1.0:
- correctness: Does the answer match the ground truth key facts? (0 = wrong, 1 = fully correct)
- faithfulness: Is the answer grounded in the provided sources with no hallucinations? \
(0 = invented facts, 1 = fully grounded)
- completeness: Does the answer cover all key aspects of the ground truth? \
(0 = misses the point, 1 = comprehensive)

Respond with ONLY a JSON object: {"correctness": 0.X, "faithfulness": 0.X, "completeness": 0.X, "reasoning": "brief explanation"}
"""


def judge_answer(
    query: str,
    answer: str,
    ground_truth: str,
    client: OpenAI,
    judge_model: str = "gpt-4o-mini",
) -> dict:
    """Score an answer using LLM-as-Judge."""
    user_prompt = (
        f"Query: {query}\n\n"
        f"Ground truth (key facts answer should include):\n{ground_truth}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        f"Provide your scores as JSON."
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def run_evaluation(
    queries: list[EvalQuery],
    system_fn: Callable,
    system_name: str,
    judge_client: OpenAI,
) -> list[EvalResult]:
    """Run a system on all queries and score with LLM judge."""
    import time
    results = []
    for q in tqdm(queries, desc=f"Evaluating {system_name}"):
        t0 = time.time()
        try:
            response = system_fn(q.query)
            answer = response.answer
            tool_calls = [tc["name"] for tc in response.tool_calls]
        except Exception as e:
            print(f"  FAILED on {q.query_id}: {e}")
            answer = f"[ERROR: {e}]"
            tool_calls = []
        latency = time.time() - t0

        try:
            scores = judge_answer(q.query, answer, q.ground_truth, judge_client)
        except Exception as e:
            print(f"  JUDGE FAILED on {q.query_id}: {e}")
            scores = {"correctness": 0.0, "faithfulness": 0.0, "completeness": 0.0}

        results.append(
            EvalResult(
                query_id=q.query_id,
                category=q.category,
                query=q.query,
                answer=answer,
                correctness=float(scores.get("correctness", 0)),
                faithfulness=float(scores.get("faithfulness", 0)),
                completeness=float(scores.get("completeness", 0)),
                tool_calls=tool_calls,
                latency_s=latency,
            )
        )
    return results


def bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap confidence interval on mean(A) - mean(B).
    Returns dict with mean_diff, ci_lower, ci_upper.
    """
    assert len(scores_a) == len(scores_b), "Paired bootstrap needs same-length arrays"
    rng = np.random.default_rng(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "significant": bool(lower > 0 or upper < 0),
    }


def permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """
    Paired permutation test. Null: scores_a and scores_b come from same distribution.
    Returns two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    observed = abs(a.mean() - b.mean())

    diffs = a - b
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_mean = (signs * diffs).mean()
        if abs(perm_mean) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def compare_systems(
    results_a: list[EvalResult],
    results_b: list[EvalResult],
    name_a: str,
    name_b: str,
) -> dict:
    """Compare two systems on the same queries with CIs and p-values."""
    # Align by query_id
    b_by_id = {r.query_id: r for r in results_b}
    paired_a, paired_b = [], []
    for r in results_a:
        if r.query_id in b_by_id:
            paired_a.append(r)
            paired_b.append(b_by_id[r.query_id])

    comparison = {"name_a": name_a, "name_b": name_b, "n_queries": len(paired_a)}
    for metric in ["correctness", "faithfulness", "completeness", "overall"]:
        a_scores = [getattr(r, metric) if metric != "overall" else r.overall
                    for r in paired_a]
        b_scores = [getattr(r, metric) if metric != "overall" else r.overall
                    for r in paired_b]
        ci = bootstrap_ci(a_scores, b_scores)
        p = permutation_test(a_scores, b_scores)
        comparison[metric] = {
            "mean_a": float(np.mean(a_scores)),
            "mean_b": float(np.mean(b_scores)),
            **ci,
            "p_value": p,
        }
    return comparison


def print_comparison(cmp: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {cmp['name_a']}  vs  {cmp['name_b']}   (n={cmp['n_queries']})")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'A':>8} {'B':>8} {'Diff':>8} {'95% CI':>20} {'p':>8}")
    print(f"{'-'*70}")
    for metric in ["correctness", "faithfulness", "completeness", "overall"]:
        m = cmp[metric]
        ci_str = f"[{m['ci_lower']:+.3f}, {m['ci_upper']:+.3f}]"
        sig = "*" if m["significant"] else " "
        print(
            f"{metric:<15} {m['mean_a']:>8.3f} {m['mean_b']:>8.3f} "
            f"{m['mean_diff']:>+8.3f} {ci_str:>20} {m['p_value']:>7.4f}{sig}"
        )
    print("  * = 95% CI excludes 0 (statistically significant)")


def load_queries(path: Path) -> list[EvalQuery]:
    with open(path) as f:
        data = json.load(f)
    return [EvalQuery(**q) for q in data]


def save_results(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    from agent import build_agent, run_agent, run_naive_rag

    queries_path = Path(__file__).parent.parent / "eval" / "queries.json"
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} eval queries")

    judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # System 1: Naive RAG
    results_naive = run_evaluation(queries, run_naive_rag, "naive_rag", judge_client)
    save_results(
        results_naive,
        Path(__file__).parent.parent / "eval" / "results" / "naive_rag.json",
    )

    # System 2: Agentic RAG
    agent = build_agent()
    results_agent = run_evaluation(
        queries, lambda q: run_agent(q, agent), "agentic_rag", judge_client
    )
    save_results(
        results_agent,
        Path(__file__).parent.parent / "eval" / "results" / "agentic_rag.json",
    )

    # Compare
    comparison = compare_systems(results_agent, results_naive, "Agentic", "Naive")
    print_comparison(comparison)
    with open(
        Path(__file__).parent.parent / "eval" / "results" / "comparison.json", "w"
    ) as f:
        json.dump(comparison, f, indent=2)
