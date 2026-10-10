"""Hybrid Search Reciprocal Rank Fusion (RRF) & Relevance Grader Parser for Agentic RAG."""
import json
import re
from typing import List, Dict, Any

JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)

def parse_relevance_grade(raw_response: str) -> str:
    """Safely parse single-line or multi-line JSON relevance scores ('yes'/'no') from LLM grader output."""
    if not raw_response:
        return "no"
    match = JSON_BLOCK_RE.search(raw_response)
    candidate = match.group(0) if match else raw_response.strip()
    try:
        parsed = json.loads(candidate)
        score = str(parsed.get("score", "no")).strip().lower()
        return "yes" if score == "yes" else "no"
    except (json.JSONDecodeError, AttributeError):
        return "yes" if '"yes"' in raw_response.lower() else "no"

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """Fuse dense vector results and sparse BM25 results using Reciprocal Rank Fusion (RRF)."""
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank_idx, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_idx + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"doc_id": doc_id, "rrf_score": round(score, 6)} for doc_id, score in sorted_docs]
