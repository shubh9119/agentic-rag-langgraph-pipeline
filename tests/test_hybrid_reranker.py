from hybrid_reranker import parse_relevance_grade, reciprocal_rank_fusion

def test_parse_relevance_grade_multiline_json():
    raw = """Here is the evaluation:
    {
      "score": "yes"
    }"""
    assert parse_relevance_grade(raw) == "yes"

def test_parse_relevance_grade_negative():
    assert parse_relevance_grade('{"score": "no"}') == "no"

def test_reciprocal_rank_fusion_orders_consensus_first():
    dense = ["doc_a", "doc_b", "doc_c"]
    sparse = ["doc_b", "doc_a", "doc_d"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_n=3)
    assert fused[0]["doc_id"] in ("doc_a", "doc_b")
    assert len(fused) == 3
