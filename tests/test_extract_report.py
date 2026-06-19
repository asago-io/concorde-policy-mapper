from asago_policy_mapper.extract.report import build_risk_extraction_report


def test_build_risk_extraction_report_creates_html(tmp_path):
    data = {
        "version": "0.3",
        "risks": [
            {
                "risk_id": "R-001",
                "risk_name": "Model Bias",
                "risk_description": "Systematic errors",
                "confidence": 0.95,
                "grounding_confidence": "high",
                "accepted_by": "threshold",
                "evidence": [
                    {
                        "text": "bias in outputs",
                        "document": "policy.pdf",
                        "chunk_index": 0,
                        "sentence_index": 0,
                        "cross_encoder_score": 0.0,
                    }
                ],
                "scores": {"bm25_rank": 3, "embedding_distance": 0.2, "cross_encoder_score": 0.95, "rrf_score": 0.05},
            }
        ],
        "source_documents": ["policy.pdf"],
        "retrieval_stats": {
            "total_chunks": 5,
            "total_candidates_retrieved": 100,
            "auto_accepted": 3,
            "llm_judged": 1,
            "grounding_filtered": 0,
            "timing_ms": {"parse_ms": 100, "chunk_ms": 50},
        },
        "metadata": {"model": "test-model", "threshold_high": 0.7, "threshold_low": 0.15},
        "token_usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600, "calls": 2},
        "chunks": [
            {
                "index": 0,
                "source": "policy.pdf",
                "text_preview": "AI systems must avoid bias...",
                "candidates_retrieved": 20,
                "auto_accepted": 1,
                "borderline": 3,
                "discarded": 16,
                "accepted_risk_ids": ["R-001"],
            }
        ],
        "llm_calls": [
            {
                "call_id": "ground-001",
                "stage": "grounding",
                "chunk_index": 0,
                "risk_ids": ["R-001"],
                "messages": [{"role": "user", "content": "Ground these risks"}],
                "response": [{"risk_id": "R-001", "grounded": True}],
                "duration_ms": 150,
                "result_summary": "1/1 grounded",
            }
        ],
        "eval": None,
    }

    output = tmp_path / "risk-extraction.html"
    result_path = build_risk_extraction_report(data, output)

    assert result_path == output
    assert output.exists()
    html = output.read_text()
    assert "Model Bias" in html
    assert "__REPORT_DATA__" not in html


def test_build_risk_extraction_report_with_eval(tmp_path):
    data = {
        "version": "0.3",
        "risks": [],
        "source_documents": ["policy.pdf"],
        "retrieval_stats": {
            "total_chunks": 0,
            "total_candidates_retrieved": 0,
            "auto_accepted": 0,
            "llm_judged": 0,
            "grounding_filtered": 0,
        },
        "metadata": {},
        "token_usage": {},
        "chunks": [],
        "llm_calls": [],
        "eval": {
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "pass": True,
            "matched": 15,
            "missing": ["R-X"],
            "spurious": [],
            "matched_ids": [],
            "total_expected": 16,
            "total_extracted": 15,
        },
    }

    output = tmp_path / "report.html"
    build_risk_extraction_report(data, output)
    assert output.exists()
