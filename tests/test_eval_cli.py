import json

from typer.testing import CliRunner

from concorde_policy_mapper.cli import app

runner = CliRunner()

SAMPLE_EXTRACTION = {
    "version": "0.2",
    "risks": [
        {
            "risk_id": "atlas-bias",
            "risk_name": "Bias",
            "risk_description": "",
            "confidence": 0.9,
            "grounding_confidence": "high",
            "accepted_by": "threshold",
            "evidence": [],
            "scores": {"bm25_rank": 1, "embedding_distance": 0.1, "cross_encoder_score": 0.9, "rrf_score": 0.5},
        },
        {
            "risk_id": "atlas-privacy",
            "risk_name": "Privacy",
            "risk_description": "",
            "confidence": 0.8,
            "grounding_confidence": "high",
            "accepted_by": "threshold",
            "evidence": [],
            "scores": {"bm25_rank": 2, "embedding_distance": 0.2, "cross_encoder_score": 0.8, "rrf_score": 0.4},
        },
    ],
    "source_documents": ["policy.md"],
    "retrieval_stats": {
        "total_chunks": 10,
        "total_candidates_retrieved": 50,
        "auto_accepted": 2,
        "llm_judged": 0,
        "grounding_filtered": 0,
    },
}


def test_eval_command_with_explicit_ground_truth(tmp_path):
    run_dir = tmp_path / "my-run"
    run_dir.mkdir()
    (run_dir / "risk-extraction.json").write_text(json.dumps(SAMPLE_EXTRACTION))

    gt = tmp_path / "gt.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n  - atlas-privacy\n")

    result = runner.invoke(app, ["eval", str(run_dir), "--ground-truth", str(gt)])
    assert result.exit_code == 0
    assert "precision" in result.stdout.lower()
    assert "recall" in result.stdout.lower()

    eval_json = run_dir / "eval.json"
    assert eval_json.exists()
    data = json.loads(eval_json.read_text())
    assert data["precision"] == 1.0
    assert data["recall"] == 1.0
    assert data["pass"] is True


def test_eval_command_missing_extraction(tmp_path):
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()

    gt = tmp_path / "gt.yaml"
    gt.write_text("risk_ids:\n  - atlas-bias\n")

    result = runner.invoke(app, ["eval", str(run_dir), "--ground-truth", str(gt)])
    assert result.exit_code == 1
