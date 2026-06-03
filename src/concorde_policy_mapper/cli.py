import json
from pathlib import Path

import typer

from concorde_policy_mapper.llm import LLMConfig, TokenTracker, create_client
from concorde_policy_mapper import debug

app = typer.Typer()


@app.callback()
def main():
    """Concorde Policy Mapper — policy risk extraction using AI Atlas Nexus."""


EXCLUDED_TAXONOMIES = {
    "mit-ai-risk-repository-causal",
    "ibm-granite-guardian",
    "nist-ai-rmf",
    "owasp-llm-2.0",
    "ailuminate-v1.0",
    "owasp-asi",
    "shieldgemma-taxonomy",
}


@app.command()
def extract(
    policy_files: list[Path] = typer.Argument(..., help="Policy document(s) to extract risks from"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    base_url: str = typer.Option(None, "--base-url", envvar="POLICY_MAPPER_BASE_URL", help="LLM API base URL"),
    model: str = typer.Option(None, "--model", envvar="POLICY_MAPPER_MODEL", help="LLM model name"),
    api_key: str = typer.Option("none", "--api-key", envvar="POLICY_MAPPER_API_KEY", help="LLM API key"),
    nexus_base_dir: str = typer.Option(None, "--nexus-base-dir", envvar="NEXUS_BASE_DIR", help="Path to ai-atlas-nexus repo"),
    debug_dir: Path = typer.Option(None, "--debug", help="Directory for per-call debug logs"),
    max_concurrent: int = typer.Option(32, "--max-concurrent", help="Max parallel LLM calls"),
    ocr: bool = typer.Option(False, "--ocr", help="Enable OCR for document conversion"),
    chunk_max_tokens: int = typer.Option(512, "--chunk-max-tokens", help="Max tokens per chunk (default: 512)"),
    top_n_accept: int = typer.Option(10, "--top-n-accept", help="Auto-accept top N candidates per chunk (rank-based)"),
    top_n_judge: int = typer.Option(10, "--top-n-judge", help="Send next N candidates to LLM judge (rank-based)"),
    min_score_floor: float = typer.Option(0.70, "--min-score-floor", help="Reject candidates below this score regardless of rank"),
    threshold_high: float = typer.Option(None, "--threshold-high", help="Legacy: absolute auto-accept threshold (overrides rank-based)"),
    threshold_low: float = typer.Option(None, "--threshold-low", help="Legacy: absolute discard threshold (overrides rank-based)"),
    bi_encoder_model: str = typer.Option("all-mpnet-base-v2", "--bi-encoder-model", help="Bi-encoder model"),
    query_instruction: str = typer.Option("", "--query-instruction", help="Instruction prefix for query encoding (e.g. for Qwen3-Embedding)"),
    cross_encoder_model: str = typer.Option("cross-encoder/ms-marco-MiniLM-L-12-v2", "--cross-encoder-model", help="Cross-encoder model"),
    bm25_rescue_rank: int = typer.Option(0, "--bm25-rescue-rank", help="BM25 rank cutoff for rescuing candidates past cross-encoder (0=disabled)"),
    no_cross_encoder: bool = typer.Option(False, "--no-cross-encoder", help="Skip cross-encoder reranking and LLM judge; use RRF score floor instead"),
    rrf_min_score: float = typer.Option(0.015, "--rrf-min-score", help="Minimum RRF score for candidates (only used with --no-cross-encoder)"),
    colbert_model: str = typer.Option(None, "--colbert-model", help="ColBERT model for late interaction retrieval (replaces bi-encoder + cross-encoder)"),
    judge_prompt: str = typer.Option("judge_risk", "--judge-prompt", help="Judge prompt template name (judge_risk, judge_risk_gepa, judge_risk_gepa_demos)"),
    judge_context_tokens: int = typer.Option(0, "--judge-context-tokens", help="Max tokens for judge context window (0=use default sentence padding)"),
    expand_siblings: bool = typer.Option(False, "--expand-siblings", help="After merge, expand to sibling risks and ground them against relevant chunks"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM judge; auto-promote borderline candidates to accepted"),
    no_grounding: bool = typer.Option(False, "--no-grounding", help="Skip LLM grounding; accepted candidates become matches without evidence"),
):
    """Extract risks from policy documents using hybrid retrieval."""
    for pf in policy_files:
        if not pf.exists():
            typer.echo(f"Error: {pf} does not exist", err=True)
            raise typer.Exit(1)

    needs_llm = not (no_judge and no_grounding) or (not no_judge and use_cross_encoder)
    if needs_llm and (not base_url or not model):
        typer.echo("Error: --base-url and --model are required (unless both --no-judge and --no-grounding are set)", err=True)
        raise typer.Exit(1)

    if not nexus_base_dir:
        typer.echo("Error: --nexus-base-dir is required", err=True)
        raise typer.Exit(1)

    if not needs_llm:
        config = LLMConfig(base_url=base_url or "unused", model=model or "unused", api_key=api_key, max_concurrent=max_concurrent)
        tracker = TokenTracker()
        client = None
    else:
        config = LLMConfig(base_url=base_url, model=model, api_key=api_key, max_concurrent=max_concurrent)
        tracker = TokenTracker()
        client = create_client(config, tracker=tracker)
    debug.configure(debug_dir)

    output.mkdir(parents=True, exist_ok=True)

    from ai_atlas_nexus import AIAtlasNexus

    nexus = AIAtlasNexus(base_dir=nexus_base_dir)
    all_risks = [
        r for r in nexus.get_all_risks()
        if getattr(r, "isDefinedByTaxonomy", "") not in EXCLUDED_TAXONOMIES
    ]

    from concorde_policy_mapper.extract.pipeline import run_extraction

    typer.echo(f"Extracting risks from {len(policy_files)} document(s) ({len(all_risks)} Nexus risks loaded)...")

    result = run_extraction(
        documents=policy_files,
        client=client,
        config=config,
        risks=all_risks,
        ocr=ocr,
        chunk_max_tokens=chunk_max_tokens,
        top_n_accept=top_n_accept,
        top_n_judge=top_n_judge,
        min_score_floor=min_score_floor,
        bi_encoder_model=bi_encoder_model,
        query_instruction=query_instruction,
        cross_encoder_model=cross_encoder_model,
        bm25_rescue_rank=bm25_rescue_rank,
        use_cross_encoder=not no_cross_encoder,
        rrf_min_score=rrf_min_score,
        colbert_model=colbert_model or None,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        expand_siblings=expand_siblings,
        no_judge=no_judge,
        no_grounding=no_grounding,
        judge_prompt=judge_prompt,
        judge_context_tokens=judge_context_tokens,
    )

    result.token_usage = tracker.to_dict()

    from concorde_policy_mapper.extract.mitigations import (
        enrich_with_mitigations,
        load_mitigation_index,
    )
    mitigation_index = load_mitigation_index()
    if mitigation_index:
        enrich_with_mitigations(result.risks, mitigation_index)
        typer.echo(f"  Mitigations attached from {len(mitigation_index)} risk entries")

    result_data = result.model_dump()
    result_path = output / "risk-extraction.json"
    result_path.write_text(json.dumps(result_data, indent=2))
    typer.echo(f"Risk extraction written to {result_path}")
    typer.echo(f"  {len(result.risks)} risks matched")
    stats = result.retrieval_stats
    if no_judge and no_grounding:
        typer.echo(f"  {stats.auto_accepted} auto-accepted (IR-only, no LLM stages)")
    elif no_grounding:
        typer.echo(f"  {stats.auto_accepted} auto-accepted, {stats.llm_judged} LLM-judged (no grounding)")
    elif no_cross_encoder:
        typer.echo(f"  {stats.auto_accepted} RRF-accepted, {stats.grounding_filtered} grounding-filtered (no cross-encoder)")
    else:
        typer.echo(f"  {stats.auto_accepted} auto-accepted, {stats.llm_judged} LLM-judged, {stats.grounding_filtered} grounding-filtered")
    if needs_llm:
        typer.echo(f"Token usage: {tracker.prompt_tokens:,} prompt + {tracker.completion_tokens:,} completion = {tracker.total_tokens:,} total ({tracker.calls} calls)")

    from concorde_policy_mapper.extract.report import build_risk_extraction_report
    report_path = build_risk_extraction_report(result_data, output / "risk-extraction.html")
    typer.echo(f"Report written to {report_path}")


@app.command(name="eval")
def eval_cmd(
    run_dir: Path = typer.Argument(..., help="Directory containing risk-extraction.json"),
    ground_truth: Path = typer.Option(None, "--ground-truth", "-g", help="Ground truth YAML file (default: evals/ground_truth/{name}.yaml)"),
    min_recall: float = typer.Option(0.80, "--min-recall", help="Minimum recall threshold"),
    min_precision: float = typer.Option(0.60, "--min-precision", help="Minimum precision threshold"),
):
    """Evaluate a risk extraction run against ground truth."""
    extracted_path = run_dir / "risk-extraction.json"
    if not extracted_path.exists():
        typer.echo(f"Error: {extracted_path} not found", err=True)
        raise typer.Exit(1)

    if ground_truth is None:
        evals_dir = Path(__file__).parent.parent.parent / "evals" / "ground_truth"
        ground_truth = evals_dir / f"{run_dir.name}.yaml"

    if not ground_truth.exists():
        typer.echo(f"Error: ground truth not found at {ground_truth}", err=True)
        raise typer.Exit(1)

    from concorde_policy_mapper.evals.eval import evaluate_extraction

    result = evaluate_extraction(
        ground_truth, extracted_path,
        policy_name=run_dir.name,
        min_recall=min_recall,
        min_precision=min_precision,
    )

    eval_path = run_dir / "eval.json"
    eval_path.write_text(json.dumps(result, indent=2))

    extraction_data = json.loads(extracted_path.read_text())
    extraction_data["eval"] = result
    extracted_path.write_text(json.dumps(extraction_data, indent=2))

    from concorde_policy_mapper.extract.report import build_risk_extraction_report
    report_path = build_risk_extraction_report(extraction_data, run_dir / "risk-extraction.html")

    status = "PASS" if result["pass"] else "FAIL"
    typer.echo(f"Eval: {result['policy']} — {status}")
    typer.echo(f"  Precision: {result['precision']:.3f} (threshold: {min_precision})")
    typer.echo(f"  Recall:    {result['recall']:.3f} (threshold: {min_recall})")
    typer.echo(f"  F1:        {result['f1']:.3f}")
    typer.echo(f"  Matched:   {result['matched']}/{result['total_expected']} expected, {result['total_extracted']} extracted")
    if result["missing"]:
        typer.echo(f"  Missing:   {', '.join(result['missing'])}")
    if result["spurious"]:
        typer.echo(f"  Spurious:  {', '.join(result['spurious'])}")
    typer.echo(f"  Written to {eval_path}")
    typer.echo(f"  Report updated at {report_path}")
