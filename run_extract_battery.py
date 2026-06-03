#!/usr/bin/env python3
"""Run the document-to-risk extraction pipeline against a battery config.

Battery configs are YAML files that specify which policy files/directories
to run and with what model. Subdirectories are multi-document groups —
all files in the subdirectory are passed together as one run.

Uses the hybrid retrieval pipeline (BM25 + semantic + cross-encoder)
to extract Nexus risk IDs directly from documents.

Usage:
    python run_extract_battery.py ../batteries/simple.yaml --base-url http://localhost:8000/v1
    python run_extract_battery.py ../batteries/real.yaml --base-url http://localhost:8000/v1 --model override-model
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

from concorde_policy_mapper.evals.eval import evaluate_extraction
from concorde_policy_mapper.tracking import (
    init_tracking,
    end_tracking,
    is_tracking_enabled,
    log_params,
    log_metrics,
    log_artifact,
    log_child_run,
    sync_prompts,
)

PACKAGE_DIR = Path(__file__).parent
ROOT = PACKAGE_DIR.parent
RUNS_DIR = PACKAGE_DIR / "extract-runs"
NEXUS_BASE_DIR = os.environ.get("NEXUS_BASE_DIR", "/Users/hjrnunes/workspace/redhat/ibm/ai-atlas-nexus")
GROUND_TRUTH_DIR = PACKAGE_DIR / "evals" / "ground_truth"

POLICY_EXTENSIONS = {".json", ".md", ".txt", ".pdf", ".docx", ".html", ".htm"}

_print_lock = threading.Lock()


def _locked_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()


def _is_policy_file(p: Path) -> bool:
    return p.is_file() and not p.name.startswith(".") and p.suffix.lower() in POLICY_EXTENSIONS


def _resolve_run(entry: str) -> tuple[str, list[Path]]:
    path = ROOT / entry.rstrip("/")
    if path.is_file():
        return path.stem, [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if _is_policy_file(p))
        if not files:
            _locked_print(f"  Warning: no policy files in {path}, skipping")
            return path.name, []
        return path.name, files
    _locked_print(f"  Warning: {path} does not exist, skipping")
    return Path(entry).stem, []


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _pad_name(name: str, width: int = 20) -> str:
    return name.ljust(width)[:width]


def run_one(
        policies: list[Path],
        name: str,
        base_url: str,
        model: str,
        runs_dir: Path,
        chunk_max_tokens: int = 512,
        top_n_accept: int = 10,
        top_n_judge: int = 10,
        min_score_floor: float = 0.70,
        threshold_high: float | None = None,
        threshold_low: float | None = None,
        bi_encoder_model: str = "all-mpnet-base-v2",
        query_instruction: str = "",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        name_width: int = 20,
        bm25_rescue_rank: int = 0,
        no_cross_encoder: bool = False,
        rrf_min_score: float = 0.015,
        colbert_model: str | None = None,
        expand_siblings: bool = False,
        judge_prompt: str = "judge_risk",
        judge_context_tokens: int = 0,
        no_judge: bool = False,
        no_grounding: bool = False,
) -> tuple[str, bool, str, int, float]:
    out = runs_dir / name
    tag = _pad_name(name, name_width)

    files_desc = ", ".join(p.name for p in policies)
    _locked_print(f"  [{tag}] starting ({files_desc})")

    cmd = [
        "uv", "run", "concorde-policy-mapper", "extract",
        *[str(p) for p in policies],
        "-o", str(out),
        "--base-url", base_url,
        "--model", model,
        "--nexus-base-dir", NEXUS_BASE_DIR,
        "--chunk-max-tokens", str(chunk_max_tokens),
        "--top-n-accept", str(top_n_accept),
        "--top-n-judge", str(top_n_judge),
        "--min-score-floor", str(min_score_floor),
        "--bi-encoder-model", bi_encoder_model,
        "--cross-encoder-model", cross_encoder_model,
    ]
    if query_instruction:
        cmd.extend(["--query-instruction", query_instruction])
    cmd += [
        "--bm25-rescue-rank", str(bm25_rescue_rank),
    ]
    if no_cross_encoder:
        cmd.extend(["--no-cross-encoder", "--rrf-min-score", str(rrf_min_score)])
    if colbert_model:
        cmd.extend(["--colbert-model", colbert_model])
    if expand_siblings:
        cmd.append("--expand-siblings")
    if judge_prompt != "judge_risk":
        cmd.extend(["--judge-prompt", judge_prompt])
    if judge_context_tokens > 0:
        cmd.extend(["--judge-context-tokens", str(judge_context_tokens)])
    if no_judge:
        cmd.append("--no-judge")
    if no_grounding:
        cmd.append("--no-grounding")
    if threshold_high is not None:
        cmd.extend(["--threshold-high", str(threshold_high)])
    if threshold_low is not None:
        cmd.extend(["--threshold-low", str(threshold_low)])

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        captured_lines.append(line)
        _locked_print(f"  [{tag}] {line}")
    proc.wait()

    elapsed = time.monotonic() - t0
    ok = proc.returncode == 0

    if ok:
        _locked_print(f"  [{tag}] done ({_fmt_elapsed(elapsed)})")
    else:
        _locked_print(f"  [{tag}] FAILED (exit {proc.returncode}, {_fmt_elapsed(elapsed)})")
        error_path = out / "error.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_data = {
            "error": True,
            "name": name,
            "exit_code": proc.returncode,
            "stderr_tail": captured_lines[-20:],
            "timestamp": datetime.now().isoformat(),
        }
        with open(error_path, "w") as f:
            json.dump(error_data, f, indent=2)

    return name, ok, "", proc.returncode, elapsed


def run_eval(name: str, runs_dir: Path, min_recall: float = 0.80, min_precision: float = 0.60) -> dict | None:
    gt_path = GROUND_TRUTH_DIR / f"{name}.yaml"
    if not gt_path.exists():
        return None
    extracted_path = runs_dir / name / "risk-extraction.json"
    if not extracted_path.exists():
        return None
    result = evaluate_extraction(gt_path, extracted_path, policy_name=name, min_recall=min_recall, min_precision=min_precision)
    eval_path = runs_dir / name / "eval.json"
    eval_path.write_text(json.dumps(result, indent=2))

    extraction_data = json.loads(extracted_path.read_text())
    extraction_data["eval"] = result
    extracted_path.write_text(json.dumps(extraction_data, indent=2))

    try:
        from concorde_policy_mapper.extract.report import build_risk_extraction_report
        build_risk_extraction_report(extraction_data, runs_dir / name / "risk-extraction.html")
    except Exception as e:
        _locked_print(f"  [{name}] Warning: could not generate HTML report: {e}")

    return result


def _build_battery_report(summary: dict, output_path: Path) -> None:
    eval_results = summary.get("eval_results", {})
    tax_agg = summary.get("taxonomy_aggregate", {})

    policies = sorted(eval_results.keys())
    all_taxonomies = set()
    for ev in eval_results.values():
        all_taxonomies.update(ev.get("per_taxonomy", {}).keys())
    taxonomies = sorted(all_taxonomies)

    def _cell_color(val: float | None) -> str:
        if val is None:
            return "background-color: #e5e7eb;"
        if val <= 0.5:
            r = 239
            g = int(68 + (val / 0.5) * (163 - 68))
            b = 68
        else:
            r = int(239 - ((val - 0.5) / 0.5) * (239 - 34))
            g = int(163 + ((val - 0.5) / 0.5) * (197 - 163))
            b = int(68 + ((val - 0.5) / 0.5) * (94 - 68))
        return f"background-color: rgb({r},{g},{b}); color: {'#fff' if val < 0.4 else '#000'};"

    def _heatmap_table(metric: str) -> str:
        rows = []
        for policy in policies:
            ev = eval_results[policy]
            per_tax = ev.get("per_taxonomy", {})
            cells = []
            for tax in taxonomies:
                td = per_tax.get(tax)
                if td and td["expected"] > 0:
                    val = td[metric]
                    style = _cell_color(val)
                    cells.append(f'<td style="{style} text-align:center; padding:4px 8px; font-size:13px;">{val:.2f}</td>')
                else:
                    cells.append(f'<td style="{_cell_color(None)} text-align:center; padding:4px 8px; font-size:13px;">—</td>')
            agg_val = ev.get(metric, 0)
            style = _cell_color(agg_val)
            rows.append(
                f'<tr><td style="padding:4px 8px; font-weight:500; white-space:nowrap;">{policy}</td>'
                + "".join(cells)
                + f'<td style="{style} text-align:center; padding:4px 8px; font-weight:600;">{agg_val:.3f}</td></tr>'
            )
        tax_headers = "".join(
            f'<th style="padding:4px 6px; text-align:center; font-size:12px; writing-mode:vertical-rl; transform:rotate(180deg); max-width:30px; white-space:nowrap;">{t}</th>'
            for t in taxonomies
        )
        return f"""
        <table style="border-collapse:collapse; margin:16px 0; font-family:system-ui,-apple-system,sans-serif; font-size:13px;">
          <thead><tr>
            <th style="padding:4px 8px; text-align:left;">Policy</th>
            {tax_headers}
            <th style="padding:4px 8px; text-align:center;">Overall</th>
          </tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>"""

    n_pass = sum(1 for ev in eval_results.values() if ev["pass"])
    macro_r = sum(ev["recall"] for ev in eval_results.values()) / len(eval_results) if eval_results else 0
    macro_p = sum(ev["precision"] for ev in eval_results.values()) / len(eval_results) if eval_results else 0
    macro_f = sum(ev["f1"] for ev in eval_results.values()) / len(eval_results) if eval_results else 0

    tax_rows = ""
    for tax in sorted(tax_agg):
        a = tax_agg[tax]
        m, e = a["matched"], a["expected"]
        x = a.get("extracted", m)
        spur = x - m
        p = m / (m + spur) if m + spur > 0 else 0.0
        r = m / e if e > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        tax_rows += f"<tr><td style='padding:4px 8px;'>{tax}</td><td style='text-align:right; padding:4px 8px;'>{e}</td><td style='text-align:right; padding:4px 8px;'>{m}</td><td style='text-align:right; padding:4px 8px;'>{p:.3f}</td><td style='text-align:right; padding:4px 8px;'>{r:.3f}</td><td style='text-align:right; padding:4px 8px;'>{f:.3f}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Battery Report — {summary.get('battery', '')} ({summary.get('timestamp', '')})</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #f9fafb; color: #111; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; margin-top: 32px; border-bottom: 1px solid #d1d5db; padding-bottom: 4px; }}
  .summary {{ display: flex; gap: 24px; margin: 12px 0; }}
  .stat {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px 20px; }}
  .stat .label {{ font-size: 12px; color: #6b7280; }}
  .stat .value {{ font-size: 22px; font-weight: 600; }}
  table {{ border-collapse: collapse; }}
  th {{ background: #f3f4f6; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 2px solid #d1d5db; }}
  td {{ border-bottom: 1px solid #e5e7eb; }}
</style>
</head>
<body>
<h1>Battery Report: {summary.get('battery', '')}</h1>
<p style="color:#6b7280; font-size:13px;">Model: {summary.get('model', '')} &middot; {summary.get('timestamp', '')}</p>

<div class="summary">
  <div class="stat"><div class="label">Evals</div><div class="value">{n_pass}/{len(eval_results)} pass</div></div>
  <div class="stat"><div class="label">Macro Recall</div><div class="value">{macro_r:.3f}</div></div>
  <div class="stat"><div class="label">Macro Precision</div><div class="value">{macro_p:.3f}</div></div>
  <div class="stat"><div class="label">Macro F1</div><div class="value">{macro_f:.3f}</div></div>
</div>

<h2>Per-Taxonomy Aggregate</h2>
<table style="font-size:13px;">
  <thead><tr><th style="padding:4px 8px; text-align:left;">Taxonomy</th><th style="padding:4px 8px; text-align:right;">Expected</th><th style="padding:4px 8px; text-align:right;">Matched</th><th style="padding:4px 8px; text-align:right;">Precision</th><th style="padding:4px 8px; text-align:right;">Recall</th><th style="padding:4px 8px; text-align:right;">F1</th></tr></thead>
  <tbody>{tax_rows}</tbody>
</table>

<h2>F1 Heatmap — Policy × Taxonomy</h2>
{_heatmap_table("f1")}

<h2>Recall Heatmap — Policy × Taxonomy</h2>
{_heatmap_table("recall")}

<h2>Precision Heatmap — Policy × Taxonomy</h2>
{_heatmap_table("precision")}

</body>
</html>"""

    output_path.write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("battery", type=Path, help="Battery config YAML file (e.g. ../batteries/simple.yaml)")
    parser.add_argument("--base-url", default=os.environ.get("POLICY_MAPPER_BASE_URL"), help="LLM API base URL (default: $POLICY_MAPPER_BASE_URL)")
    parser.add_argument("--model", default=None, help="Override model from battery config (default: $POLICY_MAPPER_MODEL)")
    parser.add_argument("-j", "--jobs", type=int, default=6, help="Max parallel jobs (default: 6)")
    parser.add_argument("--chunk-max-tokens", type=int, default=512, help="Max tokens per chunk (default: 512)")
    parser.add_argument("--top-n-accept", type=int, default=10, help="Auto-accept top N candidates per chunk (default: 10)")
    parser.add_argument("--top-n-judge", type=int, default=10, help="Send next N candidates to LLM judge (default: 10)")
    parser.add_argument("--min-score-floor", type=float, default=0.70, help="Reject candidates below this score (default: 0.70)")
    parser.add_argument("--threshold-high", type=float, default=None, help="Legacy: absolute auto-accept threshold (overrides rank-based)")
    parser.add_argument("--threshold-low", type=float, default=None, help="Legacy: absolute discard threshold")
    parser.add_argument("--bi-encoder-model", default="all-mpnet-base-v2", help="Bi-encoder model (default: all-mpnet-base-v2)")
    parser.add_argument("--query-instruction", default="", help="Instruction prefix for query encoding (e.g. for Qwen3-Embedding)")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Cross-encoder model (default: cross-encoder/ms-marco-MiniLM-L-12-v2)")
    parser.add_argument("--bm25-rescue-rank", type=int, default=0, help="BM25 rank cutoff for rescuing candidates past cross-encoder (0=disabled, default: 0)")
    parser.add_argument("--no-cross-encoder", action="store_true", help="Skip cross-encoder reranking and LLM judge; use RRF score floor instead")
    parser.add_argument("--rrf-min-score", type=float, default=0.015, help="Minimum RRF score for candidates (only used with --no-cross-encoder)")
    parser.add_argument("--colbert-model", default=None, help="ColBERT model for late interaction retrieval (replaces bi-encoder + cross-encoder)")
    parser.add_argument("--expand-siblings", action="store_true", help="Expand to sibling risks after merge and ground against relevant chunks")
    parser.add_argument("--judge-prompt", default="judge_risk", help="Judge prompt template name (default: judge_risk)")
    parser.add_argument("--judge-context-tokens", type=int, default=0, help="Max tokens for judge context window (0=default sentence padding)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge; auto-promote borderline candidates")
    parser.add_argument("--no-grounding", action="store_true", help="Skip LLM grounding; accepted candidates become matches without evidence")
    parser.add_argument("--mlflow-experiment", default="risk-extraction", help="MLflow experiment name (default: risk-extraction)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    args = parser.parse_args()

    battery_path = args.battery if args.battery.is_absolute() else PACKAGE_DIR / args.battery
    if not battery_path.exists():
        print(f"Battery config not found: {battery_path}")
        sys.exit(1)

    config = yaml.safe_load(battery_path.read_text())
    if args.no_judge and args.no_grounding:
        base_url = args.base_url or "unused"
        model = args.model or config.get("model") or os.environ.get("POLICY_MAPPER_MODEL") or "unused"
    else:
        if not args.base_url:
            print("Error: --base-url is required (or set POLICY_MAPPER_BASE_URL)")
            sys.exit(1)
        base_url = args.base_url
        model = args.model or config.get("model") or os.environ.get("POLICY_MAPPER_MODEL")
        if not model:
            print("Error: model not specified (set in battery config, pass --model, or set POLICY_MAPPER_MODEL)")
            sys.exit(1)

    battery_name = battery_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = RUNS_DIR / f"{battery_name}_{timestamp}"
    runs_dir.mkdir(parents=True)

    runs: list[tuple[str, list[Path]]] = []
    for entry in config.get("runs", []):
        name, files = _resolve_run(entry)
        if files:
            runs.append((name, files))

    if not runs:
        print("No runs resolved from battery config")
        sys.exit(1)

    n_groups = sum(1 for _, files in runs if len(files) > 1)
    n_single = len(runs) - n_groups
    name_width = max(len(name) for name, _ in runs)
    print(f"Extract Battery: {battery_name}")
    print(f"  model:          {model}")
    print(f"  bi_encoder:     {args.bi_encoder_model}")
    print(f"  chunk_tokens:   {args.chunk_max_tokens}")
    print(f"  cross_encoder:  {'DISABLED' if args.no_cross_encoder else args.cross_encoder_model}")
    if args.no_judge:
        print(f"  no_judge:       True (borderline auto-promoted)")
    if args.no_grounding:
        print(f"  no_grounding:   True (skip evidence grounding)")
    if args.no_cross_encoder:
        print(f"  rrf_min_score:  {args.rrf_min_score}")
    if args.threshold_high is not None:
        print(f"  threshold_high: {args.threshold_high} (legacy mode)")
        print(f"  threshold_low:  {args.threshold_low}")
    else:
        print(f"  top_n_accept:   {args.top_n_accept}")
        print(f"  top_n_judge:    {args.top_n_judge}")
        print(f"  min_score_floor:{args.min_score_floor}")
    print(f"  runs:           {len(runs)} ({n_single} single-doc, {n_groups} multi-doc)")
    print(f"  jobs:           {args.jobs}")
    print(f"  output:         {runs_dir}")
    print()

    # --- MLflow tracking ---
    tracking_ctx = init_tracking(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=f"{battery_name}_{timestamp}",
    )
    if is_tracking_enabled(tracking_ctx):
        log_params(tracking_ctx, {
            "model": model,
            "bi_encoder_model": args.bi_encoder_model,
            "cross_encoder_model": args.cross_encoder_model,
            "top_n_accept": str(args.top_n_accept),
            "top_n_judge": str(args.top_n_judge),
            "min_score_floor": str(args.min_score_floor),
            "threshold_high": str(args.threshold_high),
            "threshold_low": str(args.threshold_low),
            "rrf_min_score": str(args.rrf_min_score),
            "no_cross_encoder": str(args.no_cross_encoder),
            "chunk_max_tokens": str(args.chunk_max_tokens),
            "no_judge": str(args.no_judge),
            "no_grounding": str(args.no_grounding),
            "jobs": str(args.jobs),
            "battery_config": battery_path.name,
        })
        templates_dir = PACKAGE_DIR / "src" / "concorde_policy_mapper" / "templates"
        prompt_versions = sync_prompts(tracking_ctx, templates_dir)
        for pname, pversion in prompt_versions.items():
            log_params(tracking_ctx, {f"prompt/{pname}_version": str(pversion)})
        print(f"  mlflow:         {args.mlflow_experiment} (tracking enabled)")
    else:
        if not args.no_mlflow:
            print(f"  mlflow:         disabled (initialization failed)")

    t_battery = time.monotonic()
    failed = []
    timings: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_one,
                policies=files, name=name, base_url=base_url, model=model,
                runs_dir=runs_dir, name_width=name_width,
                chunk_max_tokens=args.chunk_max_tokens,
                top_n_accept=args.top_n_accept, top_n_judge=args.top_n_judge,
                min_score_floor=args.min_score_floor,
                threshold_high=args.threshold_high, threshold_low=args.threshold_low,
                bi_encoder_model=args.bi_encoder_model,
                query_instruction=args.query_instruction,
                cross_encoder_model=args.cross_encoder_model,
                bm25_rescue_rank=args.bm25_rescue_rank,
                no_cross_encoder=args.no_cross_encoder,
                rrf_min_score=args.rrf_min_score,
                colbert_model=args.colbert_model,
                expand_siblings=args.expand_siblings,
                judge_prompt=args.judge_prompt,
                judge_context_tokens=args.judge_context_tokens,
                no_judge=args.no_judge, no_grounding=args.no_grounding,
            ): name
            for name, files in runs
        }
        for future in as_completed(futures):
            name, ok, _, exit_code, elapsed = future.result()
            timings[name] = elapsed
            if not ok:
                failed.append(name)

    elapsed_total = time.monotonic() - t_battery

    # --- Eval ---
    eval_results: dict[str, dict] = {}
    for name, _ in runs:
        if name not in failed:
            ev = run_eval(name, runs_dir)
            if ev is not None:
                eval_results[name] = ev

    # Generate HTML reports for runs without eval
    for name, _ in runs:
        if name not in failed and name not in eval_results:
            extraction_path = runs_dir / name / "risk-extraction.json"
            if extraction_path.exists():
                try:
                    from concorde_policy_mapper.extract.report import build_risk_extraction_report
                    data = json.loads(extraction_path.read_text())
                    build_risk_extraction_report(data, runs_dir / name / "risk-extraction.html")
                except Exception as e:
                    _locked_print(f"  [{name}] Warning: could not generate HTML report: {e}")

    # --- MLflow child runs ---
    if is_tracking_enabled(tracking_ctx):
        for name, _ in runs:
            if name in failed:
                continue
            result_path = runs_dir / name / "risk-extraction.json"
            if not result_path.exists():
                continue

            data = json.loads(result_path.read_text())
            ev = eval_results.get(name)
            child_metrics: dict[str, float] = {}
            child_tags: dict[str, str] = {}
            child_artifacts: list[Path] = [result_path]

            n_risks = len(data.get("risks", []))
            stats = data.get("retrieval_stats", {})
            child_metrics["risks_count"] = float(n_risks)
            child_metrics["auto_accepted"] = float(stats.get("auto_accepted", 0))
            child_metrics["llm_judged"] = float(stats.get("llm_judged", 0))
            child_metrics["grounding_filtered"] = float(stats.get("grounding_filtered", 0))
            child_metrics["elapsed_seconds"] = timings.get(name, 0.0)

            if ev:
                child_metrics["recall"] = ev["recall"]
                child_metrics["precision"] = ev["precision"]
                child_metrics["f1"] = ev["f1"]
                child_metrics["total_expected"] = float(ev["total_expected"])
                child_metrics["total_extracted"] = float(ev["total_extracted"])
                child_metrics["matched"] = float(ev["matched"])
                child_tags["eval_status"] = "PASS" if ev["pass"] else "FAIL"
                for tax, td in ev.get("per_taxonomy", {}).items():
                    child_metrics[f"{tax}/recall"] = td["recall"]
                    child_metrics[f"{tax}/precision"] = td["precision"]
                    child_metrics[f"{tax}/f1"] = td["f1"]
                eval_path = runs_dir / name / "eval.json"
                if eval_path.exists():
                    child_artifacts.append(eval_path)

            html_path = runs_dir / name / "risk-extraction.html"
            if html_path.exists():
                child_artifacts.append(html_path)

            log_child_run(
                tracking_ctx,
                name=name,
                params={"policy_name": name},
                metrics=child_metrics,
                tags=child_tags,
                artifacts=child_artifacts,
            )

    # --- Summary ---
    print(f"\n{'═' * 60}")
    print(f"Extract battery complete: {len(runs) - len(failed)}/{len(runs)} succeeded in {_fmt_elapsed(elapsed_total)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    succeeded = [name for name, _ in runs if name not in failed]
    if succeeded:
        has_evals = bool(eval_results)
        header = f"{'Name':<{name_width}}  {'Risks':>6}  {'Auto':>5}  {'Judge':>6}  {'Filter':>7}  {'Time':>8}"
        separator = f"{'─' * name_width}  {'─' * 6}  {'─' * 5}  {'─' * 6}  {'─' * 7}  {'─' * 8}"
        if has_evals:
            header += f"  {'Recall':>7}  {'Prec':>7}  {'Eval':>6}"
            separator += f"  {'─' * 7}  {'─' * 7}  {'─' * 6}"
        print(f"\n{header}")
        print(separator)
        for name in succeeded:
            result_path = runs_dir / name / "risk-extraction.json"
            if result_path.exists():
                data = json.loads(result_path.read_text())
                n_risks = len(data.get("risks", []))
                stats = data.get("retrieval_stats", {})
                auto = stats.get("auto_accepted", 0)
                judged = stats.get("llm_judged", 0)
                filtered = stats.get("grounding_filtered", 0)
                elapsed = timings.get(name, 0)
                row = f"{name:<{name_width}}  {n_risks:>6}  {auto:>5}  {judged:>6}  {filtered:>7}  {_fmt_elapsed(elapsed):>8}"
                if has_evals:
                    ev = eval_results.get(name)
                    if ev:
                        status = "PASS" if ev["pass"] else "FAIL"
                        row += f"  {ev['recall']:>7.3f}  {ev['precision']:>7.3f}  {status:>6}"
                    else:
                        row += f"  {'—':>7}  {'—':>7}  {'—':>6}"
                print(row)

    if eval_results:
        n_pass = sum(1 for ev in eval_results.values() if ev["pass"])
        print(f"\nEvals: {n_pass}/{len(eval_results)} passed")

        # Per-taxonomy aggregate
        tax_agg: dict[str, dict[str, int]] = {}
        for ev in eval_results.values():
            for tax, td in ev.get("per_taxonomy", {}).items():
                agg = tax_agg.setdefault(tax, {"expected": 0, "extracted": 0, "matched": 0})
                agg["expected"] += td["expected"]
                agg["extracted"] += td["extracted"]
                agg["matched"] += td["matched"]

        if tax_agg:
            tw = max(len(t) for t in tax_agg)
            tw = max(tw, 10)
            print(f"\n{'Taxonomy':<{tw}}  {'Expect':>6}  {'Match':>5}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
            print(f"{'─' * tw}  {'─' * 6}  {'─' * 5}  {'─' * 7}  {'─' * 7}  {'─' * 7}")
            for tax in sorted(tax_agg):
                a = tax_agg[tax]
                m, e, x = a["matched"], a["expected"], a["extracted"]
                spur = x - m
                p = m / (m + spur) if m + spur > 0 else 0.0
                r = m / e if e > 0 else 0.0
                f = 2 * p * r / (p + r) if p + r > 0 else 0.0
                print(f"{tax:<{tw}}  {e:>6}  {m:>5}  {p:>7.3f}  {r:>7.3f}  {f:>7.3f}")

        battery_summary = {
            "battery": battery_name,
            "model": model,
            "timestamp": timestamp,
            "eval_results": {name: ev for name, ev in eval_results.items()},
            "taxonomy_aggregate": {
                tax: {
                    **a,
                    "precision": round(a["matched"] / (a["matched"] + a["extracted"] - a["matched"]) if a["matched"] + a["extracted"] - a["matched"] > 0 else 0.0, 3),
                    "recall": round(a["matched"] / a["expected"] if a["expected"] > 0 else 0.0, 3),
                }
                for tax, a in tax_agg.items()
            },
        }
        summary_path = runs_dir / "battery-summary.json"
        summary_path.write_text(json.dumps(battery_summary, indent=2))

        try:
            _build_battery_report(battery_summary, runs_dir / "battery-summary.html")
            print(f"Battery report: {runs_dir / 'battery-summary.html'}")
        except Exception as e:
            _locked_print(f"Warning: could not generate battery HTML report: {e}")

    # --- MLflow parent run summary ---
    if is_tracking_enabled(tracking_ctx):
        parent_metrics: dict[str, float] = {
            "runs_succeeded": float(len(runs) - len(failed)),
            "runs_failed": float(len(failed)),
        }
        if eval_results:
            parent_metrics["evals_passed"] = float(sum(1 for ev in eval_results.values() if ev["pass"]))
            parent_metrics["evals_total"] = float(len(eval_results))
            parent_metrics["macro_recall"] = sum(ev["recall"] for ev in eval_results.values()) / len(eval_results)
            parent_metrics["macro_precision"] = sum(ev["precision"] for ev in eval_results.values()) / len(eval_results)
            parent_metrics["macro_f1"] = sum(ev["f1"] for ev in eval_results.values()) / len(eval_results)
            if tax_agg:
                for tax, a in tax_agg.items():
                    m, e, x = a["matched"], a["expected"], a["extracted"]
                    spur = x - m
                    p = m / (m + spur) if m + spur > 0 else 0.0
                    r = m / e if e > 0 else 0.0
                    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
                    parent_metrics[f"{tax}/recall"] = r
                    parent_metrics[f"{tax}/precision"] = p
                    parent_metrics[f"{tax}/f1"] = f

        log_metrics(tracking_ctx, parent_metrics)

        summary_json = runs_dir / "battery-summary.json"
        summary_html = runs_dir / "battery-summary.html"
        if summary_json.exists():
            log_artifact(tracking_ctx, summary_json)
        if summary_html.exists():
            log_artifact(tracking_ctx, summary_html)

    end_tracking(tracking_ctx)

    print(f"\nOutput: {runs_dir}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
