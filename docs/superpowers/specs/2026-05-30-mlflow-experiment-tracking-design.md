# MLflow Experiment Tracking for Evals

## Goal

Add MLflow experiment tracking to the battery runner so battery runs over time can be compared — which model/threshold/prompt combination gives the best precision, recall, and F1 across the policy corpus. Additionally, sync prompt templates to the MLflow Prompt Registry for versioning and lineage tracking.

## Scope

**In scope:**
- MLflow experiment tracking in `run_extract_battery.py`
- Prompt registry sync (git → MLflow) at battery start
- Nested run structure (parent = battery, children = policies)
- Graceful degradation when MLflow is unavailable

**Out of scope:**
- Tracking standalone `asago-policy-mapper eval` CLI invocations (battery runner only)
- MLflow tracing/observability for individual LLM calls
- Loading prompts from MLflow at extraction runtime (prompts stay on disk)

## Design

### Experiment & Run Structure

One MLflow experiment named `risk-extraction` (configurable via `--mlflow-experiment` flag or `MLFLOW_EXPERIMENT_NAME` env var).

#### Parent Run (one per battery execution)

Named `{battery_name}_{timestamp}` to match the output directory.

**Params:**
- `model`
- `bi_encoder_model`
- `cross_encoder_model`
- `threshold_high`, `threshold_low`
- `rrf_min_score`
- `classify_taxonomies`
- `no_cross_encoder`
- `jobs`
- `battery_config` (battery YAML filename)
- `prompt/judge_risk_version`, `prompt/ground_evidence_version`, `prompt/classify_risks_version`

**Metrics:**
- `macro_recall`, `macro_precision`, `macro_f1`
- `evals_passed`, `evals_total`
- `runs_succeeded`, `runs_failed`
- Per-taxonomy aggregates: `{taxonomy}/recall`, `{taxonomy}/precision`, `{taxonomy}/f1`

**Artifacts:**
- `battery-summary.json`
- `battery-summary.html`

#### Child Runs (one per policy)

Named after the policy (e.g., `dhs-gov`).

**Params:**
- `policy_name`
- `source_documents` (comma-separated filenames)

**Metrics:**
- `recall`, `precision`, `f1`
- `total_expected`, `total_extracted`, `matched`
- `risks_count`
- `auto_accepted`, `llm_judged`, `grounding_filtered`
- `elapsed_seconds`
- Per-taxonomy: `{taxonomy}/recall`, `{taxonomy}/precision`, `{taxonomy}/f1`

**Tags:**
- `eval_status`: `PASS` or `FAIL`

**Artifacts:**
- `risk-extraction.json`
- `risk-extraction.html`
- `eval.json`

### Prompt Registry Sync

At the start of each battery run, before launching extractions, the runner syncs prompt templates to the MLflow Prompt Registry.

Three prompts are managed: `judge_risk`, `ground_evidence`, `classify_risks`. Each is a chat-style prompt with system and user message templates.

**Sync logic:**
1. Read `_system.j2` and `_user.j2` from `src/asago_policy_mapper/templates/prompts/`
2. Combine into a chat template (list of `{"role": "system", "content": ...}, {"role": "user", "content": ...}`)
3. Compute SHA-256 hash of the combined content
4. Load the latest version from the registry and compare its `content_hash` tag
5. If different (or prompt doesn't exist yet), register a new version with:
   - `commit_message`: git commit SHA (if available) or timestamp
   - `tags`: `{"content_hash": "<sha256>"}`
6. Record the version numbers as params on the parent run

**Failure handling:** If the registry is unreachable, log a warning and skip prompt sync. Record `prompt/*_version` params as `"untracked"`.

### Configuration

**New dependency:** `mlflow>=2.17` in pyproject.toml.

**Tracking URI:** Via standard `MLFLOW_TRACKING_URI` env var. If unset, MLflow defaults to local `mlruns/` directory.

**New CLI flags on battery runner:**
- `--mlflow-experiment` (default: `risk-extraction`): experiment name
- `--no-mlflow`: disable all MLflow tracking (for offline runs or unreachable server)

### Integration Points in `run_extract_battery.py`

All changes are in `run_extract_battery.py`. The extraction pipeline (`extract/`) and eval module (`evals/eval.py`) are untouched.

**1. Before the run loop** (in `main()`):
```
if mlflow_enabled:
    mlflow.set_experiment(experiment_name)
    parent_run = mlflow.start_run(run_name=f"{battery_name}_{timestamp}")
    mlflow.log_params({model, thresholds, encoders, ...})
    sync_prompts_to_registry()  # logs prompt versions as params
```

**2. After each policy eval** (new function or extension of `run_eval()`):
```
if mlflow_enabled:
    with mlflow.start_run(run_name=policy_name, nested=True):
        mlflow.log_metrics({recall, precision, f1, ...})
        mlflow.log_artifact(risk_extraction_json)
        mlflow.log_artifact(eval_json)
        mlflow.set_tag("eval_status", "PASS" or "FAIL")
```

**3. After the run loop** (in `main()`):
```
if mlflow_enabled:
    mlflow.log_metrics({macro_recall, macro_precision, macro_f1, ...})
    mlflow.log_artifact(battery_summary_json)
    mlflow.log_artifact(battery_summary_html)
    mlflow.end_run()
```

**4. Error handling:**
All MLflow calls are wrapped in try/except. A tracking failure logs a warning but never stops the battery run. A module-level `_mlflow_enabled` flag is checked before every MLflow call.

### Graceful Degradation

- If `--no-mlflow` is passed, all tracking is skipped
- If `mlflow` is not installed, tracking is skipped with a warning
- If `mlflow.start_run()` or any tracking call fails, the failure is logged and the battery continues without tracking
- The battery runner's existing JSON/HTML output is always produced regardless of MLflow state
