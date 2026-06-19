# MLflow Experiment Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MLflow experiment tracking to the battery runner so battery runs can be compared over time, and sync prompt templates to the MLflow Prompt Registry for versioning.

**Architecture:** All MLflow integration lives in `run_extract_battery.py`. A new helper module `src/asago_policy_mapper/tracking.py` encapsulates MLflow interactions behind a thin wrapper that gracefully degrades when MLflow is unavailable. The extraction pipeline and eval module are untouched.

**Tech Stack:** mlflow>=2.17, existing Python stdlib (hashlib, subprocess for git SHA)

---

### File Map

- **Create:** `src/asago_policy_mapper/tracking.py` — MLflow wrapper (experiment setup, run logging, prompt sync)
- **Create:** `tests/test_tracking.py` — tests for the tracking module
- **Modify:** `run_extract_battery.py` — add CLI flags and call tracking functions
- **Modify:** `pyproject.toml` — add mlflow dependency

---

### Task 1: Add mlflow dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add mlflow to dependencies**

In `pyproject.toml`, add `"mlflow>=2.17"` to the `dependencies` list, after `"numpy"`:

```toml
    "numpy",
    "mlflow>=2.17",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: mlflow and its dependencies install successfully.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import mlflow; print(mlflow.__version__)"`
Expected: prints a version >= 2.17

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add mlflow>=2.17 for experiment tracking"
```

---

### Task 2: Create tracking module — graceful import and experiment setup

**Files:**
- Create: `src/asago_policy_mapper/tracking.py`
- Create: `tests/test_tracking.py`

- [ ] **Step 1: Write failing tests for graceful import and experiment setup**

Create `tests/test_tracking.py`:

```python
from unittest.mock import patch, MagicMock
import pytest

from asago_policy_mapper.tracking import (
    is_tracking_enabled,
    init_tracking,
    end_tracking,
)


def test_tracking_disabled_by_flag():
    ctx = init_tracking(enabled=False, experiment_name="test")
    assert not is_tracking_enabled(ctx)
    end_tracking(ctx)


@patch("asago_policy_mapper.tracking.mlflow")
def test_tracking_enabled_sets_experiment_and_starts_run(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    ctx = init_tracking(enabled=True, experiment_name="test-exp", run_name="run-1")
    assert is_tracking_enabled(ctx)

    mock_mlflow.set_experiment.assert_called_once_with("test-exp")
    mock_mlflow.start_run.assert_called_once_with(run_name="run-1")


@patch("asago_policy_mapper.tracking.mlflow")
def test_tracking_survives_mlflow_failure(mock_mlflow):
    mock_mlflow.set_experiment.side_effect = Exception("connection refused")

    ctx = init_tracking(enabled=True, experiment_name="test-exp", run_name="run-1")
    assert not is_tracking_enabled(ctx)


@patch("asago_policy_mapper.tracking.mlflow")
def test_end_tracking_calls_end_run(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    ctx = init_tracking(enabled=True, experiment_name="test-exp", run_name="run-1")
    end_tracking(ctx)
    mock_mlflow.end_run.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError` because `tracking.py` doesn't exist yet.

- [ ] **Step 3: Implement the tracking module**

Create `src/asago_policy_mapper/tracking.py`:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.genai
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False


@dataclass
class TrackingContext:
    enabled: bool = False
    run_id: str | None = None
    child_run_ids: dict[str, str] = field(default_factory=dict)


def is_tracking_enabled(ctx: TrackingContext) -> bool:
    return ctx.enabled


def init_tracking(
    *,
    enabled: bool = True,
    experiment_name: str = "risk-extraction",
    run_name: str | None = None,
) -> TrackingContext:
    if not enabled or not _MLFLOW_AVAILABLE:
        if enabled and not _MLFLOW_AVAILABLE:
            logger.warning("mlflow not installed — tracking disabled")
        return TrackingContext(enabled=False)

    try:
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        logger.info("MLflow tracking started: experiment=%s run=%s", experiment_name, run.info.run_id)
        return TrackingContext(enabled=True, run_id=run.info.run_id)
    except Exception:
        logger.warning("MLflow tracking failed to initialize — continuing without tracking", exc_info=True)
        return TrackingContext(enabled=False)


def end_tracking(ctx: TrackingContext) -> None:
    if not ctx.enabled:
        return
    try:
        mlflow.end_run()
    except Exception:
        logger.warning("MLflow end_run failed", exc_info=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/asago_policy_mapper/tracking.py tests/test_tracking.py
git commit -m "feat: add tracking module with graceful MLflow init/teardown"
```

---

### Task 3: Add param and metric logging to tracking module

**Files:**
- Modify: `src/asago_policy_mapper/tracking.py`
- Modify: `tests/test_tracking.py`

- [ ] **Step 1: Write failing tests for param/metric/artifact logging**

Append to `tests/test_tracking.py`:

```python
from asago_policy_mapper.tracking import (
    log_params,
    log_metrics,
    log_artifact,
    log_child_run,
)


@patch("asago_policy_mapper.tracking.mlflow")
def test_log_params(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    ctx = init_tracking(enabled=True, experiment_name="test", run_name="r")
    log_params(ctx, {"model": "gemma", "threshold": "0.7"})
    mock_mlflow.log_params.assert_called_once_with({"model": "gemma", "threshold": "0.7"})


@patch("asago_policy_mapper.tracking.mlflow")
def test_log_params_skipped_when_disabled(mock_mlflow):
    ctx = TrackingContext(enabled=False)
    log_params(ctx, {"model": "gemma"})
    mock_mlflow.log_params.assert_not_called()


@patch("asago_policy_mapper.tracking.mlflow")
def test_log_metrics(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    ctx = init_tracking(enabled=True, experiment_name="test", run_name="r")
    log_metrics(ctx, {"recall": 0.85, "precision": 0.9})
    mock_mlflow.log_metrics.assert_called_once_with({"recall": 0.85, "precision": 0.9})


@patch("asago_policy_mapper.tracking.mlflow")
def test_log_artifact(mock_mlflow, tmp_path):
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value = mock_run

    artifact = tmp_path / "test.json"
    artifact.write_text('{"key": "value"}')

    ctx = init_tracking(enabled=True, experiment_name="test", run_name="r")
    log_artifact(ctx, artifact)
    mock_mlflow.log_artifact.assert_called_once_with(str(artifact))


@patch("asago_policy_mapper.tracking.mlflow")
def test_log_child_run(mock_mlflow):
    parent_run = MagicMock()
    parent_run.info.run_id = "parent123"
    mock_mlflow.start_run.return_value = parent_run

    ctx = init_tracking(enabled=True, experiment_name="test", run_name="r")

    child_run = MagicMock()
    child_run.info.run_id = "child456"
    mock_mlflow.start_run.return_value = child_run

    log_child_run(
        ctx,
        name="dhs-gov",
        params={"policy_name": "dhs-gov"},
        metrics={"recall": 0.8, "f1": 0.75},
        tags={"eval_status": "FAIL"},
        artifacts=[],
    )

    calls = mock_mlflow.start_run.call_args_list
    assert len(calls) == 2
    assert calls[1].kwargs["run_name"] == "dhs-gov"
    assert calls[1].kwargs["nested"] is True


@patch("asago_policy_mapper.tracking.mlflow")
def test_log_child_run_skipped_when_disabled(mock_mlflow):
    ctx = TrackingContext(enabled=False)
    log_child_run(ctx, name="policy", params={}, metrics={}, tags={}, artifacts=[])
    mock_mlflow.start_run.assert_not_called()
```

Also add this import at the top of the test file:

```python
from asago_policy_mapper.tracking import TrackingContext
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: new tests FAIL — `log_params`, `log_metrics`, `log_artifact`, `log_child_run` not yet defined.

- [ ] **Step 3: Implement logging functions**

Add to `src/asago_policy_mapper/tracking.py`:

```python
def log_params(ctx: TrackingContext, params: dict[str, str]) -> None:
    if not ctx.enabled:
        return
    try:
        mlflow.log_params(params)
    except Exception:
        logger.warning("MLflow log_params failed", exc_info=True)


def log_metrics(ctx: TrackingContext, metrics: dict[str, float]) -> None:
    if not ctx.enabled:
        return
    try:
        mlflow.log_metrics(metrics)
    except Exception:
        logger.warning("MLflow log_metrics failed", exc_info=True)


def log_artifact(ctx: TrackingContext, path: Path) -> None:
    if not ctx.enabled:
        return
    try:
        mlflow.log_artifact(str(path))
    except Exception:
        logger.warning("MLflow log_artifact failed for %s", path, exc_info=True)


def log_child_run(
    ctx: TrackingContext,
    *,
    name: str,
    params: dict[str, str],
    metrics: dict[str, float],
    tags: dict[str, str],
    artifacts: list[Path],
) -> None:
    if not ctx.enabled:
        return
    try:
        with mlflow.start_run(run_name=name, nested=True) as child:
            ctx.child_run_ids[name] = child.info.run_id
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            for artifact_path in artifacts:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))
    except Exception:
        logger.warning("MLflow child run failed for %s", name, exc_info=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/asago_policy_mapper/tracking.py tests/test_tracking.py
git commit -m "feat: add param, metric, artifact, and child run logging to tracking module"
```

---

### Task 4: Add prompt registry sync to tracking module

**Files:**
- Modify: `src/asago_policy_mapper/tracking.py`
- Modify: `tests/test_tracking.py`

- [ ] **Step 1: Write failing tests for prompt sync**

Append to `tests/test_tracking.py`:

```python
import hashlib

from asago_policy_mapper.tracking import sync_prompts


@patch("asago_policy_mapper.tracking.mlflow")
def test_sync_prompts_registers_new_prompt(mock_mlflow, tmp_path):
    prompts_dir = tmp_path / "templates" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "judge_risk_system.j2").write_text("You are a judge.")
    (prompts_dir / "judge_risk_user.j2").write_text("Judge this: {{ text }}")

    mock_mlflow.genai.load_prompt.side_effect = Exception("not found")
    mock_prompt = MagicMock()
    mock_prompt.version = 1
    mock_mlflow.genai.register_prompt.return_value = mock_prompt

    ctx = TrackingContext(enabled=True, run_id="abc")
    versions = sync_prompts(ctx, prompts_dir.parent)

    assert "judge_risk" in versions
    assert versions["judge_risk"] == 1
    mock_mlflow.genai.register_prompt.assert_called_once()


@patch("asago_policy_mapper.tracking.mlflow")
def test_sync_prompts_skips_unchanged(mock_mlflow, tmp_path):
    prompts_dir = tmp_path / "templates" / "prompts"
    prompts_dir.mkdir(parents=True)
    system = "You are a judge."
    user = "Judge this: {{ text }}"
    (prompts_dir / "judge_risk_system.j2").write_text(system)
    (prompts_dir / "judge_risk_user.j2").write_text(user)

    content_hash = hashlib.sha256((system + user).encode()).hexdigest()
    mock_existing = MagicMock()
    mock_existing.version = 3
    mock_existing.tags = {"content_hash": content_hash}
    mock_mlflow.genai.load_prompt.return_value = mock_existing

    ctx = TrackingContext(enabled=True, run_id="abc")
    versions = sync_prompts(ctx, prompts_dir.parent)

    assert versions["judge_risk"] == 3
    mock_mlflow.genai.register_prompt.assert_not_called()


@patch("asago_policy_mapper.tracking.mlflow")
def test_sync_prompts_returns_empty_when_disabled(mock_mlflow):
    ctx = TrackingContext(enabled=False)
    versions = sync_prompts(ctx, Path("/nonexistent"))
    assert versions == {}
    mock_mlflow.genai.register_prompt.assert_not_called()
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_tracking.py::test_sync_prompts_registers_new_prompt -v`
Expected: FAIL — `sync_prompts` not yet defined.

- [ ] **Step 3: Implement prompt sync**

Add to `src/asago_policy_mapper/tracking.py`, adding `import hashlib` and `import subprocess` at the top:

```python
import hashlib
import subprocess

_PROMPT_NAMES = ["judge_risk", "ground_evidence", "classify_risks"]


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def sync_prompts(ctx: TrackingContext, templates_dir: Path) -> dict[str, int]:
    if not ctx.enabled:
        return {}

    prompts_dir = templates_dir / "prompts"
    versions: dict[str, int] = {}
    git_sha = _get_git_sha()

    for name in _PROMPT_NAMES:
        system_file = prompts_dir / f"{name}_system.j2"
        user_file = prompts_dir / f"{name}_user.j2"

        if not user_file.exists():
            continue

        system_text = system_file.read_text() if system_file.exists() else ""
        user_text = user_file.read_text()
        content_hash = hashlib.sha256((system_text + user_text).encode()).hexdigest()

        try:
            existing = mlflow.genai.load_prompt(name)
            if existing.tags.get("content_hash") == content_hash:
                versions[name] = existing.version
                continue
        except Exception:
            pass

        try:
            template = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            prompt = mlflow.genai.register_prompt(
                name=name,
                template=template,
                commit_message=f"git:{git_sha}",
                tags={"content_hash": content_hash},
            )
            versions[name] = prompt.version
            logger.info("Registered prompt %s version %d", name, prompt.version)
        except Exception:
            logger.warning("Failed to register prompt %s", name, exc_info=True)

    return versions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/asago_policy_mapper/tracking.py tests/test_tracking.py
git commit -m "feat: add prompt registry sync to tracking module"
```

---

### Task 5: Integrate tracking into battery runner

**Files:**
- Modify: `run_extract_battery.py`

- [ ] **Step 1: Add CLI flags to argparse**

In `main()`, add two new arguments after the existing `--classify-taxonomies` argument:

```python
    parser.add_argument("--mlflow-experiment", default="risk-extraction", help="MLflow experiment name (default: risk-extraction)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
```

- [ ] **Step 2: Add tracking import at the top of `run_extract_battery.py`**

Add after the existing `from asago_policy_mapper.evals.eval import evaluate_extraction` import:

```python
from asago_policy_mapper.tracking import (
    init_tracking,
    end_tracking,
    is_tracking_enabled,
    log_params,
    log_metrics,
    log_artifact,
    log_child_run,
    sync_prompts,
)
```

- [ ] **Step 3: Add tracking initialization after config parsing in `main()`**

After the line `print()` (line 356 in current file, after the config summary prints), add:

```python
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
            "threshold_high": str(args.threshold_high),
            "threshold_low": str(args.threshold_low),
            "rrf_min_score": str(args.rrf_min_score),
            "classify_taxonomies": args.classify_taxonomies,
            "no_cross_encoder": str(args.no_cross_encoder),
            "jobs": str(args.jobs),
            "battery_config": battery_path.name,
        })
        templates_dir = PACKAGE_DIR / "src" / "asago_policy_mapper" / "templates"
        prompt_versions = sync_prompts(tracking_ctx, templates_dir)
        for pname, pversion in prompt_versions.items():
            log_params(tracking_ctx, {f"prompt/{pname}_version": str(pversion)})
        print(f"  mlflow:         {args.mlflow_experiment} (tracking enabled)")
    else:
        if not args.no_mlflow:
            print(f"  mlflow:         disabled (initialization failed)")
```

- [ ] **Step 4: Add child run logging after eval results are collected**

After the eval loop (the block starting with `# --- Eval ---` through the HTML report generation), add child run logging. Insert this after the "Generate HTML reports for runs without eval" block:

```python
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
```

- [ ] **Step 5: Add parent run aggregate metrics and cleanup**

At the very end of `main()`, just before the `print(f"\nOutput: {runs_dir}")` line, add:

```python
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
```

- [ ] **Step 6: Verify battery runner still works with `--no-mlflow`**

Run: `uv run python run_extract_battery.py batteries/risk-selected.yaml --base-url http://localhost:8000/v1 --model test --no-mlflow --help`
Expected: help output shows the new `--mlflow-experiment` and `--no-mlflow` flags.

- [ ] **Step 7: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/ -v`
Expected: all 85 existing tests still PASS.

- [ ] **Step 8: Commit**

```bash
git add run_extract_battery.py
git commit -m "feat: integrate MLflow tracking into battery runner"
```

---

### Task 6: Update justfile and documentation

**Files:**
- Modify: `justfile`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add mlflow-experiment flag support to justfile**

Update the `run-risk-extract-battery` recipe in `justfile` to pass through the `--no-mlflow` flag when needed. Add a new variable at the top:

```just
no_mlflow := ""
```

And update the recipe line to append:

```just
{{ if no_mlflow != "" { "--no-mlflow" } else { "" } }}
```

- [ ] **Step 2: Update CLAUDE.md**

Add to the Commands section:

```markdown
# Run battery with MLflow tracking disabled
just no_mlflow="1" run-risk-extract-battery batteries/risk-selected.yaml <base-url> <model>

# Run battery with custom MLflow experiment name
python run_extract_battery.py batteries/risk-selected.yaml --base-url <url> --model <model> --mlflow-experiment my-experiment
```

Add to the Key Conventions section:

```markdown
- MLflow tracking is enabled by default in the battery runner; set `MLFLOW_TRACKING_URI` to point to your MLflow server. Pass `--no-mlflow` to disable.
- Prompt templates are synced to the MLflow Prompt Registry at the start of each tracked battery run (hash-based dedup avoids duplicate versions).
```

- [ ] **Step 3: Commit**

```bash
git add justfile CLAUDE.md
git commit -m "docs: update justfile and CLAUDE.md for MLflow tracking"
```

---

### Task 7: End-to-end verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests PASS (existing + new tracking tests).

- [ ] **Step 2: Run battery with local MLflow (smoke test)**

Start a local MLflow server:

```bash
uv run mlflow server --port 5000 &
```

Run a battery against it:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 just run-risk-extract-battery batteries/risk-selected.yaml <base-url> <model>
```

Verify in the MLflow UI at http://localhost:5000:
- Experiment `risk-extraction` exists
- Parent run shows model, threshold, prompt version params
- Child runs show per-policy recall/precision/f1 metrics
- Artifacts (JSON, HTML) are attached to child runs

- [ ] **Step 3: Verify `--no-mlflow` produces identical output to before**

Run a battery with `--no-mlflow` and verify the console output and file output match the pre-MLflow behavior exactly.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: MLflow experiment tracking for battery runner evals"
```
