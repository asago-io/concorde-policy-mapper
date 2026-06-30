Asago Policy Mapper extracts AI risks from policy documents using the IBM AI Atlas Nexus risk taxonomy (~524 risks). It uses hybrid retrieval (BM25 + semantic embeddings + cross-encoder reranking) to match document chunks against Nexus risks, then LLM-judges borderline candidates and grounds accepted ones with evidence quotes.

## Commands

```bash
# Install / sync dependencies
uv sync

# Run all tests
uv run pytest
just test

# Run fast tests only (skip slow embedding-model tests)
uv run pytest -m "not slow"

# Run a single test file or specific test
uv run pytest tests/test_extract_pipeline.py
uv run pytest tests/test_extract_retrieve.py::test_classify_candidates -v

# Format + lint + type check (all static checks)
just tidy

# Lint
uv run ruff check src/ tests/
just lint

# Lint single file (with auto-fix)
uv run ruff check --fix path/to/file.py

# Format
uv run ruff format src/ tests/
just format

# Format single file
uv run ruff format path/to/file.py

# Type check
uv run mypy src/asago_policy_mapper/
just type-check

# Type check single file
uv run mypy path/to/file.py

# See README.md for full CLI usage examples (extract, eval, battery, remote models)
```

## Architecture

### Extraction Pipeline

The pipeline in `src/asago_policy_mapper/extract/pipeline.py` has several alternative modes controlled by CLI flags — when modifying one path, be aware the others exist:

- **Default (query-gen on):** LLM generates search queries per section group, all candidates go to grounding. Disable with `--no-query-gen` to use per-chunk BM25+semantic retrieval with cross-encoder reranking and LLM judging.
- `--no-cross-encoder`: RRF score floor filtering replaces cross-encoder reranking (no LLM judging)
- `--no-judge`: borderline candidates auto-promoted (skips LLM judge)
- `--no-grounding`: accepted candidates become matches without evidence extraction
- `--no-judge --no-grounding`: pure IR evaluation — no LLM calls at all
- `--no-causal-synthesis`: skips causal chain synthesis; static YAML chains from `src/asago_policy_mapper/data/` used as fallback

**Variant collapsing:** Risk IDs containing `---` (e.g. `unauthorized-processing---biometric-data`) are collapsed into synthetic parent entries for indexing. After grounding, a variant grounding step determines which specific sub-types have evidence. This affects how risk IDs flow through the entire pipeline — don't treat `---` IDs as regular risks.

**Multi-pass grounding:** Grounding and expansion each run multiple passes (default 3) and union results to reduce LLM non-determinism. Do not remove the extra passes — they stabilize which risks survive grounding.

## Key Conventions

- `NEXUS_BASE_DIR` env var or `--nexus-base-dir` flag points to a local clone of `github.com/IBM/ai-atlas-nexus`
- Risk IDs are taxonomy-prefixed: `atlas-` → ibm-risk-atlas, `nist-` → nist-ai-rmf, `credo-` → credo-ucf, etc. (see `evals/eval.py::_TAXONOMY_PREFIXES`)
- Excluded taxonomies (not loaded from Nexus): category-level (`nist-ai-rmf`, `owasp-llm-2.0`, `ailuminate-v1.0`, `owasp-asi`, `shieldgemma-taxonomy`) and others (`mit-ai-risk-repository-causal`, `ibm-granite-guardian`) — see `cli.py::EXCLUDED_TAXONOMIES`
- `LLMCallRecord` captures every LLM call (messages, response, timing) in the ExtractionResult for analysis/debugging
- `debug.py` writes per-call JSON files when `--debug <dir>` is passed
- MLflow tracking is enabled by default in the battery runner; set `MLFLOW_TRACKING_URI` to point to your MLflow server. Pass `--no-mlflow` to disable.
- Prompt templates are synced to the MLflow Prompt Registry at the start of each tracked battery run (hash-based dedup avoids duplicate versions)

## Retrieval Architecture Notes

- The default cross-encoder (ms-marco-MiniLM) has AUC ~0.50 on pipeline-mined negatives — it does not discriminate semantically. It functions as a volume reduction filter; the grounding stage provides the actual precision filtering. Do not attempt to tune or rely on its scores for ranking. See `experiments/EXPERIMENT_LOG.md` for details.
- Do not use cross-encoder scores for hard negative mining — use pipeline-mined negatives from `grounding_filtered_candidates` instead. Eval datasets MUST use pipeline-mined negatives (from actual battery runs), not cross-encoder-mined negatives — the latter are biased toward the mining model's specific failure modes.

## Dependency Pins

- `ai-atlas-nexus` is pinned to `@main`
- `torch<2.12` and `transformers<5.6` — newer versions introduce an MPS-incompatible `rt_detr_v2` layout model in docling's PDF pipeline on Apple Silicon

## Experiments

- Always update `experiments/EXPERIMENT_LOG.md` with results after running any experiment or battery that produces new data points
- Include MLflow experiment name and run ID in experiment log entries when MLflow tracking is enabled (e.g., `**MLflow:** experiment=risk-extraction, run_id=abc123`)

## Development

- `AGENTS.md` is a symlink to `CLAUDE.md` — they are the same file. Do not treat them as separate files or suggest deduplication.
- DO NOT skip updating the changelog with any changes made
- DO NOT skip updating `CLAUDE.md`/`AGENTS.md` and `README.md` when changes require it
