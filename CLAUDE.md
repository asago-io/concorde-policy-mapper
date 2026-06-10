Concorde Policy Mapper extracts AI risks from policy documents using the IBM AI Atlas Nexus risk taxonomy (~524 risks). It uses hybrid retrieval (BM25 + semantic embeddings + cross-encoder reranking) to match document chunks against Nexus risks, then LLM-judges borderline candidates and grounds accepted ones with evidence quotes.

## Commands

```bash
# Install / sync dependencies
uv sync

# Run all tests
uv run pytest
just test

# Run a single test file or specific test
uv run pytest tests/test_extract_pipeline.py
uv run pytest tests/test_extract_retrieve.py::test_classify_candidates -v

# Extract risks from a document
uv run concorde-policy-mapper extract policy.pdf -o output/ \
  --base-url http://localhost:8000/v1 --model my-model \
  --nexus-base-dir /path/to/ai-atlas-nexus

# Evaluate against ground truth
uv run concorde-policy-mapper eval output/ -g evals/ground_truth/policy-name.yaml

# Run full battery (27 policies, parallel)
just run-risk-extract-battery batteries/risk-selected.yaml <base-url> <model>
# Or directly with more options:
python run_extract_battery.py batteries/risk-selected.yaml --base-url <url> --model <model> -j 6
# Default config: expand-siblings + 3-pass grounding + 3-pass expansion

# Run battery with MLflow tracking disabled
just no_mlflow="1" run-risk-extract-battery batteries/risk-selected.yaml <base-url> <model>

# Run battery with custom MLflow experiment name
python run_extract_battery.py batteries/risk-selected.yaml --base-url <url> --model <model> --mlflow-experiment my-experiment

# IR-only mode (no LLM judge/grounding, no --base-url/--model needed)
uv run concorde-policy-mapper extract policy.pdf -o output/ \
  --nexus-base-dir /path/to/ai-atlas-nexus --no-judge --no-grounding
just no_judge="1" no_grounding="1" run-risk-extract-battery batteries/risk-selected.yaml

# Judge only, no grounding (test judge contribution in isolation)
uv run concorde-policy-mapper extract policy.pdf -o output/ \
  --nexus-base-dir /path/to/ai-atlas-nexus --no-grounding \
  --base-url <url> --model <model>

# Skip causal synthesis (use static YAML chains)
uv run concorde-policy-mapper extract policy.pdf -o output/ \
  --nexus-base-dir /path/to/ai-atlas-nexus --no-causal-synthesis \
  --base-url <url> --model <model>

# Smaller chunks with larger judge context window
uv run concorde-policy-mapper extract policy.pdf -o output/ \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --chunk-max-tokens 256 --judge-context-tokens 512 \
  --base-url <url> --model <model>

# Use remote embedding models on GPU cluster
uv run concorde-policy-mapper extract policy.pdf -o output/ --no-judge --no-grounding \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --bi-encoder-model https://bge-m3-model-serving.apps.example.com/v1/embeddings \
  --cross-encoder-model https://gte-reranker-model-serving.apps.example.com/v1/score

# Rebuild mitigation index (after data file changes)
python scripts/build_mitigation_index.py
```

## Architecture

### Extraction Pipeline (`extract/pipeline.py::run_extraction`)

```
Documents → parse_document() → chunk_documents() → per-chunk retrieve → judge → ground → merge → causal synthesis
                                                                    ↑ ThreadPoolExecutor (steps 6-7)
```

`run_extraction(documents, client, config, risks, retrieval, *, ocr)` accepts a `RetrievalConfig` dataclass (defined in `models.py`) that bundles all retrieval/IR parameters with pre-resolved properties (`effective_cross_encoder_model`, `effective_rrf_min_score`) and a `to_metadata()` method for the output JSON.

1. **Parse** (`parse.py`) — Docling converts PDF/DOCX/HTML to markdown; plain text passes through
2. **Chunk** (`parse.py`) — HybridChunker splits into ~512-token chunks preserving page/section metadata
3. **Agentic filter** — If document lacks agent-related terminology, agentic risks are excluded from the catalog
4. **Index** (`index.py`) — `RiskIndex` builds BM25 + bi-encoder embeddings + optional cross-encoder over risk-level taxonomies only. RRF fusion is shared via `_rrf_fuse()` between bi-encoder and ColBERT paths. Score normalization uses a `_make_score_normalizer()` factory resolved at init time.
5. **Retrieve** (`retrieve.py`) — Per-chunk: BM25 + semantic → RRF fusion → cross-encoder rerank → classify into accepted/borderline/discarded. Classification dispatches to `classify_by_rank()` or `classify_by_threshold()` based on `threshold_high`.
6. **Judge** (`retrieve.py`) — LLM judges borderline candidates using padded text (adjacent chunk context). `--judge-context-tokens` controls the context window size (default: sentence-based padding; set to e.g. 512 to give the judge a wider window when using smaller chunks). Parallel via ThreadPoolExecutor
7. **Ground** (`attribute.py`) — LLM extracts evidence quotes + confidence (high/medium/low) for accepted candidates; ungrounded ones filtered out. Parallel via ThreadPoolExecutor. Multi-pass grounding (default 3 passes, `--grounding-passes`) unions results across passes to reduce LLM non-determinism. Match construction uses `build_risk_match()` and `determine_accepted_by()` pure functions to avoid duplication across grounding/no-grounding/expansion paths.
8. **Merge** (`merge.py`) — Deduplicate matches across chunks, keep best confidence and top-3 evidence spans
9. **Expand** (`expand.py`) — Sibling expansion (enabled by default): expands found risks to parent siblings + cross-taxonomy mappings, then grounds the expanded set against relevant document chunks. Multi-pass expansion grounding (default 3 passes, `--expansion-passes`) unions results to stabilize data-type variant recovery.
10. **Causal synthesis** (`attribute.py`) — LLM synthesizes domain-specific causal chains (`threat`/`threat_source`/`vulnerability`/`consequence`/`impact`) for each matched risk, grounded in the evidence-anchored chunks. Parallel via ThreadPoolExecutor. Disabled with `--no-causal-synthesis`; static YAML chains from `data/atlas_risk_threats.yaml` and `data/atlas_risk_consequences.yaml` serve as fallback for any fields not populated.
11. **Mitigations** (`mitigations.py`) — Post-processing: enrich each matched risk with recommended mitigation actions from a pre-built index (`data/atlas_risk_to_actions.yaml`)

With `--no-cross-encoder`, steps 5-6 are replaced by RRF score floor filtering (no LLM judging).

With `--no-judge`, step 6 is skipped (borderline candidates auto-promoted to accepted). With `--no-grounding`, step 7 is skipped (accepted candidates become matches without evidence). Both can be combined for pure IR evaluation — no LLM calls at all.

Category-level taxonomy mapping (NIST, OWASP, AILuminate) is handled at eval time via a static SSSOM mapping, not during extraction.

### LLM Integration (`llm.py`)

- `create_client()` wraps OpenAI with `instructor` (JSON mode) for structured Pydantic outputs
- `TokenTracker` accumulates usage across stages; `LLMConfig` holds connection details
- Automatic retry on validation errors (appends error hint), context overflow detection (reduces max_tokens), and prompt truncation on incomplete output
- All LLM calls use `response_model=list[PydanticModel]` with `temperature=0.0`

### Prompt Templates (`templates/prompts/`)

Two Jinja2 template pairs (`_system.j2` + `_user.j2`): `judge_risk`, `ground_evidence`. Loaded by `prompts.py::render_prompt()`.

### Evaluation (`evals/eval.py`)

Two-tier evaluation:
- **Tier 1 (risk-level)**: Compares extracted risk IDs against ground truth YAML. Computes precision/recall/F1 overall and per-taxonomy. 27 ground truth files in `evals/ground_truth/` (1519 risks) — risk-level only (no category-level entries).
- **Tier 2 (category-level)**: Derives NIST/OWASP/AILuminate/ASI categories from risk IDs via `data/risk_to_category.sssom.tsv` (SSSOM mapping, 802 entries). Computes P/R/F1 per category taxonomy. Only uses strong predicates (exact/close/broadMatch), excludes relatedMatch.

### Cross-Taxonomy Mapping (`data/risk_to_category.sssom.tsv`)

Static SSSOM file mapping 486 risk-level risks to 4 category-level taxonomies (NIST AI RMF 12 risks, OWASP LLM 10 risks, AILuminate 12 risks, OWASP ASI 10 risks). Built from Nexus mapping files + manually reviewed gap-fill for IBM agentic risks, Credo, MIT, and AIR 2024 (314 risks via group-level inheritance).

### Mitigation Index (`data/atlas_risk_to_actions.yaml`)

Pre-built lookup mapping 80 Atlas risk IDs to ~552 recommended mitigation actions across 3 frameworks. Generated by `scripts/build_mitigation_index.py` from direct `action → atlas-*` mapping files:

- **OWASP LLM Top 10 v2.0** (114 action-risk links) — `data/owasp_llm_2.0_actions_data.yaml`
- **NIST AI RMF 600-1** (338 action-risk links) — `data/nist_ai_rmf_actions_to_atlas_data.yaml`
- **AIUC-1** (100 action-risk links) — `data/aiuc1_actions_to_atlas_data.yaml`

All mappings are direct `hasRelatedRisk: atlas-*` — no transitive cross-framework hops. Each action is categorized as `technical` (engineering deploys), `operational` (ops/QA executes), or `governance` (leadership/compliance owns) via rules in `data/mitigation_categories.yaml`. NIST by RMF function prefix (GV=governance, MP/MS=operational, MG=technical), AIUC-1 by principle letter (a-b=technical, c-d=operational, e-f=governance), and OWASP via explicit per-action assignments.

Regenerate after data file changes: `python scripts/build_mitigation_index.py`

### Battery Runner (`run_extract_battery.py`)

Runs `concorde-policy-mapper extract` as a subprocess per policy in a battery YAML config, with parallel execution (default 6 workers). Auto-evaluates against ground truth, generates per-run HTML reports, and a battery summary with per-taxonomy heatmaps.

## Key Conventions

- `NEXUS_BASE_DIR` env var or `--nexus-base-dir` flag points to a local clone of `github.com/IBM/ai-atlas-nexus`
- Risk IDs are taxonomy-prefixed: `atlas-` → ibm-risk-atlas, `nist-` → nist-ai-rmf, `credo-` → credo-ucf, etc. (see `evals/eval.py::_TAXONOMY_PREFIXES`)
- Excluded taxonomies (not loaded from Nexus): category-level (`nist-ai-rmf`, `owasp-llm-2.0`, `ailuminate-v1.0`, `owasp-asi`, `shieldgemma-taxonomy`) and others (`mit-ai-risk-repository-causal`, `ibm-granite-guardian`) — see `cli.py::EXCLUDED_TAXONOMIES`
- `LLMCallRecord` captures every LLM call (messages, response, timing) in the ExtractionResult for analysis/debugging
- `debug.py` writes per-call JSON files when `--debug <dir>` is passed
- MLflow tracking is enabled by default in the battery runner; set `MLFLOW_TRACKING_URI` to point to your MLflow server. Pass `--no-mlflow` to disable.
- Prompt templates are synced to the MLflow Prompt Registry at the start of each tracked battery run (hash-based dedup avoids duplicate versions)

## Retrieval Architecture Notes

- The cross-encoder (ms-marco-MiniLM) has AUC ~0.50 on pipeline-mined negatives — it does not discriminate semantically. It functions as a volume reduction filter: randomly rejecting ~70% of candidates, with the grounding stage catching the noise. See `experiments/EXPERIMENT_LOG.md` for details.
- Candidate selection supports both rank-based (`--top-n-accept`, `--top-n-judge`) and legacy threshold-based (`--threshold-high`, `--threshold-low`) modes. Default is rank-based with top_n_accept=10, top_n_judge=10, min_score_floor=0.70, bm25_rescue_rank=0 (disabled), rrf_min_score=0.015. These defaults are tuned for recall with GTE-reranker-modernbert-base or no-cross-encoder mode.
- Multi-pass grounding (default 3 passes) and expansion (default 3 passes) reduce LLM non-determinism by running each grounding call multiple times and taking the union. This stabilizes which base risks survive grounding (affecting expansion seeds) and which expansion candidates get accepted.
- ColBERT late-interaction models are supported via `--colbert-model` (replaces bi-encoder + cross-encoder with a single model using MaxSim scoring)
- Modern cross-encoders (GTE, BGE) output calibrated scores — the pipeline skips sigmoid normalisation for these (see `_SIGMOID_MODELS` in `index.py`). Score normalisation is resolved at index init time via `_make_score_normalizer()` which returns a callable (NLI softmax, sigmoid, or clip-to-[0,1]).
- Embedding/reranking models can be served on GPU via vLLM's embedding/scoring API on the cluster. Pass a URL as `--bi-encoder-model` (uses `/v1/embeddings`) or `--cross-encoder-model` (uses `/v1/score`). ColBERT models (`--colbert-model`) are local-only — vLLM returns pooled embeddings, not token-level

## Dependency Pins

- `ai-atlas-nexus` is pinned to a specific commit (v1.2.1) — `@main` may have breaking Pydantic schema changes
- `torch<2.12` and `transformers<5.6` — newer versions introduce an MPS-incompatible `rt_detr_v2` layout model in docling's PDF pipeline on Apple Silicon

## Experiments

- Always update `experiments/EXPERIMENT_LOG.md` with results after running any experiment or battery that produces new data points
- Include MLflow experiment name and run ID in experiment log entries when MLflow tracking is enabled (e.g., `**MLflow:** experiment=risk-extraction, run_id=abc123`)
- Cross-encoder scores are random on pipeline-mined negatives (AUC ~0.50) — the ms-marco cross-encoder does NOT discriminate semantically; it acts as a volume reduction filter. Do not rely on cross-encoder scores for hard negative mining — use pipeline-mined negatives from `grounding_filtered_candidates` instead.
- Cross-encoder eval datasets MUST use pipeline-mined negatives (from actual battery runs), not cross-encoder-mined negatives. The latter are biased toward the mining model's specific failure modes.

## Development

- DO NOT skip updating the changelog with any changes made
- DO NOT skip updating AGENTS.md and the README.md when changes require it
