# Experiments

DSPy prompt optimization and cross-encoder evaluation experiments for the risk extraction pipeline. Each experiment optimizes a specific LLM stage using [DSPy GEPA](https://dspy.ai/) to find better instructions via reflection.

## Prerequisites

```bash
uv sync  # installs dspy>=2.6 from dev dependencies
```

You need:
- A running LLM endpoint (OpenAI-compatible API, e.g. vLLM serving Gemma)
- The [ai-atlas-nexus](https://github.com/ibm/ai-atlas-nexus) repo cloned locally
- For the judge experiment: no other dependencies
- For cross-encoder evaluation: models downloaded automatically from HuggingFace

## Experiments

### `dspy_judge/` — LLM Judge Prompt Optimization

Optimizes the judge that decides whether borderline candidates (cross-encoder score between threshold_low–threshold_high) are relevant to a text chunk.

**Dataset:** Built from enriched ground truth (27 policies, risk-level only). Each example is a (chunk_text, candidate_risks) → expected_verdicts mapping. Hard negatives are mined using the cross-encoder: for each chunk, all non-GT risks are scored with ms-marco and the highest-scoring false matches become negatives. Category-level taxonomies (NIST, OWASP, AILuminate, etc.) are excluded from the risk pool.

**Build dataset + run optimization:**

```bash
uv run python -m experiments.dspy_judge \
  --base-url http://your-llm-endpoint/v1 \
  --model gemma-4-26b-a4b-it \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --auto medium
```

**Baseline only (no optimization):**

```bash
uv run python -m experiments.dspy_judge \
  --base-url http://your-llm-endpoint/v1 \
  --model gemma-4-26b-a4b-it \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --baseline-only
```

**Options:**
- `--auto light|medium|heavy` — GEPA optimization intensity (default: medium)
- `--baseline-only` — skip optimization, just evaluate the current prompt

**Output:** Results and optimized program saved to `experiments/dspy_judge/runs/`.

**Applying results:** Extract the `optimized_instructions` from the run JSON and update `src/asago_policy_mapper/templates/prompts/judge_risk_system.j2`.

**Time:** ~30–60 min depending on endpoint speed.

---

### `dspy_ground/` — Grounding Prompt Optimization

Optimizes the grounding stage that determines whether accepted candidates are actually discussed in a text chunk and extracts evidence quotes. This is the precision filter after retrieval+judging.

**Dataset:** Built from enriched ground truth evidence. Hard negatives use same-document other-chunk negatives and optionally pipeline-mined negatives (`grounding_filtered_candidates` from battery runs).

**Metric:** Combined score = 80% decision F1 (grounded true/false) + 20% quote quality (token-level F1).

```bash
uv run python -m experiments.dspy_ground \
  --base-url http://your-llm-endpoint/v1 \
  --model gemma-4-26b-a4b-it \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --auto medium
```

**With pipeline-mined negatives:**

```bash
uv run python -m experiments.dspy_ground \
  --base-url http://your-llm-endpoint/v1 \
  --model gemma-4-26b-a4b-it \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --run-dir extract-runs/risk-selected_YYYYMMDD_HHMMSS \
  --auto medium
```

**Key finding:** GEPA consistently converges on "Mitigation-as-Evidence" prompts that improve isolated recall but over-ground in production, particularly for ai-risk-taxonomy risks. The original 4-line grounding prompt achieves near-optimal end-to-end precision (0.813). See EXPERIMENT_LOG.md for details.

---

### `dspy_embedding/` — Embedding Instruction Optimization

Optimizes the query instruction prefix for instruction-aware embedding models (e.g. Qwen3-Embedding). The instruction is prepended to queries as `"Instruct: {instruction}\nQuery: {text}"` when using remote embedding endpoints. GEPA tunes this instruction to maximize risk-level retrieval recall.

**Metric:** Risk-level recall (per-policy) — fraction of ground truth risks retrieved across all chunks via hybrid search (BM25 + semantic + RRF, no cross-encoder).

**Optimization target:** The instruction string is the DSPy signature's `instructions` field. GEPA rewrites it via reflection; the module injects it into the remote bi-encoder's query prefix on each trial.

```bash
uv run python -m experiments.dspy_embedding \
  --bi-encoder-model https://qwen-embedding.apps.example.com/v1/embeddings \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --base-url http://your-llm-endpoint/v1 \
  --model gemma-4-26b-a4b-it \
  --auto medium
```

**Quick prototyping (subset of policies):**

```bash
uv run python -m experiments.dspy_embedding \
  --bi-encoder-model https://qwen-embedding.../v1/embeddings \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --base-url http://your-llm-endpoint/v1 \
  --model gemma-4-26b-a4b-it \
  --train-policies sap,cisco-supplier,firstsource,guy-nhs,rdash-nhs \
  --auto light
```

**Baseline only:**

```bash
uv run python -m experiments.dspy_embedding \
  --bi-encoder-model https://qwen-embedding.../v1/embeddings \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --base-url http://your-llm-endpoint/v1 \
  --model gemma-4-26b-a4b-it \
  --baseline-only
```

**Options:**
- `--auto light|medium|heavy` — GEPA optimization intensity (default: medium)
- `--train-policies` — comma-separated subset for fast iteration
- `--baseline-only` — skip optimization, just measure current recall
- `--top-k` — candidates per chunk from hybrid search (default: 50)
- `--rrf-min-score` — RRF score floor (default: 0.015)

**Output:** Results and optimized program saved to `experiments/dspy_embedding/runs/`.

**Applying results:** Update `RetrievalConfig.query_instruction` in `src/asago_policy_mapper/extract/models.py` with the optimized instruction, or pass it via `--query-instruction` at extraction time.

---

### `cross_encoder_tuning/` — Cross-Encoder Evaluation & Fine-Tuning

Scripts for building datasets, evaluating, and fine-tuning cross-encoder reranker models.

**Dataset:** JSONL pairs of (risk_description, chunk_text) with positive/hard-negative/easy-negative labels. Built from enriched GT + extraction run results. Category-level taxonomies excluded from risk pool.

**Evaluate a model:**

```bash
uv run python -m experiments.cross_encoder_tuning.evaluate \
  --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --dataset-dir experiments/cross_encoder_tuning/datasets
```

**Build dataset:**

```bash
uv run python -m experiments.cross_encoder_tuning.dataset \
  --run-dir extract-runs/risk-selected_YYYYMMDD_HHMMSS \
  --nexus-base-dir /path/to/ai-atlas-nexus \
  --output-dir experiments/cross_encoder_tuning/datasets
```

**Scripts:**
- `dataset.py` — builds train/eval JSONL from enriched GT + extraction results
- `evaluate.py` — scores a cross-encoder on the dataset at various thresholds
- `finetune.py` — fine-tunes a cross-encoder using sentence-transformers
- `rewrite_descriptions.py` — rewrites risk descriptions via LLM for better cross-encoder matching

---

### `dspy_classify/` — NIST Classification Prompt Optimization (archived)

Previously optimized the post-retrieval NIST classification step. This step has been removed from the pipeline — category-level assessment is now done via the SSSOM mapping at eval time. Kept for reference.

---

### `dspy_extract_ai/` — KG AI System Extraction (separate experiment)

Optimizes AI system extraction for the knowledge graph pipeline (policy-extractor). Not related to the direct risk extraction pipeline.

## Policy Split

All experiments use a 13/14 train/eval policy split:

**Train (13):** sap, cisco-supplier, firstsource, guy-nhs, rdash-nhs, dhs-gov, eu-com, ovic, camden-borough-work, llvm, amadeus, fs-isac, gray

**Eval (14):** ars, leicestershire_police, lse-legreg, aus-gov, lenovo, prosus, new-york-state, lse-marking, ebay, vps, npcc, penn, st-johns, icrc

Amadeus (train) and ICRC (eval) are split to test generalisation on EU AI Act prohibition-list patterns.

## Experiment Log

See `EXPERIMENT_LOG.md` for a chronological record of all experiments, configurations, and results.

## End-to-End Testing

After updating prompt templates with optimized instructions, run a full battery to measure end-to-end impact:

```bash
just run-risk-extract-battery batteries/risk-selected.yaml
```

Compare risk-level AND category-level P/R/F1 against the baseline. Current baseline: risk-level F1=0.708, NIST category F1=0.923.
