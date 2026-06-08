# Changelog

## [Unreleased]

### Changed
- **Trim mitigation sources to 3 frameworks**: dropped MIT AI Risk Repository (831 actions, 60% of index) and Credo UCF (41 controls) from the mitigation index. Kept NIST AI RMF 600-1, OWASP LLM Top 10 v2.0, and AIUC-1. Index reduced from 1,693 to 552 action-risk links across 80 risks (was 83). Max mitigations per risk drops from 162 to 26. MIT was dominated by governance questionnaires with ~24% sourced from NIST anyway; Credo overlapped heavily with NIST policy guidance.

### Added
- **In-context causal chain synthesis**: new post-merge pipeline stage that uses the LLM to synthesize domain-specific causal chains (`threat`, `threat_source`, `vulnerability`, `consequence`, `impact`) grounded in the actual policy document. Replaces static YAML-sourced chains with actionable, context-aware decompositions. Enabled by default when an LLM is available; disable with `--no-causal-synthesis`. Static YAML chains serve as fallback for risks where synthesis fails or is disabled.
- **DSPy embedding instruction optimization** (`experiments/dspy_embedding/`): uses GEPA to optimize the query instruction prefix for instruction-aware embedding models (e.g. Qwen3-Embedding), maximizing risk-level retrieval recall
- **`RiskIndex.set_query_instruction()`**: update the remote bi-encoder's query instruction without rebuilding the index
- **Generative reranker support** (`--cross-encoder-type generative`): supports Qwen3-Reranker and similar models that use `/v1/chat/completions` with yes/no logprob scoring instead of `/v1/score`. Pass `--cross-encoder-type generative` with the model URL. Nemotron-rerank uses `/v1/score` and works with the default `--cross-encoder-type score`.
- **`--grounding-passes N` flag**: runs per-chunk grounding N times and unions the results, reducing variance from LLM non-determinism. Stabilizes which base risks survive grounding, which in turn stabilizes expansion seeds.
- **GT enrichment round 2**: +84 ai-risk-taxonomy risks across 15 policies (1435→1519 total, 206→290 AIR). All high-confidence keyword-matched expansion candidates from Qwen3-Reranker-4B run.

### Changed
- **New pipeline defaults**: `expand_siblings=True`, `grounding_passes=3`, `expansion_passes=3`. Multi-pass grounding reduces LLM non-determinism, stabilizing which risks survive grounding and expansion. Micro F1 0.717 on 27 policies (1519 GT risks).
- **`--expansion-passes N` flag**: runs expansion grounding N times and unions the results, reducing variance from LLM non-determinism on borderline data-type variant risks.

### Fixed
- **`accepted_by` label bug**: when `no_judge=True` with grounding enabled, candidates were incorrectly tagged `"llm_judge"` instead of `"auto_promoted"`. The grounding path now uses the same `determine_accepted_by()` function as the no-grounding path.

### Changed
- **Pipeline decomplection** (steps from `docs/decomplection-analysis.md`):
  - Extracted `determine_accepted_by()` and `build_risk_match()` pure functions — eliminates 3-way RiskMatch construction duplication and scattered `accepted_by` computation.
  - Extracted `timed()` context manager — replaces 8 inline `t0 = time.time()` / `timing[...] = ...` pairs.
  - Split `classify_candidates()` into `classify_by_rank()` and `classify_by_threshold()` with a thin dispatcher for backward compatibility.
  - Extracted `_rrf_fuse()` shared function and `_make_score_normalizer()` factory from `RiskIndex` — eliminates RRF accumulation duplication between `hybrid_search` and `_hybrid_search_colbert`, and replaces inline score normalization branching with a factory resolved at init time.
  - Bundled 18 retrieval parameters into `RetrievalConfig` dataclass with pre-resolved properties (`effective_cross_encoder_model`, `effective_rrf_min_score`) and `to_metadata()`. `run_extraction` now accepts `retrieval: RetrievalConfig` instead of individual keyword arguments.
  - Extracted `_collect_ungrounded()` and `_run_grounding()` from `run_extraction` — reduces CC from 48 to ~25.
  - Split `_call_with_retry()` into `_retry_with_validation()` (handles context overflow + validation errors) and the outer truncation retry loop.
  - Extracted `_load_colbert()`, `_load_bi_encoder()`, `_load_cross_encoder()` factory functions from `RiskIndex.__init__` — reduces CC from 24 to ~8.
  - Extracted `_pad_with_budget()` from `build_padded_text()` — separates token-budget padding from sentence-based padding.
  - Extracted `_run_judge()` and `_build_chunk_risk_map()` from `run_extraction` — CC 38→24.
  - Extracted `_resolve_via_crossmap()` from `enrich_with_mitigations` — deduplicates crossmap fallback for mitigations, threats, and consequences. CC 21→13.
  - Extracted `_rescue_and_merge_scores()` from `hybrid_search` — CC 18→12.
- **Decomplection analysis revised**: fixed inaccurate counts (18→23 params, 7→8 timing sites, 30→27 CLI params), replaced strategy pattern proposal for RiskIndex with lighter shared-function approach, replaced debug module globals entry with eval↔extraction schema drift concern, added caveat to RetrievalConfig that a dataclass alone is cosmetic without pre-resolving downstream decisions.
- **Schema drift smoke test**: `test_extraction_result_schema_compatible_with_eval` constructs an `ExtractionResult` via Pydantic, serializes to JSON, and runs eval — catches silent breakage if extraction output fields drift from what eval reads.

### Added
- **`tests/test_llm.py`**: unit tests for low-coverage LLM utility functions — `_strip_titles`, `TokenTracker` methods (`add`, `_usage_values`, `to_dict`, `record_incident`, `set_stage`), `_track_completion`, `_extract_response_content`, `_truncate_messages`, and `_call_with_retry` outer loop (43 tests).
- **Mitigation recommendations**: each extracted risk now includes recommended mitigation actions from 3 frameworks (OWASP LLM Top 10, NIST AI RMF 600-1, AIUC-1). Pre-built index maps 80 Atlas risks to ~552 action entries via direct `action → atlas-*` mappings (no transitive cross-framework hops). Non-Atlas risks resolve to Atlas equivalents via Nexus cross-framework mappings at enrichment time. Mitigations appear in JSON output (`mitigations` field on `RiskMatch` with `action_id`, `action_name`, `description`, `source`, `category`) and in the HTML report as an expandable section per risk, grouped by category (technical/operational/governance) then source.
- **`scripts/build_mitigation_index.py`**: generates `data/atlas_risk_to_actions.yaml` from 3 direct mapping files. Each action is categorized as `technical`, `operational`, or `governance` via rules in `data/mitigation_categories.yaml`.
- **Direct action→risk mapping files**: `data/nist_ai_rmf_actions_to_atlas_data.yaml` (212 NIST actions → 338 risk links), `data/aiuc1_actions_to_atlas_data.yaml` (49 AIUC-1 requirements → 100 risk links). All hand-reviewed.
- **`data/mitigation_categories.yaml`**: category assignment rules (NIST RMF prefix → category, AIUC-1 principle → category) plus explicit assignments for OWASP actions.
- **`data/owasp_llm_2.0_actions_data.yaml`**: 80 structured mitigation actions extracted from OWASP LLM Top 10 v2.0, each mapped to Atlas risk IDs.
- **`--no-judge` flag**: skips LLM judge stage; auto-promotes borderline candidates to accepted.
- **`--no-grounding` flag**: skips LLM grounding stage; accepted candidates become matches with empty evidence. Both flags can be used independently or together. `--base-url`/`--model` are optional when both are set.
- **`--chunk-max-tokens` flag**: configurable chunk size (default: 512). Exposed in CLI and battery runner.
- **`--judge-context-tokens` flag**: controls the judge's context window independently of chunk size. When set (e.g. 512), the judge receives text from adjacent chunks up to the token budget, allowing smaller retrieval chunks without sacrificing judge context.
- **Remote embedding/reranking model support**: pass a URL (e.g. `--bi-encoder-model https://host/v1/embeddings`) to use vLLM-served models on GPU instead of local sentence_transformers. Supports `/v1/embeddings` for bi-encoders and `/v1/score` for cross-encoders.

### Changed
- **Retrieval defaults tuned for recall**: `top_n_accept` 5→10, `top_n_judge` 5→10, `min_score_floor` 0.0→0.70, `bm25_rescue_rank` 10→0, `rrf_min_score` 0.01→0.015. BM25 rescue disabled (10.8% precision, −0.034 F1). RRF floor raised to 0.015 for no-cross-encoder mode (R=0.852, cuts ~16% of noise with minimal recall loss).
- **Two-tier evaluation**: category-level precision/recall/F1 alongside risk-level metrics. Evaluates whether the pipeline captures the right risk *themes* (NIST AI RMF, OWASP LLM, OWASP ASI categories) even when individual risk-level matches are missed. Category-level NIST F1=0.938 vs risk-level F1=0.771.
- **Cross-taxonomy SSSOM mapping** (`data/risk_to_category.sssom.tsv`): 802 mappings linking 486 specific risks (IBM Risk Atlas, Credo UCF, AIR 2024, MIT AI Risk Repository) to 4 category-level taxonomies (NIST AI RMF, OWASP Top 10 LLM, AILuminate, OWASP ASI). Built from existing Nexus mapping files + manually reviewed gap-fill.
- Risk ID sanitisation in eval to handle malformed upstream Nexus IDs (name appended after space).
- OWASP ASI (Agentic Security Initiative) taxonomy support as a category-level target.

### Changed
- **Pipeline no longer runs taxonomy classification step.** Category-level taxonomies (NIST, OWASP LLM, AILuminate, OWASP ASI, ShieldGemma) are excluded from Nexus risk loading and evaluated via the SSSOM mapping instead of LLM classification.
- Ground truth files stripped to risk-level annotations only (107 category-level entries removed across 19 files). Category-level coverage is now derived programmatically from risk-level GT via the SSSOM mapping.
- `--classify-taxonomies` CLI option removed from both `extract` command and battery runner.

### Removed
- `extract/classify.py` module (LLM-based taxonomy classification)
- `classify_risks` prompt templates (`classify_risks_system.j2`, `classify_risks_user.j2`)
- `source_risk_ids` field from `RiskMatch` model
- `"classify"` stage from `LLMCallRecord.stage` literal

### Fixed
- `_infer_taxonomy` prefix for OWASP LLM risks: `"llm0"` → `"llm"` to correctly match `llm102025-unbounded-consumption`.
- `guy-nhs` ground truth: replaced category-level `llm102025-unbounded-consumption` with risk-level `credo-risk-004` (Environmental harm).
- `dhs-gov` ground truth: added `credo-risk-014` (Obscene and sexually abusive content) to make `ail-child-sexual-exploitation` derivable before stripping.
