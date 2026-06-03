# Changelog

## [Unreleased]

### Added
- **Mitigation recommendations**: each extracted risk now includes recommended mitigation actions from 5 frameworks (MIT AI Risk Repository, OWASP LLM Top 10, NIST AI RMF 600-1, Credo UCF, AIUC-1). Pre-built index maps 95 Atlas risks to ~5,190 action entries. Mitigations appear in JSON output (`mitigations` field on `RiskMatch`) and in the HTML report as an expandable section per risk.
- **`scripts/build_mitigation_index.py`**: generates `data/atlas_risk_to_actions.yaml` by reading local MIT/OWASP action files and resolving NIST/Credo/AIUC-1 transitive cross-framework mappings from Nexus. Each action is categorized as `technical`, `operational`, or `governance` via rules in `data/mitigation_categories.yaml`.
- **`data/mitigation_categories.yaml`**: category assignment rules (MIT group → category, NIST RMF prefix → category, AIUC-1 principle → category) plus explicit assignments for OWASP and Credo actions.
- **`data/mit_ai_risk_mitigation_to_atlas_data.yaml`**: maps 831 MIT AI Risk Repository controls to IBM Atlas risk IDs (MIT's own risk-to-mitigation mappings are not yet published; these were generated independently).
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
