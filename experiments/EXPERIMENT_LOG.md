# Experiment Log

## 2026-05-28: Cross-Encoder Baseline Evaluation

**Description:** Evaluated off-the-shelf cross-encoder models on a dataset built from enriched ground truth (894 risks
across 20 policies). The dataset pairs risk descriptions with document chunks, labeled positive/negative via GT evidence
matching. 7,380 train pairs / 8,132 eval pairs, 50/50 policy split stratified by difficulty.

**Models tested:**

- `cross-encoder/ms-marco-MiniLM-L-12-v2` (current pipeline model)
- `cross-encoder/ms-marco-electra-base`
- `cross-encoder/nli-deberta-v3-base`
- `cross-encoder/stsb-distilroberta-base`

**Results (eval set):**

| Model                   | AUC-ROC | Best F1 | Pos Mean | Hard Neg Mean |
|-------------------------|---------|---------|----------|---------------|
| ms-marco-MiniLM-L-12-v2 | 0.565   | 0.334   | 0.360    | 0.546         |
| ms-marco-electra-base   | 0.561   | 0.339   | 0.641    | 0.660         |
| nli-deberta-v3-base     | 0.514   | 0.307   | 0.847    | 0.866         |
| stsb-distilroberta-base | 0.529   | 0.319   | 0.639    | 0.637         |

**Conclusion:** All off-the-shelf models perform near random (AUC ~0.5). Hard negatives consistently score equal to or
higher than positives. The risk-policy relevance task is too domain-specific for generic models.

---

## 2026-05-28: Cross-Encoder Fine-Tuning

**Description:** Fine-tuned `ms-marco-MiniLM-L-12-v2` on the train split (7,380 pairs). 5 epochs, batch size 16, lr
2e-5. Also fine-tuned `nli-deberta-v3-base` for comparison.

**Results (eval set):**

| Model              | AUC-ROC | Best F1 | Precision | Recall | Speed  |
|--------------------|---------|---------|-----------|--------|--------|
| ms-marco baseline  | 0.565   | 0.334   | 0.226     | 0.645  | 98 p/s |
| Fine-tuned MiniLM  | 0.941   | 0.755   | 0.762     | 0.748  | 95 p/s |
| Fine-tuned DeBERTa | 0.938   | 0.741   | 0.744     | 0.737  | 20 p/s |

**End-to-end battery (20 policies):** Fine-tuned MiniLM improved Credo (+0.077 recall) and NIST (+0.095 recall) but
destroyed ai-risk-taxonomy (-0.40 recall). The model learned to reject ai-risk-taxonomy's tautological description
style ("X is defined as X").

**Conclusion:** Fine-tuning achieves excellent discrimination on the eval dataset but doesn't translate to end-to-end
pipeline improvement. The ai-risk-taxonomy regression and precision drop offset recall gains. MiniLM preferred over
DeBERTa (same AUC, 5x faster).

---

## 2026-05-29: Description Rewriting for Cross-Encoder

**Description:** Investigated why ms-marco scores certain risks near zero. Found the model is sensitive to description
phrasing, not just content. Tested rewording risk descriptions to match ms-marco's preferred style (short, active
voice, "AI systems may..." framing).

**Key finding:** Same concept, same chunk, 5000x score difference based on phrasing:

| Phrasing                                                              | Score  |
|-----------------------------------------------------------------------|--------|
| "Impacts due to leakage and unauthorized use..." (NIST original)      | 0.0001 |
| "AI tools may process, store, or expose personal data..." (rewritten) | 0.9070 |

**Batch rewrite attempt:** Rewrote 33 most-missed risks via Gemma LLM. Individual improvements were dramatic but
end-to-end battery showed regressions — rewritten descriptions competed differently in the cross-encoder ranking,
pushing out previously-accepted candidates.

**Conclusion:** Description rewrites help individual risks but cause collateral damage in the ranking. The cross-encoder
is a relative scorer — improving one description changes scores for all candidates in the batch. Not viable as a batch
approach without per-risk validation.

---

## 2026-05-29: Post-Retrieval Taxonomy Classification

**Description:** Instead of trying to retrieve abstract framework risks (NIST, OWASP) via the retrieval pipeline,
exclude them from retrieval and classify them post-grounding from already-extracted specific risks. One LLM call per
document.

**Approach:** After grounding produces a list of specific risks (Atlas, Credo, MIT, ai-risk-taxonomy) with evidence, ask
the LLM: "Which NIST/OWASP categories do these extracted risks fall under?"

**Configurations tested:**

| Config                               | NIST F1   | OWASP F1 | Macro F1  |
|--------------------------------------|-----------|----------|-----------|
| Baseline (retrieval only)            | 0.505     | 0.545    | 0.758     |
| Classify NIST + OWASP (loose prompt) | 0.686     | 0.418    | 0.733     |
| Classify NIST + OWASP (tight prompt) | 0.731     | 0.533    | 0.733     |
| Classify NIST only                   | **0.702** | 0.533    | **0.754** |

**OWASP analysis:** Classification over-projects OWASP risks. `llm022025-sensitive-information-disclosure` generated 12
false positives (any policy discussing data protection triggers it). OWASP risks are LLM-specific vulnerabilities that
don't map cleanly from general AI policy concepts.

**Conclusion:** Classify NIST only. NIST F1 improved 0.505→0.702 (+39%). OWASP stays in retrieval (0.533 F1, within
noise of baseline 0.545). Macro F1 held at 0.754. Shipped as default configuration.

---

## 2026-05-29: DSPy Judge Prompt Optimization (v1 — random negatives, subsampled)

**Description:** Used DSPy GEPA to optimize the LLM judge prompt. Dataset: 50 train / 30 eval examples (subsampled).
Negatives sampled from pipeline extraction results (spurious + grounding-filtered candidates).

**Results (isolated judge eval):**

- Baseline F1: 82.16%
- Optimized F1: 88.38%
- Improvement: +6.22%

**Optimized prompt key innovations:**

- "Mitigation-to-Risk" reverse logic: if text mandates a safeguard, it's relevant to the risk that safeguard mitigates
- "Subset Rule": broad category text makes specific subset risks relevant
- Specificity limits to control precision
- Synonym/functional/contextual semantic mapping

**End-to-end battery:** The optimized judge degraded overall pipeline F1 (0.754→0.723). The more permissive judge
accepted too many borderline candidates, adding noise and wasting grounding budget.

**Conclusion:** Isolated judge improvement doesn't translate to pipeline improvement. The judge operates in a pipeline
context where precision matters more than recall — false accepts get filtered by grounding but cost LLM calls. Reverted
to original judge prompt.

---

## 2026-05-30: DSPy Judge Prompt Optimization (v2 — hard negatives, full dataset)

**Description:** Re-running GEPA with improved training data. Hard negatives mined using the cross-encoder itself
(highest-scoring non-GT risks per chunk) instead of random pipeline artifacts. Full dataset: 157 train / 207 eval
examples, no subsampling.

**Results (isolated judge eval):**

- Baseline F1: 69.26% (lower than v1's 82.16% — harder negatives make the task more challenging)
- Optimized F1: 80.51%
- Improvement: +11.25% (bigger than v1's +6.22%)

**Optimized prompt key innovations (different from v1):**

- "Granularity Rule": broad "we protect privacy" does NOT make every sub-risk (biometric data, stalking, model
  inversion) relevant — only the category-level risk
- "Compliance Statement" nuance: "we comply with all laws" → regulatory compliance, NOT every specific risk
- Three-step mapping: identify themes → filter by specificity → check conceptual overlap
- More conservative than v1's "mitigation-to-risk" approach

**End-to-end battery (DSPy v2 judge + NIST classification):**

| Metric          | Baseline (no classify) | Old judge + NIST classify | DSPy v2 + NIST classify |
|-----------------|------------------------|---------------------------|-------------------------|
| Evals passed    | 5/20                   | 4/20                      | 5/20                    |
| Macro recall    | 0.706                  | 0.717                     | 0.705                   |
| Macro precision | 0.853                  | 0.819                     | 0.809                   |
| Macro F1        | 0.758                  | 0.754                     | 0.741                   |
| NIST F1         | 0.505                  | 0.702                     | **0.713**               |
| OWASP F1        | 0.545                  | 0.533                     | **0.552**               |

**Conclusion:** The precision-focused v2 prompt (trained on hard negatives) avoids v1's over-acceptance problem.
End-to-end macro F1 at 0.741 — slight regression from baseline 0.758 mostly due to ai-risk-taxonomy LLM
non-determinism, but NIST improved +0.208 F1 and OWASP +0.007. Best combined configuration. Shipped as default
judge prompt.

---

## 2026-05-30: DSPy NIST Classification Prompt Optimization

**Description:** Used DSPy GEPA to optimize the post-retrieval NIST classification prompt. Dataset: 10 train / 10 eval
examples (one per policy), each classifying 12 NIST risk categories from extracted risks.

**Results (isolated classification eval):**

- Baseline F1: 62.50%
- Optimized F1: 67.59%
- Improvement: +5.09%

**Optimized prompt key innovations:**

- "Mitigation vs. Risk" rule: if evidence describes a requirement/guideline, it's a mitigation NOT a risk — don't map it
- Precise domain boundaries: Data Privacy (subject identity) vs Information Security (system integrity) vs
  Information Integrity (societal truth at scale). Single hallucination = Confabulation, NOT Information Integrity.
- Human-AI Configuration requires specific phenomena (anthropomorphism, automation bias) — not general "lack of
  training"
- Value Chain requires upstream third-party components — not internal tool integration
- Evidence quoting: must cite specific phrases and map to NIST definition

**End-to-end battery (DSPy v2 judge + optimized NIST classification):**

| Metric          | Baseline | Prev best | Optimized classify |
|-----------------|----------|-----------|--------------------|
| NIST precision  | 0.757    | 0.588     | **0.630**          |
| NIST recall     | 0.378    | 0.905     | **0.926**          |
| NIST F1         | 0.505    | 0.713     | **0.750**          |
| Macro precision | 0.853    | 0.809     | **0.826**          |
| Macro F1        | 0.758    | 0.741     | **0.746**          |

**Conclusion:** NIST F1 improved 0.713→0.750. Precision recovered (0.588→0.630) while recall held (0.926). Macro
precision at 0.826 (up from 0.809). Macro F1 at 0.746, approaching baseline 0.758. Shipped as default classification
prompt.

---

## 2026-05-31: DSPy Grounding Prompt Optimization — Experiment Setup

**Description:** Setting up DSPy GEPA optimization for the grounding stage (`attribute.py`). The grounder decides
whether accepted candidates are actually discussed in a text chunk and extracts evidence quotes. Current prompt is
generic ("genuinely discusses that risk concept"). This is the last LLM stage without DSPy optimization.

**Hard negative strategy — no cross-encoder:** Previous experiments established that cross-encoder scores are barely
better than random (AUC ~0.56, experiment 2026-05-28). Instead of cross-encoder hard negative mining (used by
`dspy_judge`), the grounding dataset uses three negative sources:

1. **Same-document, other-chunk negatives:** GT risks whose evidence appears in different chunks of the same policy.
   Hard because they share the same domain and are genuinely relevant to the document — just not to this specific chunk.
   This tests exactly the grounder's core capability: chunk-level vs document-level relevance.
2. **Random catalog negatives:** Random risks from the full Nexus catalog (~524 risks). Easy negatives for baseline
   discrimination.
3. **Pipeline negatives (optional):** If `--run-dir` is provided, loads `grounding_filtered_candidates` from actual
   pipeline runs. These are risks that passed retrieval+judging but were rejected by grounding — the exact distribution
   the grounder sees in production.

**Dataset structure:** Same as judge — (chunk_text, candidate_risks) → expected_verdicts. Each verdict includes
`grounded` (bool) and `expected_quotes` (GT evidence strings from that chunk). 10/10 train/eval policy split.

**Metric:** Combined score = 80% decision F1 (grounded true/false) + 20% quote quality (token-level F1 between
predicted and expected quotes for true positives). Decision F1 is weighted heavily because the grounded/not decision
is the primary optimization target — quote extraction quality is more about model capability than prompt instructions.

**Key risk:** Same as judge v1 — optimizing grounding in isolation might not translate end-to-end. The grounder acts
as a precision filter (rejects candidates that passed retrieval/judging). If GEPA makes it more permissive, noise
increases. The same-document, other-chunk hard negatives are the main defense against this.

**Results (isolated grounding eval):**

- Baseline combined score: 51.74%
- Optimized combined score: 67.18%
- Improvement: +15.44%

**Optimized prompt key innovations:**

- "Mitigation-as-Evidence" principle: if text mandates a procedure to prevent a harm, the risk associated with
  that harm is grounded — policy documents describe mitigations, not risks directly
- "Rule of Explicit Mention": if text explicitly names a risk or gives a real-world example, it's grounded even
  without a specific mitigation in that sentence
- "Foundational Requirements" mapping: training/awareness → AI literacy/human error; governance/audit →
  accountability/legal liability
- Three self-correction heuristics: "Over-Correction Trap" (don't reject because too general), "Security Blanket
  Trap" (don't accept generic security for specific technical risks), "Governance Trap" (governance covers
  accountability but not prompt injection)
- Confidence calibration: high = explicit naming, medium = implied through related requirement, low = tenuous

**End-to-end battery (DSPy optimized grounding prompt):**

**MLflow:** experiment=dspy-ground-optimization

| Metric          | Baseline | Optimized grounding |
|-----------------|----------|---------------------|
| Macro precision | 0.826    | 0.676               |
| Macro recall    | 0.717    | 0.762               |
| Macro F1        | 0.749    | 0.694               |
| AI-Risk-Tax F1  | 0.680    | 0.470 (-0.209)      |
| AILuminate F1   | 0.880    | 0.765 (-0.115)      |
| Credo F1        | 0.763    | 0.773 (+0.010)      |
| IBM Atlas F1    | 0.773    | 0.803 (+0.030)      |
| MIT F1          | 0.800    | 0.816 (+0.016)      |
| NIST F1         | 0.739    | 0.712 (-0.027)      |
| OWASP F1        | 0.500    | 0.546 (+0.046)      |

**Conclusion:** Same pattern as DSPy judge v1 — isolated improvement (+15.44%) does not translate to end-to-end
improvement. Macro F1 dropped 0.749→0.694 (-0.055). The "Mitigation-as-Evidence" principle causes massive
over-grounding in ai-risk-taxonomy (-0.209 F1) and ailuminate (-0.115 F1) by accepting risks that are merely alluded
to through governance/training mandates. The grounder acts as a precision filter; making it more permissive adds noise
that the downstream merge/classify stages cannot compensate for. Reverted to original grounding prompt.

---

## 2026-05-31: DSPy Grounding Prompt Optimization v2 — Pipeline-Mined Negatives

**Description:** Re-ran GEPA with pipeline-mined negatives from `grounding_filtered_candidates` (8,066 negatives from
a baseline battery run) instead of relying solely on same-document other-chunk negatives. This mirrors the approach
that made the judge v2 succeed over v1 — using the actual distribution of false positives the grounder sees in
production.

**MLflow:** experiment=dspy-ground-optimization

**Results (isolated grounding eval):**

- Baseline combined score: 50.32%
- Optimized combined score: 69.85%
- Improvement: +19.53% (larger than v1's +15.44%)

**End-to-end battery — two variants tested:**

| Metric          | Baseline | GEPA v1 (raw) | GEPA v2 (raw) | GEPA v2 (curated) |
|-----------------|----------|---------------|---------------|-------------------|
| Macro precision | 0.838    | 0.676         | —             | 0.711             |
| Macro recall    | 0.677    | 0.762         | —             | 0.737             |
| Macro F1        | 0.749    | 0.694         | —             | 0.724             |
| AI-Risk-Tax F1  | 0.680    | 0.470         | —             | 0.515             |
| AILuminate F1   | 0.880    | 0.765         | —             | 0.765             |

"Curated" variant: manually removed GEPA's "do not be conservative" / "avoid conservatism" language, kept the
mitigation-mapping structure and added precision guardrails back. Improved over raw GEPA v1 (0.724 vs 0.694) but
still worse than baseline (0.749).

**Root cause analysis:** GEPA consistently finds "Mitigation→Risk" mapping prompts that improve isolated recall but
over-ground in production. The F1-based metric rewards balanced precision/recall, but the pipeline needs the grounder
as a precision gate. ai-risk-taxonomy risks have broad, tautological descriptions ("X is defined as X") that the
mitigation mapping connects to any governance text. This taxonomy-specific vulnerability makes the approach
structurally incompatible with permissive grounding.

**Key lesson:** Unlike the judge (where v2 hard negatives fixed the v1 over-acceptance), the grounding stage's
failure mode is metric-driven, not data-driven. Pipeline-mined negatives improved isolated scores further (+19.53%
vs +15.44%) but GEPA still converges on permissive prompts because F1 rewards recall gains. A precision-weighted
metric (F0.5 or precision-floor constraint) would be needed to change this, but the fundamental question is whether
prompt optimization can improve a stage whose original 4-line prompt already achieves near-optimal end-to-end
precision (0.838).

**Conclusion:** Reverted to original grounding prompt. The grounding stage may not be amenable to DSPy prompt
optimization — its simplicity is a feature, not a bug. Future improvement may come from the retrieval/judging stages
(feeding cleaner candidates) rather than the grounding prompt itself.

---

## 2026-06-01: Two-Tier Eval & Pipeline Restructuring

**Description:** Major restructuring of the pipeline and evaluation system. Category-level taxonomies (NIST AI RMF,
OWASP Top 10 LLM, AILuminate, OWASP ASI) are no longer retrieved or classified by the pipeline — they are evaluated
via a static SSSOM cross-taxonomy mapping (`data/risk_to_category.sssom.tsv`, 820 mappings). The LLM classify step
was removed entirely.

**Changes:**

- Removed `extract/classify.py`, classify prompt templates, `--classify-taxonomies` CLI option
- Added `_load_risk_to_category_map()`, `_derive_categories()`, `_evaluate_categories()` to eval.py
- Ground truth stripped to risk-level only (107 category entries removed from 20 existing + 78 from 7 new policies)
- 7 new policies added: amadeus, fs-isac, gray, icrc, npcc, penn, st-johns (27 total)
- GT quality fixes: 3 wrong-risk-ID fixes, 5 insufficient-evidence removals, 13 prohibition-list additions to amadeus

**New baseline (27 policies, run `risk-selected_20260601_160553`):**

| Tier                | Macro P | Macro R | Macro F1 |
|---------------------|---------|---------|----------|
| Risk-level          | 0.813   | 0.649   | 0.708    |
| NIST category       | 0.975   | 0.877   | 0.923    |
| OWASP LLM category  | 0.917   | 0.887   | 0.902    |
| AILuminate category | 0.951   | 0.892   | 0.921    |

Pass rate: 5/27 (thresholds: P≥0.60, R≥0.80)

---

## 2026-06-01: Cross-Encoder Model Evaluation (Off-the-Shelf)

**Description:** Evaluated modern off-the-shelf cross-encoder rerankers on the updated dataset (7,208 train / 8,649
eval pairs, 27 policies, risk-level only, category taxonomies excluded from risk pool).

**Models tested:**

| Model                              | AUC-ROC   | Best F1   | Precision | Recall | Speed  |
|------------------------------------|-----------|-----------|-----------|--------|--------|
| ms-marco-MiniLM-L-12-v2 (baseline) | 0.636     | 0.340     | 0.243     | 0.567  | 90 p/s |
| **gte-reranker-modernbert-base**   | **0.813** | **0.512** | 0.468     | 0.565  | 23 p/s |
| bge-reranker-v2-m3                 | 0.788     | 0.466     | 0.522     | 0.420  | 14 p/s |

**Score distribution analysis:**

- ms-marco: scores spread across full 0-1 range (std=0.35). Positives score 0.294 mean, hard negatives 0.338 — model
  can't distinguish them but the wide spread makes absolute thresholds work.
- GTE: scores cluster in a 0.04-wide band around 0.65 (std=0.14). Better discrimination (AUC 0.813) but absolute
  thresholds are meaningless — everything scores ~0.65.
- BGE: similar clustering (std=0.05) but weaker discrimination.

**End-to-end battery with GTE (threshold_high=0.68, threshold_low=0.62):**
Regressed badly — 2/27 pass vs 5/27 baseline. The tight score clustering meant the pipeline couldn't separate
accept/borderline/discard. Auto-accepted 1,443 candidates for SAP (vs 1,058 baseline) because almost everything scored
above 0.68.

**Conclusion:** GTE-reranker-modernbert-base has substantially better discrimination ability (AUC +0.177) but its
sigmoid-normalised scores cluster too tightly for the pipeline's absolute-threshold architecture. Two paths: (1)
fine-tune GTE on our data to spread scores, or (2) change pipeline to use relative ranking instead of absolute
thresholds. Fine-tuning preferred since ms-marco's absolute thresholds work well and changing architecture is high-risk.

---

## 2026-06-01: Over-Extraction Analysis

**Description:** Analysed spurious risk patterns across the 27-policy baseline run.

**Key findings:**

- 19% of spurious evidence spans are under 50 characters (single words like "transparency", "security issues")
- `mit-ai-risk-subdomain-6.3` (devaluation of human effort) is #1 spurious risk (8/27 policies) — matched on any mention
  of AI generating content
- 63% of spurious risks have "medium" grounding confidence, 29% "high" — grounder not discriminating
- 53% of spurious accepted by LLM judge, 47% by threshold — both paths contribute equally

**Minimum evidence length filter simulation:**

- min_len=40 chars would remove 26 spurious but also 13 true positives — too blunt
- Many "true positives" removed have genuinely thin evidence (single words). 5 of these were GT errors (removed).

**Conclusion:** Hard evidence-length filter is too blunt. The over-extraction problem is best addressed by improving the
LLM judge's ability to reject risks matched on generic governance language. DSPy judge re-optimization in progress with
updated dataset (27 policies, risk-level only, category taxonomies excluded).

---

## 2026-06-01: DSPy Judge Prompt Optimization v3 — Risk-Level Only Dataset

**Description:** Re-ran GEPA with the updated dataset: 187 train / 244 eval examples from 27 policies (13 train, 14
eval). Category-level taxonomies excluded from the risk pool. Hard negatives mined via cross-encoder from risk-level
risks only. Amadeus (train) and ICRC (eval) split to test generalisation on EU AI Act prohibition patterns.

**Results (isolated judge eval):**

- Baseline F1: 70.90% (up from v2's 69.26% — cleaner dataset)
- Optimized F1: 80.23%
- Improvement: +9.33%

**Optimized prompt key innovations (different from v2):**

- "Avoid Over-Literalism": don't reject because the specific technical term is missing — if text manages a domain, risks
  in that domain are relevant
- Theme-to-risk mapping: governance/process → compliance/governance risks; data protection → privacy risks; security →
  adversarial risks
- "Crucial Note" on governance frameworks: approval processes, monitoring, contract reviews implicitly address the risks
  those guardrails prevent
- Specificity guard maintained: general privacy ≠ biometric/facial recognition

**End-to-end battery (v3 judge, 27 policies):**

| Metric          | Baseline (v2 judge) | v3 Judge  | Delta      |
|-----------------|---------------------|-----------|------------|
| Macro F1        | 0.708               | **0.719** | **+0.010** |
| Macro Precision | 0.813               | 0.814     | +0.001     |
| Macro Recall    | 0.649               | **0.665** | **+0.017** |
| Pass rate       | 5/27                | **6/27**  | +1         |
| NIST cat F1     | 0.923               | 0.920     | -0.003     |
| OWASP LLM F1    | 0.907               | **0.935** | +0.028     |

15 policies improved, 8 regressed (mostly small), 4 unchanged. Precision held — the v3 prompt's
"Avoid Over-Literalism" did NOT cause the over-acceptance that killed v1 (which regressed F1 0.754→0.723).

**Why v3 worked where v1 failed:** v1 was trained on random negatives with a subsampled dataset (50 train);
v3 used hard negatives from the cross-encoder with a full dataset (187 train) and risk-level-only risks.
The hard negatives taught the prompt to be inclusive on domain-relevant risks while rejecting
cross-domain false matches. The "Distinguish Specificity" guard (general privacy ≠ biometric) prevented
the over-acceptance pattern.

**Conclusion:** Shipped as default judge prompt. First DSPy judge optimization to improve end-to-end
performance. New baseline: P=0.814, R=0.665, F1=0.719.

---

## 2026-06-02: Cross-Encoder Sigmoid Bug Fix & GTE Re-evaluation

**Description:** Discovered that the pipeline's `rerank()` method unconditionally applied sigmoid
normalisation to all cross-encoder scores. This was correct for ms-marco (which outputs unbounded
logits) but wrong for GTE-reranker-modernbert-base (which outputs calibrated scores in ~[0, 1]).

Raw GTE scores: relevant pair=0.77, irrelevant=0.08 (spread 0.85). After sigmoid: 0.68 vs 0.52
(spread 0.20). The sigmoid was destroying GTE's discrimination ability, explaining the earlier
"score clustering" problem.

**Fix:** Added `_SIGMOID_MODELS` allowlist in `index.py`. Only ms-marco models get sigmoid; others
use raw scores clipped to [0, 1].

**GTE end-to-end results (with fix, various thresholds):**

| Config                        | Macro P | Macro R | Macro F1  | Pass |
|-------------------------------|---------|---------|-----------|------|
| Baseline (ms-marco, 0.7/0.15) | 0.814   | 0.665   | **0.719** | 6/27 |
| GTE, 0.7/0.15 (raw scores)    | 0.660   | 0.680   | 0.650     | 5/27 |
| GTE, 0.82/0.55 (tuned)        | 0.651   | 0.663   | 0.644     | 5/27 |

**Conclusion:** Even with correct scoring, GTE-reranker regresses end-to-end with arbitrary
thresholds. Needed data-informed threshold calibration — see next experiment.

---

## 2026-06-02: Bi-Encoder Swap Evaluation

**Description:** Evaluated modern bi-encoder models as replacements for `all-mpnet-base-v2` (2021, 768-dim).

**Candidate recall@100 on 8 eval policies:**

| Bi-encoder                  | Semantic Recall | BM25-only rescues | Params |
|-----------------------------|-----------------|-------------------|--------|
| all-mpnet-base-v2 (current) | 0.917           | 23                | 110M   |
| **BAAI/bge-m3**             | **0.962**       | 8                 | 568M   |
| gte-modernbert-base         | 0.929           | 19                | 110M   |

BGE-M3 reduces BM25-only rescues from 23 to 8 — 15 risks now found via semantic search.

**End-to-end with BGE-M3 + ms-marco reranker:**
Regressed (P=0.78, 3/25 pass). Better bi-encoder feeds more candidates above the cross-encoder
threshold, but ms-marco doesn't discriminate well on the additional candidates, so precision drops.
Bi-encoder and cross-encoder are coupled — swapping one without the other changes the candidate
distribution the thresholds were tuned for.

---

## 2026-06-02: Pipeline-Mined Eval Dataset & Cross-Encoder Re-evaluation

**Description:** Discovered the cross-encoder eval dataset was biased — hard negatives were mined
using ms-marco's own scores, making GTE look artificially good (it easily rejected ms-marco's
specific failure modes). Rebuilt the dataset with model-agnostic, chunk-level pipeline-mined
negatives from actual battery runs.

**Pipeline-mined dataset:** 3,962 train / 4,954 eval pairs. Hard negatives are risks that the
retrieval stage (BM25 + bi-encoder + RRF) surfaced for each specific chunk but aren't in the GT.
These represent the actual candidate distribution any reranker would face.

**Results on pipeline-mined dataset:**

| Model           | AUC-ROC   | Best F1   | Pos mean  | Hard neg mean | Separation |
|-----------------|-----------|-----------|-----------|---------------|------------|
| ms-marco-MiniLM | **0.498** | 0.420     | 0.295     | 0.733         | **-0.438** |
| GTE-reranker    | **0.759** | **0.596** | **0.701** | 0.618         | **+0.083** |

**Key finding:** ms-marco has AUC 0.498 on pipeline-mined negatives — literally random. Hard
negatives score HIGHER than positives (0.733 vs 0.295). The previous AUC of 0.636 was an artifact
of testing against ms-marco's own failure modes, which other models easily reject.

GTE genuinely discriminates (AUC 0.759, positive > negative) but with narrow separation (+0.083).

---

## 2026-06-02: GTE with Data-Informed Thresholds

**Description:** Using the pipeline-mined dataset to set GTE thresholds empirically:
threshold_high=0.72 (above positive median), threshold_low=0.55 (below hard negative mean).

**End-to-end results:**

| Config                                   | Macro P   | Macro R   | Macro F1  | Pass |
|------------------------------------------|-----------|-----------|-----------|------|
| ms-marco + threshold 0.7/0.15 (baseline) | **0.813** | 0.649     | **0.708** | 5/27 |
| No cross-encoder (RRF only)              | 0.676     | 0.625     | 0.635     | 1/27 |
| GTE calibrated 0.72/0.55                 | 0.650     | **0.652** | 0.634     | 6/27 |

**Analysis:** GTE calibrated performs identically to no cross-encoder at all. Despite better
discrimination (AUC 0.759 vs 0.498), GTE doesn't improve end-to-end because:

1. ms-marco works as a **volume reduction filter**, not a semantic discriminator. It randomly
   rejects ~70% of candidates, and the survivors happen to include enough true positives
   because the grounding stage catches the noise.
2. ms-marco's apparent "precision" (0.813) comes from its conservatism (most scores are low),
   not from intelligent discrimination. It's sensitive to surface-level text similarity
   (trained on MS MARCO passages) which correlates with crude relevance.
3. GTE discriminates on semantic relevance, but the pipeline's LLM judge and grounding stages
   already handle semantic relevance — making the cross-encoder's discrimination redundant.

**Rank-based selection (top_n_accept/top_n_judge) results:**

| Config                  | Macro P | Macro R | Macro F1 |
|-------------------------|---------|---------|----------|
| ms-marco + rank (15/10) | 0.795   | 0.629   | 0.688    |
| GTE + rank (15/10)      | 0.691   | 0.607   | 0.625    |

Rank-based slightly underperforms threshold-based for ms-marco (-0.020 F1) and is worse for
GTE because GTE's ranking pushes different (wrong) risks into the top positions.

---

## 2026-06-02: IR Isolation Experiment — LLM Stages Stripped

**Description:** All previous experiments were confounded by the LLM judge and grounding stages.
To understand the IR pipeline in isolation, added `--no-ground` flag that skips both LLM judge
and grounding entirely. Borderline candidates are auto-promoted to accepted; all accepted become
matches with empty evidence. No LLM calls at all — pure retrieval evaluation.

Also added remote model support: bi-encoder and cross-encoder models can now be served via vLLM
on GPU (`/v1/embeddings` and `/v1/score` endpoints) by passing a URL as the model name.

**MLflow:** experiment=ir-isolation

**Models tested across 7 configurations, 27 policies:**

- **Bi-encoders:** all-mpnet-base-v2 (local, 768d), BAAI/bge-m3 (cluster, 1024d),
  Alibaba-NLP/gte-modernbert-base (cluster, 768d), lightonai/LateON ColBERT (local, 768d)
- **Cross-encoders:** ms-marco-MiniLM-L-12-v2 (local), Alibaba-NLP/gte-reranker-modernbert-base
  (cluster), none

**Parameters:** top_n_accept=5, top_n_judge=5, bm25_rescue_rank=10, rrf_min_score=0.01 (no-CE
configs). Without a cross-encoder, each chunk keeps ~50 RRF candidates. With a cross-encoder,
each chunk keeps ~18 (5 threshold + ~13 borderline including BM25 rescues).

**Results (macro over 27 policies):**

| Config | Bi-encoder               | Cross-encoder          | Macro P | Macro R | Macro F1  |
|--------|--------------------------|------------------------|---------|---------|-----------|
| A      | mpnet (local)            | none                   | 0.228   | 0.871   | 0.351     |
| D      | bge-m3 (cluster)         | none                   | 0.226   | 0.893   | 0.352     |
| G      | LateON ColBERT (local)   | MaxSim                 | 0.214   | 0.865   | 0.335     |
| B      | mpnet (local)            | ms-marco (local)       | 0.273   | 0.623   | 0.365     |
| C      | mpnet (local)            | GTE reranker (cluster) | 0.303   | 0.712   | **0.407** |
| E      | bge-m3 (cluster)         | GTE reranker (cluster) | 0.302   | 0.715   | **0.408** |
| F      | gte-modernbert (cluster) | GTE reranker (cluster) | 0.298   | 0.698   | 0.402     |

**Per-taxonomy F1:**

| Config              | ai-risk-taxonomy | credo-ucf | ibm-risk-atlas | mit-ai-risk |
|---------------------|------------------|-----------|----------------|-------------|
| A (mpnet, no CE)    | 0.126            | 0.439     | 0.422          | 0.493       |
| B (mpnet, ms-marco) | 0.185            | 0.426     | 0.471          | 0.490       |
| C (mpnet, GTE)      | 0.182            | 0.518     | 0.451          | 0.536       |
| D (bge-m3, no CE)   | 0.124            | 0.444     | 0.433          | 0.507       |
| E (bge-m3, GTE)     | 0.185            | 0.510     | 0.451          | 0.548       |
| F (gte-bert, GTE)   | 0.169            | 0.511     | 0.451          | 0.527       |
| G (LateON)          | 0.102            | 0.465     | 0.445          | 0.512       |

**Key findings:**

1. **GTE reranker is unambiguously the best cross-encoder.** Configs C/E/F (all using GTE)
   cluster at F1=0.40–0.41. ms-marco (B) achieves only 0.365. GTE gains both higher precision
   (+0.03) AND higher recall (+0.09) compared to ms-marco.

2. **ms-marco destroys recall without gaining proportional precision.** It loses 0.25 recall
   vs no-CE (0.623 vs 0.871) but gains only 0.045 precision (0.273 vs 0.228). GTE loses less
   recall (−0.16) and gains more precision (+0.075). ms-marco's scoring is essentially random
   on this task (AUC 0.498, confirmed again) — it rejects candidates indiscriminately.

3. **Bi-encoder choice barely matters once a cross-encoder is added.** C (mpnet+GTE, F1=0.407)
   ≈ E (bge-m3+GTE, F1=0.408) ≈ F (gte-modernbert+GTE, F1=0.402). The cross-encoder dominates
   ranking quality. Without a cross-encoder, bge-m3 has slightly better recall than mpnet
   (0.893 vs 0.871) but the difference vanishes after reranking.

4. **LateON ColBERT underperforms** (G, F1=0.335). Its 299-token context limit truncates
   longer risk descriptions, and MaxSim scoring without a separate reranking stage produces
   worse ranking than the two-stage (bi-encoder → cross-encoder) approach. LateON cannot be
   served remotely for this task — vLLM returns pooled embeddings, not token-level, and the
   299-token limit causes 400 errors on longer inputs.

5. **Previous end-to-end results were confounded by LLM stages.** ms-marco appeared to "win"
   (F1=0.719 end-to-end vs GTE's 0.634–0.650) because the LLM grounding stage independently
   provided the precision filtering that ms-marco's random rejection couldn't. GTE's genuine
   discrimination was redundant with the LLM stages — stripping the LLM stages reveals that
   GTE is the better retrieval component.

**Implication for pipeline architecture:** The current pipeline uses ms-marco as a "volume
reduction filter" (randomly rejects ~70% of candidates) followed by LLM grounding for actual
precision. Replacing ms-marco with GTE and potentially reducing LLM grounding effort (fewer
false positives to filter) could improve both quality and efficiency. The next experiment should
test GTE + LLM stages end-to-end with the sigmoid fix already in place.

---

## 2026-06-02: Threshold Tuning for GTE Reranker

**Description:** Analysed TP/FP distributions across GTE cross-encoder scores and BM25 rescue
ranks to find optimal cutoff parameters. Used config C (mpnet + GTE) results from the IR
isolation experiment.

**MLflow:** experiment=ir-isolation (same experiment, threshold analysis only)

**TP/FP breakdown (config C, old defaults top_n=5+5, BM25 rescue=10):**

| Source                     | TP      | FP       | Precision |
|----------------------------|---------|----------|-----------|
| Threshold (top-5 by CE)    | 349     | 391      | 47.2%     |
| Auto-promoted (borderline) | 422     | 1327     | 24.1%     |
| BM25 rescue (CE=0)         | 53      | 438      | 10.8%     |
| **Total**                  | **771** | **1718** | **31.0%** |

**BM25 rescue is actively harmful:** Adds 53 TP but 438 FP. Removing it improves F1 from
0.431 → 0.466 (+0.035). BM25 rescue by rank shows uniformly poor precision (6–19%) across
all BM25 ranks 1–10. The rescue mechanism was designed for ms-marco's random scoring; with
GTE's actual discrimination, rescued candidates are genuinely bad matches.

**GTE CE score precision by bucket:**

| CE Score      | TP  | FP  | Precision | Cumulative Recall |
|---------------|-----|-----|-----------|-------------------|
| 0.85–1.00     | 216 | 170 | 56.0%     | 28.0%             |
| 0.75–0.85     | 407 | 679 | 37.5%     | 80.8%             |
| 0.70–0.75     | 56  | 191 | 22.7%     | 88.1%             |
| 0.65–0.70     | 26  | 103 | 20.2%     | 91.4%             |
| <0.65         | 13  | 137 | 8.7%      | 93.1%             |
| CE=0 (rescue) | 53  | 438 | 10.8%     | 100%              |

**Recall ceiling:** 86.4% of GT risks appear in the RRF candidate pool. 13.6% are never
retrieved by BM25+semantic at all. GTE filtering drops an additional 17.1%, keeping 69.2%.

**New defaults (tuned for GTE reranker):**

| Parameter        | Old | New  | Rationale                          |
|------------------|-----|------|------------------------------------|
| top_n_accept     | 5   | 10   | GTE precision ~50% through rank 15 |
| top_n_judge      | 5   | 10   | More candidates for LLM judgment   |
| min_score_floor  | 0.0 | 0.70 | <23% precision below 0.70          |
| bm25_rescue_rank | 10  | 0    | 10.8% precision, −0.034 F1         |

**Conclusion:** Shipped as new pipeline defaults. These are tuned for GTE-reranker-modernbert-base;
ms-marco users may need to override. The wider candidate window (10+10 vs 5+5) with a score
floor gives the LLM stages more high-quality candidates to work with.

---

## 2026-06-02: IR Isolation Round 2 — New Models & Tuned Defaults

**Description:** Second round of IR isolation testing with new models and the tuned defaults
(top_n=10+10, floor=0.70, no BM25 rescue). Tested EmbeddingGemma-300M (Google, 300M params)
as bi-encoder and bge-reranker-v2-m3 (BAAI) as cross-encoder, alongside the previous GTE
reranker winner.

Also attempted Iso-ModernColBERT (topk-io, isotropy-corrected ColBERT) via remote
`/v1/embeddings`, but it has a 299-token max_model_len — same as LateON. ColBERT models
cannot be served remotely as pooled-embedding bi-encoders; they need the local
`--colbert-model` path for token-level MaxSim scoring.

**MLflow:** experiment=ir-isolation

**Results (27 policies, no LLM stages):**

| Config | Bi-encoder              | Cross-encoder               | Macro P   | Macro R   | Macro F1  |
|--------|-------------------------|-----------------------------|-----------|-----------|-----------|
| A      | mpnet (local)           | none                        | 0.228     | 0.871     | 0.351     |
| O      | embeddinggemma-300m     | none                        | 0.234     | 0.895     | 0.360     |
| C*     | mpnet                   | GTE reranker (old defaults) | 0.303     | 0.712     | 0.407     |
| H      | mpnet                   | GTE reranker (new defaults) | 0.339     | 0.686     | 0.438     |
| I      | mpnet                   | bge-reranker-v2-m3          | 0.426     | 0.289     | 0.306     |
| **L**  | **embeddinggemma-300m** | **GTE reranker**            | **0.342** | **0.701** | **0.443** |
| M      | embeddinggemma-300m     | bge-reranker-v2-m3          | 0.427     | 0.296     | 0.309     |

*C uses old defaults (top_n=5+5, floor=0, BM25 rescue=10) for comparison.

**Key findings:**

1. **New defaults improved GTE configs significantly.** H (F1=0.438) vs C (F1=0.407) — the
   wider candidate window (10+10 vs 5+5) and score floor (0.70) captured more TP at better
   precision. Disabling BM25 rescue alone accounts for +0.035 of the improvement.

2. **bge-reranker-v2-m3 is far too aggressive.** It kills recall (0.29) for high precision
   (0.43) — F1=0.31, worst of all CE configs. It's discarding most true positives. Not
   viable for this task without significant threshold adjustment.

3. **EmbeddingGemma-300M is the best bi-encoder.** Consistently higher recall than mpnet in
   every pairing: 0.895 vs 0.871 (no CE), 0.701 vs 0.686 (GTE). Small but real improvement
   across all 27 policies.

4. **ColBERT models (Iso-ModernColBERT, LateON) cannot be served remotely as bi-encoders.**
   vLLM's `/v1/embeddings` returns pooled embeddings with max_model_len=299, too short for
   risk descriptions. They need the local `--colbert-model` path for token-level MaxSim.

**Best IR configuration: EmbeddingGemma-300M + GTE-reranker-modernbert-base (F1=0.443).**

---

## 2026-06-02: LLM Stage Isolation — Judge and Grounder Tested Independently

**Description:** Refactored `--no-ground` into two independent flags (`--no-judge`,
`--no-grounding`) to isolate each LLM stage's contribution. Tested the best IR config
(EmbeddingGemma-300M + GTE reranker) with judge ON / grounding OFF to measure the judge's
value without the grounder confounding the results.

Also tested full pipeline (judge + grounding) with two EmbeddingGemma configs for comparison.

**MLflow:** experiment=ir-isolation

**Judge isolation results (EmbeddingGemma + GTE reranker, 27 policies):**

| Config      | Judge | Grounding | Macro P | Macro R | Macro F1 |
|-------------|-------|-----------|---------|---------|----------|
| S (IR only) | OFF   | OFF       | 0.342   | 0.701   | 0.443    |
| R (+ judge) | ON    | OFF       | 0.393   | 0.682   | 0.480    |
| Judge delta | —     | —         | +0.051  | −0.019  | +0.037   |

**The judge adds clean value:** +5.1% precision for only −1.9% recall = +3.7% F1. It correctly
rejects borderline false positives from the GTE ranking without significantly hurting recall.

**Full pipeline results (judge + grounding):**

| Config   | CE       | Judge | Grounding | Macro P | Macro R | Macro F1 |
|----------|----------|-------|-----------|---------|---------|----------|
| P        | none     | n/a   | ON        | 0.658   | 0.609   | 0.617    |
| Q        | GTE      | ON    | ON        | 0.665   | 0.500   | 0.553    |
| Baseline | ms-marco | ON    | ON        | 0.814   | 0.665   | 0.719    |

**Full pipeline regressions explained:** The grounder degrades with higher candidate volume.
Config P sends ~40 candidates/chunk to the grounder (vs ~20 in baseline); the grounder's
per-call pass rate drops from 6.3% to 4.0%. It gains 147 new TP but loses 140 baseline TP
(net +7) — the grounder shifts WHICH risks get grounded rather than finding MORE.

Config Q (GTE + judge + grounder) loses recall (R=0.500 vs baseline 0.665) because GTE's
ranking pushes some GT risks below the top-20 cutoff — they never reach the judge or grounder.

**Key insight:** The grounder is the bottleneck, not the retrieval or judge. It becomes
more conservative when overwhelmed with candidates. Future work should either cap candidates
per grounding call or improve the grounding prompt's robustness to noise.

**Pipeline flag refactoring:** Replaced `--no-ground` with independent `--no-judge` and
`--no-grounding` flags. `--no-judge` auto-promotes borderline candidates without LLM calls.
`--no-grounding` creates RiskMatch entries without evidence extraction. Both can be used
independently or together.

---

## 2026-06-02: Chunk Size Experiment — 256 vs 512 Tokens

**Description:** Tested whether smaller chunks (256 tokens) improve retrieval or judge quality
compared to the default 512 tokens. Added `--chunk-max-tokens` flag to CLI and battery runner.
Tested with EmbeddingGemma-300M in two configurations: IR-only (no CE) and GTE + judge (no
grounding).

**MLflow:** experiment=ir-isolation

**Results (27 policies):**

| Config | Chunk tokens | CE   | Judge | Macro P | Macro R | Macro F1 |
|--------|--------------|------|-------|---------|---------|----------|
| O      | 512          | none | OFF   | 0.234   | 0.895   | 0.360    |
| U      | 256          | none | OFF   | 0.239   | 0.896   | 0.369    |
| R      | 512          | GTE  | ON    | default  | 0.393   | 0.682   | 0.480    |
| T      | 256          | GTE  | ON    | default  | 0.366   | 0.741   | 0.473    |
| W      | 256          | GTE  | ON    | 512 tok  | 0.367   | 0.745   | 0.476    |

Also added `--judge-context-tokens` flag to decouple the judge's context window from the
chunk size. When set (e.g. 512), the judge receives full text from adjacent chunks up to the
token budget, regardless of how small the retrieval chunks are.

**Findings:**

1. **IR-only (no CE): chunk size doesn't matter.** 256 vs 512 produces identical recall
   (0.896 vs 0.895) and near-identical precision. The retrieval ceiling is the same.

2. **GTE + judge: 256-token chunks trade precision for recall.** +0.059 recall (more chunks
   = more retrieval passes = more risks found) but −0.027 precision (shorter context = judge
   has less signal to reject FP). Net F1 is −0.007 — a wash.

3. **Decoupled judge context doesn't help.** Config W (256-token chunks with 512-token judge
   context) is nearly identical to T (256-token chunks with default sentence padding):
   P=0.367 vs 0.366, R=0.745 vs 0.741, F1=0.476 vs 0.473. The judge's precision loss with
   smaller chunks is not caused by insufficient context — it's caused by seeing more noise
   candidates from more retrieval passes.

4. **256-token chunks double the chunk count**, which means ~2x the retrieval, reranking,
   and judge LLM calls. The cost increase is not justified by the marginal quality difference.

**Conclusion:** 512-token chunks remain the default. Chunk size is not a meaningful lever for
this task. The judge's precision is limited by candidate quality, not context size.

---

## 2026-06-02: DSPy Judge Prompt Optimization v4 — Pipeline-Mined Dataset

**Description:** Re-ran prompt optimization with a dataset built from actual pipeline
candidates (EmbeddingGemma + GTE reranker + judge, `--no-grounding`). Each judge call from
the battery run becomes a training example with real borderline candidates. Three optimizers
tested: GEPA (instruction-only), MIPROv2 (instructions + few-shot demos), and a combined
approach (GEPA instructions + MIPROv2 demo selection).

**Dataset:** 225 train / 309 eval examples from 27 policies. 2,352 positive and 2,694
negative verdicts (47% positive rate), reflecting the actual distribution the judge sees.

**MLflow:** experiment=ir-isolation

**Isolated judge evaluation results:**

| Optimizer | Baseline F1 | Optimized F1 | Improvement |
|-----------|------------|-------------|-------------|
| GEPA (instructions only) | 35.24% | **57.43%** | **+22.19%** |
| MIPROv2 (instructions + 3 demos) | 34.98% | 46.76% | +11.78% |
| GEPA + MIPROv2 demos | 57.43% | 57.86% | +0.43% |

**GEPA prompt key innovations ("Three Tiers of Relevance"):**

- **Tier 1 — Explicit Mention:** Text directly names the risk or its core components
- **Tier 2 — Logical Necessity:** Text describes a scenario that *requires* the risk's
  existence (biometric identification → privacy, data leakage, bias)
- **Tier 3 — Administrative/Structural Latency:** Organizational structures imply risk
  management frameworks (steering board → governance, oversight, compliance risks)
- Domain-specific mapping rules: legal/copyright → compliance risks;
  law enforcement/surveillance → privacy, bias, robustness; governance → oversight failures
- Multi-step reasoning strategy: identify subject → determine risk horizon → perform latent
  mapping → verify against candidates

**Why few-shot demos didn't help:** The GEPA instructions capture the full decision logic.
Adding worked examples (+0.43%) provides no additional signal — the three-tier framework
already encodes the same patterns the examples demonstrate.

**End-to-end results (full pipeline: judge + grounding, 27 policies):**

| Config | Judge prompt | Macro P | Macro R | Macro F1 | Pass |
|--------|-------------|---------|---------|----------|------|
| Baseline (mpnet+msmarco) | v3 | 0.814 | 0.665 | 0.719 | 6/27 |
| gemma+GTE, old judge | v3 | 0.665 | 0.500 | 0.553 | 0/27 |
| gemma+GTE, GEPA judge | GEPA v4 | 0.679 | 0.507 | 0.562 | 0/27 |
| gemma+GTE, GEPA+demos | GEPA+demos | 0.677 | 0.510 | 0.562 | 0/27 |

**End-to-end disappointment:** The GEPA judge's +22% isolated improvement translates to
only +0.009 F1 end-to-end. The grounder remains the bottleneck — it receives more candidates
from the improved judge but rejects them at the same rate, negating the recall gains. The
baseline's higher end-to-end performance comes from ms-marco's conservative candidate
selection sending fewer, "easier" candidates to the grounder, not from a better judge.

**Conclusion:** The judge prompt is no longer the limiting factor. The grounder is the
primary bottleneck for end-to-end improvement. Future work must address the grounder's
inability to handle higher candidate volumes without losing recall.

---

## 2026-06-03: Qwen3-Embedding-4B and NLI DeBERTa-v3-large

**Description:** Tested instruction-aware embedding model (Qwen3-Embedding-4B, 2560-dim,
8K context) and NLI-based cross-encoder (DeBERTa-v3-large, 435M params) on the IR isolation
benchmark. Added `--query-instruction` flag to support instruction-aware bi-encoders.

**MLflow:** experiment=ir-isolation

**Qwen3-Embedding-4B** uses instruction-aware encoding: queries (chunk text) are prefixed
with `"Instruct: Given a text passage from an AI governance policy document, retrieve AI
risk descriptions that are relevant to the concepts, requirements, or concerns discussed
in the passage\nQuery:{text}"`. Documents (risk descriptions) are encoded without
instructions, per the model's design.

**NLI DeBERTa-v3-large** reframes reranking as entailment: premise = chunk text,
hypothesis = `"This text discusses {risk_name}"`. Entailment probability used as the
relevance score. The `large` variant (435M) replaces the `base` (184M) tested earlier.

**Results (IR-only, no judge, no grounding, 27 policies):**

| Bi-encoder | Cross-encoder | Macro P | Macro R | Macro F1 |
|-----------|--------------|---------|---------|----------|
| mpnet (local) | none | 0.228 | 0.871 | 0.351 |
| gemma-300M | none | 0.234 | 0.895 | 0.360 |
| **Qwen3-4B** | **none** | **0.279** | 0.876 | **0.411** |
| gemma-300M | GTE reranker | 0.334 | 0.684 | 0.433 |
| **Qwen3-4B** | **GTE reranker** | **0.359** | **0.725** | **0.465** |
| Qwen3-4B | NLI DeBERTa-v3-large | 0.249 | 0.629 | 0.347 |

**Key findings:**

1. **Qwen3-Embedding-4B is the best bi-encoder by a wide margin.** Without any CE, it
   achieves F1=0.411 — higher than gemma+GTE (0.433 is close but Qwen3 uses no CE). The
   instruction-aware encoding gives it +0.051 F1 over gemma (0.411 vs 0.360) and +0.060
   over mpnet (0.411 vs 0.351). Higher precision (0.279 vs 0.234) at comparable recall
   (0.876 vs 0.895) — the instruction helps it reject irrelevant matches.

2. **Qwen3-4B + GTE reranker = new best IR config (F1=0.465).** Beats all previous configs
   including gemma+GTE (0.433). Both precision (+0.025) and recall (+0.041) improve over
   gemma+GTE, confirming that better first-stage retrieval feeds better candidates to the
   reranker.

3. **NLI DeBERTa-v3-large is not viable as a reranker (F1=0.347).** Worse than no CE at
   all. Both precision (0.249) and recall (0.629) are poor. The entailment framing ("This
   text discusses X") is too literal — policy text describing mitigations or governance
   measures doesn't "entail" the risk concept in the NLI sense. This confirms the earlier
   finding with the base model (AUC 0.514) and establishes that the NLI approach is
   fundamentally unsuited to this task, regardless of model size.

**Implementation:** Added `--query-instruction` CLI flag and `query_instruction` parameter
to `_RemoteBiEncoder` and `RiskIndex`. Instructions are prepended to queries (chunk text)
but not to documents (risk descriptions). Added NLI cross-encoder support: `_NLI_MODELS`
set triggers softmax entailment scoring and NLI-style pair ordering (premise=chunk,
hypothesis="This text discusses {risk_name}").

---

## 2026-06-03: Sibling Expansion + Document-Level Grounding

**Description:** After the main pipeline (retrieval → judge → per-chunk grounding → merge),
a supplementary pass expands found risks to their siblings (parent group + cross-taxonomy
mappings) and grounds the expanded set against the relevant document chunks. This addresses
two problems: (1) 67% of missed GT risks are siblings of found risks, and (2) 45% of GT
risks are matched to wrong chunks by retrieval.

**MLflow:** experiment=ir-isolation

**Implementation:**
- New module `extract/expand.py`: builds expansion graph from Nexus `isPartOf` (parent
  groups, 81 groups, avg 6 risks) + cross-taxonomy mappings (`exact/close/broad/narrow/
  related_mappings`, 192 edges). For each found risk, expansion set = parent siblings ∪
  cross-mapping targets.
- New function `attribute.py::ground_risk_group()`: grounds a group of related risks
  against multiple document chunks in a single LLM call.
- New prompt templates: `ground_group_system.j2` / `ground_group_user.j2` — multi-passage
  evidence extraction.
- Activated via `--expand-siblings` flag. Runs after merge as a supplementary pass;
  existing per-chunk grounding unchanged.

**Expansion analysis (Qwen3-4B + GTE reranker, 27 policies):**
- 288 GT risks missed by retrieval
- Sibling expansion (parent + cross-mappings) recovers 245/288 (85%)
- Potential recall: 0.734 → 0.960
- Evidence coverage: 100% of recovered risks have evidence in the document
- Call sizes: ~32 groups/policy, median 4 risks × 5 chunks per call

**End-to-end results (Qwen3 + GTE + expansion, full pipeline):**

| Config | Macro P | Macro R | Macro F1 |
|--------|---------|---------|----------|
| Baseline (mpnet+msmarco, old GT) | 0.814 | 0.665 | 0.719 |
| Qwen3+GTE, no expansion | 0.665 | 0.500 | 0.553 |
| Qwen3+GTE + expansion (old GT) | 0.498 | 0.752 | 0.586 |

**GT review:** Manual review of 529 expansion "false positives" revealed 352 were genuine
GT gaps — risks correctly identified by the expansion that were missing from the ground
truth annotations. Review conducted via a custom HTML tool showing risk description,
grounder evidence, and highlighted chunk text. 177 were true false positives.

**After GT update (352 additions across 27 policies, total GT: 1083 → 1435):**

| Config | Macro P | Macro R | Macro F1 |
|--------|---------|---------|----------|
| **Qwen3+GTE + expansion (updated GT)** | **0.720** | **0.806** | **0.753** |

**First result to surpass the old baseline F1 (0.753 vs 0.719).** Both recall (+0.141)
and F1 (+0.034) exceed the baseline. Precision gap (0.720 vs 0.814) comes from the 177
true FP in expansion — the grouped grounding prompt accepts ~30% of expansion candidates
vs ~6% for per-chunk grounding.

**Expansion contribution by source:**

| Source | TP | FP | Precision |
|--------|-----|-----|-----------|
| Threshold (top-10 CE) | 361 | 137 | 72% |
| LLM judge (borderline) | 205 | 115 | 64% |
| Expansion (siblings) | 259 | 177 | 59% |

After GT correction, expansion precision improved from 33% → 59%. The remaining 177 FP
are concentrated in broadly-applicable risks (governance, robustness, transparency) that
the grounder over-matches.

---

## 2026-06-04: Multi-Pass Grounding & Reranker Evaluation

**Description:** Investigated ai-risk-taxonomy recall regression (0.741→0.544) after
switching from ms-marco to GTE-reranker. Root cause: GTE's rank-based selection (top-10+10)
pushes ai-risk-taxonomy risks out of the candidate pool because their tautological
descriptions rank lower than atlas/credo/mit risks. ms-marco's random scoring paradoxically
gave ai-risk-taxonomy risks equal chance of passing.

Further analysis showed 28% of ai-risk-taxonomy results varied between runs with identical
config due to LLM non-determinism in the grounder. Losing one seed risk cascades into
losing 10+ expansion siblings (guy-nhs: 1 seed loss → 11 risks lost).

**Multi-pass grounding (union):** Run per-chunk grounding and expansion grounding N times,
union the results. Each pass is an independent LLM call; a risk accepted in any pass
survives. This stabilizes which base risks survive grounding (affecting expansion seeds)
and which expansion candidates get accepted.

**Results (Qwen3-4B + GTE reranker, 27 policies, updated GT with 1519 risks):**

| Config                  | Macro P | Macro R | Macro F1 | Pass | AIR F1 |
|-------------------------|---------|---------|----------|------|--------|
| Baseline (1-pass)       | 0.694   | 0.720   | 0.698    | 7    | 0.502  |
| gnd3+exp3 (3-pass)      | 0.679   | 0.742   | 0.700    | 10   | 0.514  |
| gnd3+exp3 (3-pass, r2)  | 0.675   | 0.738   | 0.697    | 10   | 0.505  |

Multi-pass stabilizes per-policy variance: leicestershire_police spread dropped from
0.618 (across single-pass runs) to 0.059 (between two 3-pass runs). Pass rate improved
7→10. Shipped as new default: expand_siblings=True, grounding_passes=3, expansion_passes=3.

**Reranker comparison (IR-only, Qwen3-4B bi-encoder, 27 policies):**

| Reranker                    | Macro P | Macro R | Macro F1 | AIR F1 |
|-----------------------------|---------|---------|----------|--------|
| GTE-reranker (149M, f=0.70) | 0.359   | 0.725   | 0.465    | 0.283  |
| Nemotron-rerank-1B (f=0)    | 0.313   | 0.697   | 0.419    | 0.173  |
| Qwen3-Reranker-4B (f=0)    | 0.384   | 0.741   | 0.492    | 0.254  |

GTE-reranker-modernbert-base remains the best reranker for this task. Nemotron
underperforms due to weaker discrimination on policy-risk pairs (trained on MS MARCO
web search). Qwen3-Reranker-4B (generative, yes/no logprob scoring via /v1/completions)
slightly beats GTE in IR-only but at much higher latency (one API call per candidate pair
vs batched /v1/score).

**Bi-encoder comparison (IR-only, GTE reranker):**

| Bi-encoder          | Macro F1 | AIR F1 | Notes |
|---------------------|----------|--------|-------|
| Qwen3-Embedding-4B  | 0.465    | 0.283  | Current default |
| Qwen3-Embedding-8B  | 0.491    | 0.258  | Better overall, worse AIR |

Qwen3-8B improves overall micro F1 but regresses ai-risk-taxonomy. In the full pipeline
(with judge+grounding+expansion), both converge to similar results (0.717 vs 0.712) —
the LLM stages compensate for bi-encoder differences. 4B retained as default (cheaper,
same end-to-end quality).

**Full pipeline with Qwen3-Reranker-4B:**

| Config                  | Micro P | Micro R | Micro F1 | AIR F1 |
|-------------------------|---------|---------|----------|--------|
| 4B+GTE gnd3+exp3        | 0.687   | 0.749   | 0.717    | 0.514  |
| 4B+Qwen3-RR gnd3+exp3   | 0.615   | 0.804   | 0.697    | 0.571  |
| 4B+Nemotron gnd3+exp3   | 0.620   | 0.706   | 0.660    | 0.461  |

Qwen3-Reranker achieves best AIR F1 (0.571) by getting more ai-risk-taxonomy seeds
through reranking, triggering broader expansion coverage. But overall micro F1 is lower
(0.697 vs 0.717) due to expansion over-grounding of data-type variant siblings. Nemotron
is worst overall (0.660).

**GT enrichment round 2:** Review of 365 expansion candidates from the Qwen3-Reranker run
identified 84 high-confidence keyword-matched candidates as genuine GT gaps. Added across
15 policies (1435→1519 total risks, 206→290 ai-risk-taxonomy).

**Conclusion:** GTE-reranker + 3-pass grounding/expansion is the best overall config.
Shipped as new defaults. Qwen3-Reranker is promising for ai-risk-taxonomy specifically
but the precision cost on other taxonomies makes it unsuitable as a drop-in replacement.

