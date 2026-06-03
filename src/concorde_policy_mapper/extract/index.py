from __future__ import annotations

import logging

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from concorde_policy_mapper.extract.models import ScoredCandidate

logger = logging.getLogger(__name__)

_DEFAULT_BI_ENCODER = "all-mpnet-base-v2"
_DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-12-v2"

_SIGMOID_MODELS = {"cross-encoder/ms-marco-MiniLM-L-12-v2", "cross-encoder/ms-marco-electra-base"}
_NLI_MODELS = {"cross-encoder/nli-deberta-v3-base", "cross-encoder/nli-deberta-v3-large", "cross-encoder/nli-deberta-v3-small"}


def _is_remote(model: str) -> bool:
    return model.startswith("http://") or model.startswith("https://")


def _parse_remote_url(url: str) -> tuple[str, str]:
    """Parse a remote model URL into (base_url ending in /v1, model_name).

    Accepts URLs with or without trailing endpoint paths (/embeddings, /score).
    Extracts model_name from hostname pattern '{name}-model-serving.apps...'.
    Falls back to querying /v1/models if the pattern doesn't match.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    path = parsed.path.rstrip("/")
    for suffix in ("/embeddings", "/score", "/rerank"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if not path.endswith("/v1"):
        path = path.rstrip("/") + "/v1"
    base_url = base + path

    hostname = parsed.hostname or ""
    if "-model-serving" in hostname:
        model_name = hostname.split("-model-serving")[0]
    else:
        model_name = hostname.split(".")[0]
    return base_url, model_name


class _RemoteBiEncoder:
    """Wraps a vLLM /v1/embeddings endpoint via the OpenAI SDK.

    Supports instruction-aware models (e.g. Qwen3-Embedding) via
    query_instruction. When set, queries are prefixed with the instruction
    but documents (corpus) are encoded without it.
    """

    def __init__(self, url: str, batch_size: int = 64, query_instruction: str = ""):
        from openai import OpenAI

        base_url, self._model = _parse_remote_url(url)
        self._client = OpenAI(base_url=base_url, api_key="none")
        self._batch_size = batch_size
        self._query_instruction = query_instruction

    def _format_query(self, text: str) -> str:
        if self._query_instruction:
            return f"Instruct: {self._query_instruction}\nQuery:{text}"
        return text

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        all_embeddings = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend(d.embedding for d in sorted_data)
        arr = np.array(all_embeddings, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    def encode_query(self, text: str, normalize: bool = True) -> np.ndarray:
        query = self._format_query(text)
        response = self._client.embeddings.create(
            model=self._model,
            input=[query],
        )
        arr = np.array(response.data[0].embedding, dtype=np.float32)
        if normalize:
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        return arr


class _RemoteCrossEncoder:
    """Wraps a vLLM /v1/score endpoint via httpx."""

    def __init__(self, url: str):
        import httpx

        base_url, self._model = _parse_remote_url(url)
        self._score_url = base_url + "/score"
        self._client = httpx.Client(timeout=120.0)

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([])
        # pairs are (risk_desc, chunk_text) — all share the same chunk_text
        # vLLM /v1/score: text_1=query (chunk), text_2=documents (risks)
        text_1 = pairs[0][1]
        text_2_list = [p[0] for p in pairs]
        response = self._client.post(
            self._score_url,
            json={
                "model": self._model,
                "text_1": text_1,
                "text_2": text_2_list,
            },
        )
        response.raise_for_status()
        data = response.json()
        scores_data = sorted(data["data"], key=lambda d: d["index"])
        return np.array([d["score"] for d in scores_data], dtype=np.float64)


def _maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """ColBERT MaxSim: for each query token, find max cosine similarity with any doc token, then sum."""
    sim = np.dot(query_tokens, doc_tokens.T)
    return float(sim.max(axis=1).sum())


def _rrf_fuse(
    results_a: list[ScoredCandidate],
    results_b: list[ScoredCandidate],
    rrf_k: int = 60,
) -> tuple[dict[str, float], dict[str, ScoredCandidate], dict[str, int]]:
    rrf_scores: dict[str, float] = {}
    candidate_data: dict[str, ScoredCandidate] = {}
    bm25_ranks: dict[str, int] = {}

    for rank, c in enumerate(results_a, 1):
        rrf_scores[c.risk_id] = rrf_scores.get(c.risk_id, 0) + 1 / (rrf_k + rank)
        candidate_data[c.risk_id] = c
        bm25_ranks[c.risk_id] = rank

    for rank, c in enumerate(results_b, 1):
        rrf_scores[c.risk_id] = rrf_scores.get(c.risk_id, 0) + 1 / (rrf_k + rank)
        if c.risk_id not in candidate_data:
            candidate_data[c.risk_id] = c

    return rrf_scores, candidate_data, bm25_ranks


def _make_score_normalizer(*, is_nli=False, apply_sigmoid=False):
    if is_nli:
        def normalize(raw):
            if raw.ndim == 2:
                exp_scores = np.exp(raw - np.max(raw, axis=1, keepdims=True))
                softmax = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                return softmax[:, -1]
            return raw
        return normalize
    elif apply_sigmoid:
        def normalize(raw):
            return 1.0 / (1.0 + np.exp(-raw))
        return normalize
    else:
        def normalize(raw):
            return np.clip(raw.astype(np.float64), 0.0, 1.0)
        return normalize


def _load_colbert(model_name, descriptions):
    if _is_remote(model_name):
        raise ValueError(
            "ColBERT models cannot be served remotely (vLLM returns pooled "
            "embeddings, not token-level). Use --bi-encoder-model for remote "
            "embedding models."
        )
    import torch
    colbert = SentenceTransformer(
        model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    raw = colbert.encode(
        descriptions, output_value="token_embeddings",
        show_progress_bar=False, batch_size=32,
    )
    doc_embeddings = []
    for emb in raw:
        arr = emb.cpu().float().numpy() if hasattr(emb, "cpu") else np.array(emb, dtype=np.float32)
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        doc_embeddings.append(arr)
    logger.info("ColBERT index built: %d risks, %s", len(descriptions), model_name)
    return colbert, doc_embeddings


def _load_bi_encoder(model_name, descriptions, query_instruction=""):
    if _is_remote(model_name):
        remote = _RemoteBiEncoder(model_name, query_instruction=query_instruction)
        try:
            embeddings = remote.encode(descriptions, normalize=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to encode corpus via remote bi-encoder at {model_name}: {e}"
            ) from e
        logger.info("Remote bi-encoder index built: %d risks, %s", len(descriptions), model_name)
        return None, remote, embeddings
    local = SentenceTransformer(model_name)
    embeddings = local.encode(descriptions, normalize_embeddings=True, show_progress_bar=False)
    return local, None, embeddings


def _load_cross_encoder(model_name):
    if _is_remote(model_name):
        logger.info("Remote cross-encoder: %s", model_name)
        return None, _RemoteCrossEncoder(model_name), False, False
    local = CrossEncoder(model_name)
    logger.info("Local cross-encoder: %s", model_name)
    return local, None, model_name in _SIGMOID_MODELS, model_name in _NLI_MODELS


class RiskIndex:
    def __init__(
        self,
        risks: list,
        bi_encoder_model: str = _DEFAULT_BI_ENCODER,
        cross_encoder_model: str | None = _DEFAULT_CROSS_ENCODER,
        colbert_model: str | None = None,
        query_instruction: str = "",
    ):
        self._risk_ids: list[str] = []
        self._risk_meta: dict[str, dict] = {}
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None
        self._colbert_doc_embeddings: list[np.ndarray] | None = None

        self._remote_bi_encoder: _RemoteBiEncoder | None = None
        self._remote_cross_encoder: _RemoteCrossEncoder | None = None

        if not risks:
            self._bi_encoder = None
            self._cross_encoder = None
            self._colbert = None
            return

        for r in risks:
            rid = r.id
            self._risk_ids.append(rid)
            self._risk_meta[rid] = {
                "name": r.name or "",
                "description": r.description or "",
                "concern": getattr(r, "concern", "") or "",
                "taxonomy": getattr(r, "isDefinedByTaxonomy", "") or "",
            }

        bm25_corpus = []
        for r in risks:
            parts = [
                r.name or "",
                r.description or "",
                getattr(r, "concern", "") or "",
                getattr(r, "isPartOf", "") or "",
            ]
            bm25_corpus.append(" ".join(parts).lower().split())
        self._bm25 = BM25Okapi(bm25_corpus)

        descriptions = [f"{r.name or ''}: {r.description or ''}" for r in risks]

        if colbert_model:
            self._colbert, self._colbert_doc_embeddings = _load_colbert(colbert_model, descriptions)
            self._bi_encoder = None
            self._cross_encoder = None
            self._apply_sigmoid = False
            self._is_nli = False
        else:
            self._colbert = None
            self._bi_encoder, self._remote_bi_encoder, self._embeddings = _load_bi_encoder(
                bi_encoder_model, descriptions, query_instruction,
            )

            if cross_encoder_model:
                self._cross_encoder, self._remote_cross_encoder, self._apply_sigmoid, self._is_nli = (
                    _load_cross_encoder(cross_encoder_model)
                )
            else:
                self._cross_encoder = None
                self._apply_sigmoid = False
                self._is_nli = False

            self._score_normalizer = _make_score_normalizer(
                is_nli=self._is_nli, apply_sigmoid=self._apply_sigmoid,
            )

    @property
    def risk_count(self) -> int:
        return len(self._risk_ids)

    @property
    def cross_encoder(self):
        return self._cross_encoder or self._remote_cross_encoder

    @property
    def has_colbert(self) -> bool:
        return self._colbert is not None

    def get_taxonomy(self, risk_id: str) -> str:
        return self._risk_meta.get(risk_id, {}).get("taxonomy", "")

    def search_bm25(self, text: str, top_k: int = 100) -> list[ScoredCandidate]:
        if not self._bm25:
            return []
        tokens = text.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break
            rid = self._risk_ids[idx]
            meta = self._risk_meta[rid]
            results.append(
                ScoredCandidate(
                    risk_id=rid,
                    risk_name=meta["name"],
                    risk_description=meta["description"],
                    bm25_rank=rank + 1,
                )
            )
        return results

    def search_semantic(self, text: str, top_k: int = 100) -> list[ScoredCandidate]:
        if self._bi_encoder is None and self._remote_bi_encoder is None:
            return []
        if self._embeddings is None:
            return []
        if self._remote_bi_encoder is not None:
            query_emb = self._remote_bi_encoder.encode_query(text, normalize=True)
        else:
            query_emb = self._bi_encoder.encode(text, normalize_embeddings=True)
        similarities = np.dot(self._embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            rid = self._risk_ids[idx]
            meta = self._risk_meta[rid]
            results.append(
                ScoredCandidate(
                    risk_id=rid,
                    risk_name=meta["name"],
                    risk_description=meta["description"],
                    embedding_distance=float(1.0 - similarities[idx]),
                )
            )
        return results

    def search_colbert(self, text: str, top_k: int = 100) -> list[ScoredCandidate]:
        if self._colbert is None or self._colbert_doc_embeddings is None:
            return []
        query_emb = self._colbert.encode(text, output_value="token_embeddings")
        query_arr = query_emb.cpu().float().numpy() if hasattr(query_emb, "cpu") else np.array(query_emb, dtype=np.float32)
        query_arr = query_arr / np.linalg.norm(query_arr, axis=1, keepdims=True)
        n_query_tokens = query_arr.shape[0]

        scores = np.array([
            _maxsim(query_arr, doc_emb) / n_query_tokens
            for doc_emb in self._colbert_doc_embeddings
        ])
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            rid = self._risk_ids[idx]
            meta = self._risk_meta[rid]
            results.append(
                ScoredCandidate(
                    risk_id=rid,
                    risk_name=meta["name"],
                    risk_description=meta["description"],
                    cross_encoder_score=float(scores[idx]),
                    embedding_distance=0.0,
                )
            )
        return results

    def rerank(
        self, text: str, candidates: list[ScoredCandidate], top_k: int = 50
    ) -> list[ScoredCandidate]:
        if not candidates or (not self._cross_encoder and not self._remote_cross_encoder):
            return []
        if self._is_nli:
            pairs = [(text, f"This text discusses {c.risk_name}") for c in candidates]
        else:
            pairs = [(f"{c.risk_name}: {c.risk_description}", text) for c in candidates]
        if self._remote_cross_encoder is not None:
            raw_scores = self._remote_cross_encoder.predict(pairs)
        else:
            raw_scores = self._cross_encoder.predict(pairs)
        raw_scores = np.array(raw_scores)
        scores = self._score_normalizer(raw_scores)
        scored = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        results = []
        for c, score in scored[:top_k]:
            s = float(score)
            if np.isnan(s) or np.isinf(s):
                logger.warning("Skipping candidate %s with invalid score", c.risk_id)
                continue
            results.append(c.model_copy(update={"cross_encoder_score": s}))
        return results

    def hybrid_search(
        self,
        text: str,
        top_k: int = 50,
        bm25_top_k: int = 100,
        semantic_top_k: int = 100,
        rrf_k: int = 60,
        bm25_rescue_rank: int = 0,
        rrf_min_score: float = 0.0,
    ) -> list[ScoredCandidate]:
        if self._colbert is not None:
            return self._hybrid_search_colbert(text, top_k, bm25_top_k, rrf_k, bm25_rescue_rank)

        bm25_results = self.search_bm25(text, top_k=bm25_top_k)
        semantic_results = self.search_semantic(text, top_k=semantic_top_k)

        if not bm25_results and not semantic_results:
            return []

        rrf_scores, candidate_data, bm25_ranks = _rrf_fuse(
            bm25_results, semantic_results, rrf_k=rrf_k,
        )
        semantic_distances = {c.risk_id: c.embedding_distance for c in semantic_results}

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        rrf_candidates = []
        for rid in sorted_ids[: top_k * 2]:
            c = candidate_data[rid]
            rrf_candidates.append(
                c.model_copy(
                    update={
                        "rrf_score": rrf_scores[rid],
                        "bm25_rank": bm25_ranks.get(rid, 0),
                        "embedding_distance": semantic_distances.get(
                            rid, c.embedding_distance
                        ),
                    }
                )
            )

        if rrf_min_score > 0:
            return [
                c for c in rrf_candidates
                if c.rrf_score >= rrf_min_score
            ][:top_k]

        reranked = self.rerank(text, rrf_candidates, top_k=top_k)

        if bm25_rescue_rank > 0:
            reranked_ids = {c.risk_id for c in reranked}
            for c in rrf_candidates:
                if (
                    c.risk_id not in reranked_ids
                    and c.bm25_rank > 0
                    and c.bm25_rank <= bm25_rescue_rank
                ):
                    reranked.append(c)

        rrf_lookup = {c.risk_id: c for c in rrf_candidates}
        results = []
        for c in reranked:
            source = rrf_lookup.get(c.risk_id)
            if source:
                results.append(
                    c.model_copy(
                        update={
                            "rrf_score": source.rrf_score,
                            "bm25_rank": source.bm25_rank,
                            "embedding_distance": source.embedding_distance,
                        }
                    )
                )
            else:
                results.append(c)
        return results

    def _hybrid_search_colbert(
        self,
        text: str,
        top_k: int = 50,
        bm25_top_k: int = 100,
        rrf_k: int = 60,
        bm25_rescue_rank: int = 0,
    ) -> list[ScoredCandidate]:
        """Hybrid search using ColBERT MaxSim + BM25 with RRF fusion.

        ColBERT replaces both bi-encoder retrieval and cross-encoder reranking.
        MaxSim scores are stored in cross_encoder_score for threshold compatibility.
        """
        bm25_results = self.search_bm25(text, top_k=bm25_top_k)
        colbert_results = self.search_colbert(text, top_k=top_k * 2)

        if not bm25_results and not colbert_results:
            return []

        rrf_scores, candidate_data, bm25_ranks = _rrf_fuse(
            bm25_results, colbert_results, rrf_k=rrf_k,
        )
        colbert_scores = {c.risk_id: c.cross_encoder_score for c in colbert_results}

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        results = []
        for rid in sorted_ids:
            c = candidate_data[rid]
            results.append(
                c.model_copy(
                    update={
                        "rrf_score": rrf_scores[rid],
                        "bm25_rank": bm25_ranks.get(rid, 0),
                        "cross_encoder_score": colbert_scores.get(rid, 0.0),
                    }
                )
            )

        if bm25_rescue_rank > 0:
            result_ids = {c.risk_id for c in results}
            for c in bm25_results:
                if c.risk_id not in result_ids and c.bm25_rank <= bm25_rescue_rank:
                    results.append(
                        c.model_copy(
                            update={
                                "rrf_score": rrf_scores.get(c.risk_id, 0),
                                "cross_encoder_score": colbert_scores.get(c.risk_id, 0.0),
                            }
                        )
                    )

        return results
