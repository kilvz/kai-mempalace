"""Search utilities — BM25 ranking, closet helpers, virtual line numbering.

The primary search entry point is ``Palace.search()`` in ``palace.py``.
This module provides pure-function helpers for re-ranking and display.
"""

import logging
import math
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)
_CLOSET_DRAWER_REF_RE = re.compile(r"→([\w,]+)")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(
    query: str, documents: list, k1: float = 1.5, b: float = 0.75
) -> list[float]:
    n_docs = len(documents)
    query_terms = set(_tokenize(query))
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    tokenized = [_tokenize(d) for d in documents]
    doc_lens = [len(toks) for toks in tokenized]
    if not any(doc_lens):
        return [0.0] * n_docs
    avgdl = sum(doc_lens) / n_docs or 1.0

    df = {term: 0 for term in query_terms}
    for toks in tokenized:
        seen = set(toks) & query_terms
        for term in seen:
            df[term] += 1

    idf = {term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in query_terms}

    scores = []
    for toks, dl in zip(tokenized, doc_lens):
        if dl == 0:
            scores.append(0.0)
            continue
        tf = {}
        for t in toks:
            if t in query_terms:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * num / den
        scores.append(score)
    return scores


def _hybrid_rank(
    results: list,
    query: str,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list:
    if not results:
        return results

    docs = [r.get("text", "") for r in results]
    bm25_raw = _bm25_scores(query, docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / max_bm25 for s in bm25_raw] if max_bm25 > 0 else [0.0] * len(bm25_raw)

    scored = []
    for r, raw, norm in zip(results, bm25_raw, bm25_norm):
        distance = r.get("distance")
        vec_sim = max(0.0, 1.0 - distance) if distance is not None else 0.0
        r["bm25_score"] = round(raw, 3)
        scored.append((vector_weight * vec_sim + bm25_weight * norm, r))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [r for _, r in scored]


def _extract_drawer_ids_from_closet(closet_doc: str) -> list[str]:
    seen = {}
    for match in _CLOSET_DRAWER_REF_RE.findall(closet_doc):
        for did in match.split(","):
            did = did.strip()
            if did and did not in seen:
                seen[did] = None
    return list(seen.keys())


# ── Candidate strategy system ──────────────────────────────────────────


def _merge_bm25_union_candidates(
    hits: list,
    query: str,
    keyword_results: list,
    n_results: int,
    max_distance: float = 0.0,
) -> None:
    """Append top-K BM25-only candidates into ``hits`` in place.

    Used by ``candidate_strategy="union"`` to widen the rerank pool beyond
    vector-only candidates — catches docs with strong BM25 signal that are
    vector-distant from the query.

    BM25-only additions carry ``distance=None`` so ``_hybrid_rank`` scores
    them on BM25 contribution alone.

    When ``max_distance > 0.0``, BM25-only candidates are skipped —
    they have no vector distance to satisfy the threshold.
    """
    if max_distance > 0.0:
        return
    if not keyword_results:
        return

    seen_keys = {_dedup_key(h) for h in hits}
    for kr in keyword_results:
        key = _dedup_key(kr)
        if not key or key == "?" or key in seen_keys:
            continue
        kr["distance"] = None
        hits.append(kr)
        seen_keys.add(key)


def _dedup_key(entry: dict) -> str:
    src = entry.get("source_file") or entry.get("metadata", {}).get("source_file")
    return src or "?"


_CANDIDATE_MERGERS = {
    "vector": None,
    "union": _merge_bm25_union_candidates,
}


def validate_candidate_strategy(strategy: str) -> None:
    if strategy not in _CANDIDATE_MERGERS:
        raise ValueError(
            f"candidate_strategy must be one of {tuple(_CANDIDATE_MERGERS)}, got {strategy!r}"
        )


def apply_candidate_strategy(
    strategy: str,
    hits: list,
    query: str,
    keyword_results: list,
    n_results: int,
    max_distance: float = 0.0,
) -> None:
    merger = _CANDIDATE_MERGERS[strategy]
    if merger is not None:
        merger(hits, query, keyword_results, n_results, max_distance=max_distance)


# ── Neighbor chunk expansion ───────────────────────────────────────────


def expand_with_neighbors(
    row: tuple,
    db,
    radius: int = 1,
) -> dict:
    """Expand a matched drawer with its *radius* sibling chunks in the same source file.

    Returns a dict with ``text`` (combined chunks), ``drawer_index``, and
    ``total_drawers`` for the source file.
    """
    rid, content, meta_json = row[0], row[3] if len(row) > 3 else "", row[4] if len(row) > 4 else "{}"
    try:
        meta = __import__("json").loads(meta_json) if isinstance(meta_json, str) else meta_json
    except Exception:
        meta = {}
    src = meta.get("source_file")
    chunk_idx = meta.get("chunk_index")
    if not src or not isinstance(chunk_idx, int):
        return {"text": content, "drawer_index": chunk_idx, "total_drawers": None}

    target_indexes = [chunk_idx + offset for offset in range(-radius, radius + 1)]
    try:
        cursor = db.execute(
            "SELECT id, wing, room, content, metadata FROM drawers WHERE source_file = ?",
            (src,),
        )
        all_docs = cursor.fetchall()
    except Exception:
        return {"text": content, "drawer_index": chunk_idx, "total_drawers": None}

    indexed = []
    for d in all_docs:
        try:
            m = __import__("json").loads(d[4]) if isinstance(d[4], str) else d[4]
        except Exception:
            m = {}
        ci = m.get("chunk_index")
        if isinstance(ci, int) and ci in target_indexes:
            indexed.append((ci, d[3]))
    indexed.sort(key=lambda p: p[0])

    combined = "\n\n".join(doc for _, doc in indexed) if indexed else content
    return {
        "text": combined,
        "drawer_index": chunk_idx,
        "total_drawers": len(all_docs),
    }


# ── Virtual line numbering (read-time grid for drawers) ─────────────────


_ALREADY_NUMBERED_RE = re.compile(r"^\[\d+\]")


def render_with_line_numbers(text: Optional[str], start_line: int = 1) -> str:
    if not text:
        return ""
    out = []
    for i, line in enumerate(text.split("\n"), start=start_line):
        if _ALREADY_NUMBERED_RE.match(line):
            out.append(line)
        else:
            out.append(f"[{i}] {line}")
    return "\n".join(out)


def extract_line_range(text: str, line_start: int, line_end: int) -> str:
    if not text:
        return ""
    if line_end < line_start:
        return ""

    lines = text.split("\n")
    effective_start = max(1, line_start)
    effective_end = min(len(lines), line_end)

    if effective_start > effective_end:
        return ""

    section = "\n".join(lines[effective_start - 1 : effective_end])
    return render_with_line_numbers(section, start_line=effective_start)
