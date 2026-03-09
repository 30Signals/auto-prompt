"""
Legal Contract Validation Metrics

Custom metrics for evaluating legal clause extraction with span overlap scoring.
"""

import json
import os
from typing import Iterable, Optional

import dspy

# Higher weights force optimization to focus on historically weak clause types.
CLAUSE_TYPE_WEIGHTS = {
    "Parties": 1.35,
    "Governing Law": 1.25,
    "Effective Date": 1.20,
    "Expiration Date": 1.20,
}


def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def word_tokenize(text):
    """Simple word tokenization."""
    return normalize_text(text).split()


def compute_word_overlap_f1(pred_text, gold_text):
    """
    Compute word-level F1 score between predicted and gold text.

    Args:
        pred_text: Predicted clause text
        gold_text: Gold standard clause text

    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_words = set(word_tokenize(pred_text))
    gold_words = set(word_tokenize(gold_text))

    if not pred_words and not gold_words:
        return 1.0, 1.0, 1.0

    if not pred_words or not gold_words:
        return 0.0, 0.0, 0.0

    overlap = len(pred_words & gold_words)
    precision = overlap / len(pred_words)
    recall = overlap / len(gold_words)

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_exact_match(pred_text, gold_text):
    """Check if texts match exactly after normalization."""
    return normalize_text(pred_text) == normalize_text(gold_text)


def compute_substring_match(pred_text, gold_text):
    """Check if one text contains the other (partial match)."""
    pred_norm = normalize_text(pred_text)
    gold_norm = normalize_text(gold_text)

    if not pred_norm or not gold_norm:
        return 0.0

    if pred_norm == gold_norm:
        return 1.0
    elif pred_norm in gold_norm or gold_norm in pred_norm:
        return 0.4
    else:
        return 0.0


def validate_clause_extraction(example, pred, trace=None):
    """
    Validation metric for DSPy optimization.

    Combines word overlap F1 with exact/substring matching.

    Args:
        example: Ground truth dspy.Example
        pred: Model prediction
        trace: Optional trace (unused)

    Returns:
        Score between 0 and 1
    """
    gold_candidates = getattr(example, "gold_clauses", None)
    if isinstance(gold_candidates, list):
        gold_candidates = [str(x).strip() for x in gold_candidates if str(x).strip()]
    else:
        gold_candidates = []
    if not gold_candidates:
        gold_candidates = [str(getattr(example, "clause_text", "")).strip() or "NOT FOUND"]

    pred_text = pred.clause_text

    # Handle "NOT FOUND" cases
    gold_not_found = all(normalize_text(x) == "not found" for x in gold_candidates)
    pred_not_found = normalize_text(pred_text) == "not found"

    if gold_not_found and pred_not_found:
        base_score = 1.0
    elif gold_not_found or pred_not_found:
        base_score = 0.0
    else:
        # Score against all valid gold spans and keep the best match.
        base_score = 0.0
        for candidate in gold_candidates:
            if compute_exact_match(pred_text, candidate):
                candidate_score = 1.0
            else:
                substring_score = compute_substring_match(pred_text, candidate)
                if substring_score > 0:
                    candidate_score = substring_score
                else:
                    _, _, candidate_score = compute_word_overlap_f1(pred_text, candidate)
            if candidate_score > base_score:
                base_score = candidate_score

    clause_type = getattr(example, "clause_type", "")
    weight = CLAUSE_TYPE_WEIGHTS.get(clause_type, 1.0)
    return min(1.0, base_score * weight)


# Alias for backward compatibility
enhanced_validate_clause_extraction = validate_clause_extraction


def _extract_first_json_object(text: str) -> str:
    s = str(text or "").strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    start = s.find("{")
    if start < 0:
        return ""
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return ""


def llm_semantic_match_score(
    pred_text: str,
    gold_candidates: Iterable[str],
    clause_type: str,
) -> Optional[float]:
    """
    Ask an LLM judge for semantic match score in [0, 1].
    This is robust to format differences (e.g., date formats, phrasing).

    Returns:
        float score in [0,1], or None when LLM eval is disabled/unavailable.
    """
    use_llm_eval = os.getenv("LEGAL_CONTRACT_LLM_EVAL", "1").strip().lower()
    if use_llm_eval in {"0", "false", "no"}:
        return None

    try:
        lm = dspy.settings.lm
    except Exception:
        return None
    if lm is None:
        return None

    gold_list = [str(x).strip() for x in gold_candidates if str(x).strip()]
    if not gold_list:
        gold_list = ["NOT FOUND"]

    prompt = f"""
You are a strict legal evaluation judge.
Score whether prediction matches any gold answer for the requested clause type.

Clause type: {clause_type}
Gold candidates: {json.dumps(gold_list, ensure_ascii=False)}
Prediction: {json.dumps(str(pred_text or ''), ensure_ascii=False)}

Rules:
- Return score from 0.0 to 1.0.
- Consider semantic equivalence and formatting variants:
  - Date variants (e.g., "03/01/05" vs "March 1, 2005")
  - Minor punctuation/casing/whitespace differences
  - Equivalent legal phrasing
- Score near 1.0 for equivalent meaning to any gold candidate.
- Score near 0.0 for wrong/irrelevant clause.
- If prediction is "NOT FOUND", score high only when gold is also "NOT FOUND".

Return ONLY JSON:
{{"score": 0.0}}
""".strip()

    try:
        raw = lm(prompt)
        text = raw[0] if isinstance(raw, list) and raw else raw
        payload = _extract_first_json_object(text)
        if not payload:
            return None
        score = float(json.loads(payload).get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        return None
