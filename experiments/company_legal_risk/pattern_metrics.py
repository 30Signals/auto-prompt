"""
Pattern-level metrics for company legal risk dimensions.
"""

from .replay_logging import RISK_DIMENSIONS


def normalize_severity(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip().upper()
        mapping = {
            "NONE": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "SEVERE": 3,
            "CRITICAL": 3,
        }
        if text in mapping:
            return mapping[text]
        try:
            value = int(text)
        except ValueError:
            return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= 3 else None


def normalize_score(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value))


def severity_exact_score(pred_value, gold_value):
    pred = normalize_severity(pred_value)
    gold = normalize_severity(gold_value)
    if pred is None or gold is None:
        return None
    return 1.0 if pred == gold else 0.0


def severity_distance_score(pred_value, gold_value):
    pred = normalize_severity(pred_value)
    gold = normalize_severity(gold_value)
    if pred is None or gold is None:
        return None
    dist = abs(pred - gold)
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.67
    if dist == 2:
        return 0.33
    return 0.0


def confidence_alignment_score(pred_value, gold_value):
    pred = normalize_score(pred_value)
    gold = normalize_score(gold_value)
    if pred is None or gold is None:
        return None
    return max(0.0, 1.0 - abs(pred - gold))


def iter_dimension_pairs(pred_dimensions, gold_dimensions):
    pred_dimensions = pred_dimensions or {}
    gold_dimensions = gold_dimensions or {}
    for name in RISK_DIMENSIONS:
        yield name, pred_dimensions.get(name, {}) or {}, gold_dimensions.get(name, {}) or {}
