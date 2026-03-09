"""
Skill normalization and matching helpers for resume extraction.
"""

from typing import List, Set

# Canonical mapping for common skill variants.
SKILL_ALIASES = {
    "ml": "machine learning",
    "ai": "machine learning",
    "nlp": "natural language processing",
    "dbms": "sql",
    "mysql": "sql",
    "postgresql": "sql",
    "postgres": "sql",
    "pandas library": "pandas",
    "numpy library": "numpy",
    "ms excel": "excel",
    "google analytics": "google analytics",
    "web programming": "web development",
    "backend development": "backend engineering",
}

# Vague terms that should not dominate skills output.
GENERIC_SKILLS = {
    "communication",
    "coordination",
    "reporting",
    "documentation",
    "teamwork",
    "collaboration",
    "time management",
    "problem solving",
}


def _to_tokens(skills_value) -> List[str]:
    if skills_value is None:
        return []
    if isinstance(skills_value, list):
        parts = [str(s).strip().lower() for s in skills_value]
    else:
        parts = [s.strip().lower() for s in str(skills_value).split(",")]
    return [p for p in parts if p]


def normalize_skill_token(token: str) -> str:
    token = token.strip().lower()
    token = token.replace("-", " ")
    token = " ".join(token.split())
    return SKILL_ALIASES.get(token, token)


def normalize_skills(skills_value, keep_generic: bool = False) -> List[str]:
    normalized = []
    seen: Set[str] = set()
    for token in _to_tokens(skills_value):
        canonical = normalize_skill_token(token)
        if not canonical:
            continue
        if not keep_generic and canonical in GENERIC_SKILLS:
            continue
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized


def skills_match_score(pred_skills, gt_skills) -> float:
    gt = normalize_skills(gt_skills)
    pred = normalize_skills(pred_skills)
    if not gt:
        return 0.0
    if not pred:
        return 0.0

    matched = 0
    pred_set = set(pred)
    for gt_skill in gt:
        if gt_skill in pred_set:
            matched += 1
            continue
        if any(gt_skill in p or p in gt_skill for p in pred_set):
            matched += 0.8
            continue
        if gt_skill == "machine learning" and any(p in pred_set for p in {"ml", "ai"}):
            matched += 0.7
            continue

    recall = matched / len(gt)
    precision = matched / len(pred) if pred else 0.0
    return max(0.0, min(1.0, (0.7 * recall) + (0.3 * precision)))


def format_skills(skills_value) -> str:
    return ", ".join(normalize_skills(skills_value))
