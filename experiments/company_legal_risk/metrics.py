"""
Company Legal Risk Validation Metrics
"""
# Custom metrics for evaluating company legal risk assessments, including risk level accuracy, decision correctness, and evidence recall.
RISK_TO_INDEX = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

# Normalize label with a default value to handle missing or empty inputs gracefully.
def normalize_label(value, default=""):
    text = str(value or "").strip().upper()
    return text or default

# Normalize risk level to one of the predefined categories, with leniency for various input formats.

def normalize_decision(value):
    text = normalize_label(value)
    if text in {"YES", "Y", "WORK", "APPROVE"}:
        return "YES"
    if text in {"NO", "N", "REJECT", "DO NOT WORK"}:
        return "NO"
    return "NO"

# Normalize key findings to a comma-separated list, handling various output formats and ensuring consistent formatting for evaluation.
def risk_exact_score(pred_risk, gold_risk):
    return 1.0 if normalize_label(pred_risk) == normalize_label(gold_risk) else 0.0

# The risk_distance_score function computes a score based on how close the predicted risk level is to the gold risk level, giving partial credit for near misses.
def risk_distance_score(pred_risk, gold_risk):
    pred = RISK_TO_INDEX.get(normalize_label(pred_risk), -1)
    gold = RISK_TO_INDEX.get(normalize_label(gold_risk), -1)
    if pred < 0 or gold < 0:
        return 0.0
    dist = abs(pred - gold)
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.6
    if dist == 2:
        return 0.25
    return 0.0


def risk_underestimation_penalty(pred_risk, gold_risk):
    pred = RISK_TO_INDEX.get(normalize_label(pred_risk), -1)
    gold = RISK_TO_INDEX.get(normalize_label(gold_risk), -1)
    if pred < 0 or gold < 0:
        return 0.0
    # Penalize underestimation more than overestimation for legal/compliance safety.
    if pred < gold:
        return min(1.0, 0.35 * (gold - pred))
    return 0.0

# The decision_score function evaluates the correctness of the recommendation to work with the company, giving a binary score based on whether the predicted decision matches the gold decision after normalization.
def decision_score(pred_decision, gold_decision):
    return 1.0 if normalize_decision(pred_decision) == normalize_decision(gold_decision) else 0.0

# The findings_recall_score function computes a recall score for the key findings by checking how many of the expected findings are mentioned in the predicted findings, giving partial credit based on the proportion of expected findings that are covered.   
def findings_recall_score(pred_findings, expected_findings):
    pred_text = str(pred_findings or "").lower()
    expected = [x.strip().lower() for x in str(expected_findings or "").split(",") if x.strip()]
    if not expected:
        return 1.0
    hits = sum(1 for finding in expected if finding in pred_text)
    return hits / len(expected)

# The validate_company_risk_decision function combines the various metrics into a single score for evaluating the overall quality of the risk assessment, using weighted contributions from exact risk accuracy, near miss scoring, decision correctness, and evidence recall.
def validate_company_risk_decision(example, pred, trace=None):
    exact_risk = risk_exact_score(pred.risk_level, example.risk_level)
    near_risk = risk_distance_score(pred.risk_level, example.risk_level)
    rec_score = decision_score(pred.should_work_with_company, example.should_work_with_company)
    evidence_score = findings_recall_score(pred.key_findings, example.expected_findings)
    under_penalty = risk_underestimation_penalty(pred.risk_level, example.risk_level)
    false_yes_penalty = 0.25 if (
        normalize_decision(example.should_work_with_company) == "NO"
        and normalize_decision(pred.should_work_with_company) == "YES"
    ) else 0.0

    score = (0.28 * exact_risk) + (0.22 * near_risk) + (0.35 * rec_score) + (0.15 * evidence_score)
    score -= under_penalty
    score -= false_yes_penalty
    return max(0.0, score)
