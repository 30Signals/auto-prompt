"""
Company Legal Risk Detailed Evaluation
"""

import time

from shared.evaluation import EvaluationResult

# Custom evaluation logic for the company legal risk experiment, including detailed scoring for risk level accuracy, decision correctness, and evidence recall, as well as a summary printout of results.
from .metrics import (
    risk_exact_score,
    risk_distance_score,
    decision_score,
    findings_recall_score,
)


def _predict_with_retry(module, company_name, search_context, retries=2, base_sleep=1.5):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return module(company_name=company_name, search_context=search_context)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            transient = any(x in msg for x in ["connection error", "timeout", "rate limit", "internalservererror"])
            if attempt >= retries or not transient:
                break
            time.sleep(base_sleep * (attempt + 1))
    raise last_exc

# The detailed_evaluation function runs through the test set, applies the module to get predictions, computes various metrics for each example, and aggregates them into an EvaluationResult object that can be used for comparison and analysis.
def detailed_evaluation(module, testset, model_name="Model"):
    rows = []

    risk_exact_values = []
    risk_near_values = []
    decision_values = []
    evidence_values = []

    for idx, example in enumerate(testset):
        try:
            pred = _predict_with_retry(
                module=module,
                company_name=example.company_name,
                search_context=example.search_context,
            )

            risk_exact = risk_exact_score(pred.risk_level, example.risk_level)
            risk_near = risk_distance_score(pred.risk_level, example.risk_level)
            decision_acc = decision_score(
                pred.should_work_with_company,
                example.should_work_with_company,
            )
            evidence_recall = findings_recall_score(
                pred.key_findings,
                example.expected_findings,
            )

            overall = (0.40 * risk_near) + (0.40 * decision_acc) + (0.20 * evidence_recall)

            risk_exact_values.append(risk_exact)
            risk_near_values.append(risk_near)
            decision_values.append(decision_acc)
            evidence_values.append(evidence_recall)

            rows.append(
                {
                    "index": idx,
                    "company_name": example.company_name,
                    "gold_risk": example.risk_level,
                    "pred_risk": pred.risk_level,
                    "gold_decision": example.should_work_with_company,
                    "pred_decision": pred.should_work_with_company,
                    "risk_exact": risk_exact,
                    "risk_distance": risk_near,
                    "decision_accuracy": decision_acc,
                    "evidence_recall": evidence_recall,
                    "overall_score": overall,
                    "summary": getattr(pred, "summary", ""),
                    "key_findings": getattr(pred, "key_findings", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "index": idx,
                    "company_name": example.company_name,
                    "error": str(exc),
                    "risk_exact": 0.0,
                    "risk_distance": 0.0,
                    "decision_accuracy": 0.0,
                    "evidence_recall": 0.0,
                    "overall_score": 0.0,
                }
            )
            risk_exact_values.append(0.0)
            risk_near_values.append(0.0)
            decision_values.append(0.0)
            evidence_values.append(0.0)

    total = max(1, len(testset))
    metrics = {
        "risk_exact_accuracy": sum(risk_exact_values) / total,
        "risk_distance_score": sum(risk_near_values) / total,
        "decision_accuracy": sum(decision_values) / total,
        "evidence_recall": sum(evidence_values) / total,
    }

    overall_accuracy = (
        0.40 * metrics["risk_distance_score"]
        + 0.40 * metrics["decision_accuracy"]
        + 0.20 * metrics["evidence_recall"]
    )

    return EvaluationResult(
        name=model_name,
        results=rows,
        field_accuracies={
            "risk_distance": metrics["risk_distance_score"],
            "decision_accuracy": metrics["decision_accuracy"],
            "evidence_recall": metrics["evidence_recall"],
        },
        overall_accuracy=overall_accuracy,
        total_samples=len(testset),
        metadata=metrics,
    )

# The print_evaluation_summary function provides a concise summary of the evaluation results, printing out the key metrics in a readable format for quick analysis and comparison across different models or configurations.

def print_evaluation_summary(result):
    print("\n" + "=" * 50)
    print(f"Evaluation Summary: {result.name}")
    print("=" * 50)
    print(f"Risk Exact Accuracy: {result.metadata.get('risk_exact_accuracy', 0):.2%}")
    print(f"Risk Distance Score: {result.metadata.get('risk_distance_score', 0):.2%}")
    print(f"Decision Accuracy:   {result.metadata.get('decision_accuracy', 0):.2%}")
    print(f"Evidence Recall:     {result.metadata.get('evidence_recall', 0):.2%}")
    print(f"Overall Score:       {result.overall_accuracy:.2%}")
