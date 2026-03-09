"""
Legal Contract Analysis Detailed Evaluation

Provides detailed evaluation metrics for clause extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from shared.evaluation import EvaluationResult
from .metrics import (
    compute_word_overlap_f1,
    compute_exact_match,
    normalize_text
)


# Extended result container for domain-specific metrics
@dataclass
class LegalEvaluationMetrics:
    """Additional metrics specific to legal contract evaluation."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    exact_match_rate: float = 0.0
    not_found_accuracy: float = 0.0
    per_clause_type: Dict[str, float] = field(default_factory=dict)


def detailed_evaluation(module, testset, model_name="Model"):
    """
    Run detailed evaluation on test set.
    
    Args:
        module: DSPy module to evaluate
        testset: List of dspy.Example objects
        model_name: Name for logging
    
    Returns:
        EvaluationResult compatible with shared compare_results
    """
    results = []  # For compare_results compatibility
    
    all_precision = []
    all_recall = []
    all_f1 = []
    exact_matches = 0
    not_found_correct = 0
    not_found_total = 0
    
    per_clause_type = {}
    
    for i, example in enumerate(testset):
        try:
            pred = module(
                contract_text=example.contract_text,
                clause_type=example.clause_type
            )
            
            gold_text = example.clause_text
            pred_text = pred.clause_text
            
            # Check NOT FOUND cases
            gold_not_found = normalize_text(gold_text) == "not found"
            pred_not_found = normalize_text(pred_text) == "not found"
            
            if gold_not_found:
                not_found_total += 1
                if pred_not_found:
                    not_found_correct += 1
                    precision, recall, f1 = 1.0, 1.0, 1.0
                else:
                    precision, recall, f1 = 0.0, 0.0, 0.0
            elif pred_not_found:
                precision, recall, f1 = 0.0, 0.0, 0.0
            else:
                precision, recall, f1 = compute_word_overlap_f1(pred_text, gold_text)
            
            # Track exact matches
            if compute_exact_match(pred_text, gold_text):
                exact_matches += 1
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            
            # Track per-clause-type performance
            clause_type = example.clause_type
            if clause_type not in per_clause_type:
                per_clause_type[clause_type] = {'scores': [], 'count': 0}
            per_clause_type[clause_type]['scores'].append(f1)
            per_clause_type[clause_type]['count'] += 1
            
            # Use 'results' with 'overall_score' for compare_results compatibility
            results.append({
                'index': i,
                'clause_type': clause_type,
                'gold_clause': gold_text[:200] + "..." if len(gold_text) > 200 else gold_text,
                'pred_clause': pred_text[:200] + "..." if len(pred_text) > 200 else pred_text,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'overall_score': f1  # Required for compare_results
            })
            
        except Exception as e:
            results.append({
                'index': i,
                'clause_type': example.clause_type,
                'error': str(e),
                'gold_clause': example.clause_text[:100],
                'pred_clause': '',
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'overall_score': 0.0  # Required for compare_results
            })
            all_precision.append(0.0)
            all_recall.append(0.0)
            all_f1.append(0.0)
    
    # Compute overall metrics
    n = len(testset)
    avg_precision = sum(all_precision) / n if n > 0 else 0
    avg_recall = sum(all_recall) / n if n > 0 else 0
    avg_f1 = sum(all_f1) / n if n > 0 else 0
    exact_match_rate = exact_matches / n if n > 0 else 0
    not_found_accuracy = not_found_correct / not_found_total if not_found_total > 0 else 1.0
    
    # Compute per-clause-type averages
    per_clause_avg = {
        ct: sum(data['scores']) / len(data['scores'])
        for ct, data in per_clause_type.items()
    }
    
    # Create result compatible with shared EvaluationResult
    eval_result = EvaluationResult(
        name=model_name,
        results=results,
        field_accuracies={'clause_text': avg_f1},
        overall_accuracy=avg_f1,
        total_samples=n,
        metadata={
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'exact_match_rate': exact_match_rate,
            'not_found_accuracy': not_found_accuracy,
            'per_clause_type': per_clause_avg
        }
    )
    
    return eval_result


def print_evaluation_summary(result: EvaluationResult):
    """Print evaluation summary."""
    # Access domain-specific metrics from metadata
    meta = result.metadata
    
    print(f"\n{'=' * 50}")
    print(f"Evaluation Summary: {result.name}")
    print(f"{'=' * 50}")
    print(f"Precision:           {meta.get('precision', 0):.2%}")
    print(f"Recall:              {meta.get('recall', 0):.2%}")
    print(f"F1 Score:            {meta.get('f1_score', 0):.2%}")
    print(f"Exact Match Rate:    {meta.get('exact_match_rate', 0):.2%}")
    print(f"NOT FOUND Accuracy:  {meta.get('not_found_accuracy', 0):.2%}")
    
    per_clause_type = meta.get('per_clause_type', {})
    if per_clause_type:
        print("\nPer-Clause-Type F1:")
        for clause_type, score in sorted(per_clause_type.items()):
            print(f"  {clause_type:25} {score:.2%}")
