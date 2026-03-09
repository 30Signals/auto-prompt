"""
Medical NER Detailed Evaluation

Provides detailed evaluation metrics for disease entity extraction.
"""

from shared.evaluation import EvaluationResult
from .metrics import parse_diseases, compute_f1


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
    total_gold_entities = 0
    total_pred_entities = 0
    total_correct = 0
    
    all_precision = []
    all_recall = []
    all_f1 = []
    
    for i, example in enumerate(testset):
        try:
            pred = module(abstract_text=example.abstract_text)
            
            gold_diseases = parse_diseases(example.diseases)
            pred_diseases = parse_diseases(pred.diseases)
            
            # Compute exact match F1
            precision, recall, f1 = compute_f1(pred_diseases, gold_diseases)
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            
            total_gold_entities += len(gold_diseases)
            total_pred_entities += len(pred_diseases)
            total_correct += len(pred_diseases & gold_diseases)
            
            # Use 'results' with 'overall_score' for compare_results compatibility
            results.append({
                'index': i,
                'abstract_text': example.abstract_text[:200] + "...",
                'gold_diseases': list(gold_diseases),
                'pred_diseases': list(pred_diseases),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'overall_score': f1  # Required for compare_results
            })
            
        except Exception as e:
            results.append({
                'index': i,
                'error': str(e),
                'gold_diseases': list(parse_diseases(example.diseases)),
                'pred_diseases': [],
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

    # Micro metrics at entity level across the full corpus.
    if total_pred_entities > 0:
        micro_precision = total_correct / total_pred_entities
    else:
        micro_precision = 1.0 if total_gold_entities == 0 else 0.0

    if total_gold_entities > 0:
        micro_recall = total_correct / total_gold_entities
    else:
        micro_recall = 1.0 if total_pred_entities == 0 else 0.0

    if micro_precision + micro_recall == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    
    # Create result compatible with shared EvaluationResult
    eval_result = EvaluationResult(
        name=model_name,
        results=results,
        field_accuracies={'diseases': avg_f1},
        overall_accuracy=avg_f1,  # Using F1 as overall accuracy for NER
        total_samples=n,
        metadata={
            # Backward-compatible keys kept as macro (sample-averaged) metrics.
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'macro_precision': avg_precision,
            'macro_recall': avg_recall,
            'macro_f1': avg_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'total_gold_entities': total_gold_entities,
            'total_pred_entities': total_pred_entities,
            'total_correct': total_correct
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
    print(f"Macro Precision:  {meta.get('macro_precision', meta.get('precision', 0)):.2%}")
    print(f"Macro Recall:     {meta.get('macro_recall', meta.get('recall', 0)):.2%}")
    print(f"Macro F1:         {meta.get('macro_f1', meta.get('f1_score', 0)):.2%}")
    print(f"Micro Precision:  {meta.get('micro_precision', 0):.2%}")
    print(f"Micro Recall:     {meta.get('micro_recall', 0):.2%}")
    print(f"Micro F1:         {meta.get('micro_f1', 0):.2%}")
    print(f"Total Gold:       {meta.get('total_gold_entities', 0)}")
    print(f"Total Predicted:  {meta.get('total_pred_entities', 0)}")
    print(f"Exact Matches:    {meta.get('total_correct', 0)}")
