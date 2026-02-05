"""
Medical NER Validation Metrics

Custom metrics for evaluating disease entity extraction with F1 scoring.
"""


def normalize_entity(entity):
    """Normalize entity string for comparison."""
    return entity.strip().lower()


def parse_diseases(diseases_str):
    """Parse comma-separated diseases string into set of normalized entities."""
    if not diseases_str or not diseases_str.strip():
        return set()
    
    entities = [normalize_entity(d) for d in diseases_str.split(',')]
    return set(e for e in entities if e)


def compute_f1(pred_set, gold_set):
    """
    Compute precision, recall, and F1 score.
    
    Args:
        pred_set: Set of predicted entities
        gold_set: Set of gold entities
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    
    if not pred_set:
        return 0.0, 0.0, 0.0
    
    if not gold_set:
        return 0.0, 0.0, 0.0
    
    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(gold_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def fuzzy_match(pred_entity, gold_entities):
    """
    Check if predicted entity fuzzy matches any gold entity.
    
    Uses substring matching for partial credit.
    """
    for gold in gold_entities:
        # Exact match
        if pred_entity == gold:
            return 1.0
        # Substring match
        if pred_entity in gold or gold in pred_entity:
            return 0.8
        # Word overlap
        pred_words = set(pred_entity.split())
        gold_words = set(gold.split())
        overlap = len(pred_words & gold_words)
        if overlap > 0:
            return 0.5 * overlap / max(len(pred_words), len(gold_words))
    
    return 0.0


def validate_disease_output(example, pred, trace=None):
    """
    Validation metric for DSPy optimization.
    
    Uses fuzzy F1 scoring for entity matching.
    
    Args:
        example: Ground truth dspy.Example
        pred: Model prediction
        trace: Optional trace (unused)
    
    Returns:
        Score between 0 and 1
    """
    gold_diseases = parse_diseases(example.diseases)
    pred_diseases = parse_diseases(pred.diseases)
    
    if not gold_diseases and not pred_diseases:
        return 1.0
    
    if not gold_diseases:
        # No gold entities but model predicted some - small penalty
        return 0.2 if pred_diseases else 1.0
    
    if not pred_diseases:
        return 0.0
    
    # Compute fuzzy matching score
    total_score = 0.0
    matched_gold = set()
    
    for pred_entity in pred_diseases:
        best_match = 0.0
        best_gold = None
        
        for gold_entity in gold_diseases:
            if gold_entity in matched_gold:
                continue
            
            match_score = fuzzy_match(pred_entity, {gold_entity})
            if match_score > best_match:
                best_match = match_score
                best_gold = gold_entity
        
        if best_gold and best_match > 0.3:
            matched_gold.add(best_gold)
            total_score += best_match
    
    # Calculate precision-weighted score
    precision_score = total_score / len(pred_diseases) if pred_diseases else 0
    recall_score = len(matched_gold) / len(gold_diseases) if gold_diseases else 0
    
    # F1-like combination
    if precision_score + recall_score == 0:
        return 0.0
    
    return 2 * (precision_score * recall_score) / (precision_score + recall_score)


def compute_exact_f1(example, pred):
    """Compute exact match F1 score for evaluation."""
    gold_diseases = parse_diseases(example.diseases)
    pred_diseases = parse_diseases(pred.diseases)
    
    _, _, f1 = compute_f1(pred_diseases, gold_diseases)
    return f1


# Alias for backward compatibility
enhanced_validate_disease_output = validate_disease_output
