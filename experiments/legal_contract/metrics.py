"""
Legal Contract Validation Metrics

Custom metrics for evaluating legal clause extraction with span overlap scoring.
"""


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
        return 0.8
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
    gold_text = example.clause_text
    pred_text = pred.clause_text
    
    # Handle "NOT FOUND" cases
    gold_not_found = normalize_text(gold_text) == "not found"
    pred_not_found = normalize_text(pred_text) == "not found"
    
    if gold_not_found and pred_not_found:
        return 1.0
    elif gold_not_found or pred_not_found:
        return 0.0
    
    # Compute various matching scores
    exact_match = compute_exact_match(pred_text, gold_text)
    if exact_match:
        return 1.0
    
    substring_score = compute_substring_match(pred_text, gold_text)
    if substring_score > 0:
        return substring_score
    
    # Fall back to word overlap F1
    _, _, f1 = compute_word_overlap_f1(pred_text, gold_text)
    return f1


# Alias for backward compatibility
enhanced_validate_clause_extraction = validate_clause_extraction
