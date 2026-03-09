"""
Resume Extraction Validation Metrics

Custom metrics for evaluating resume extraction quality with semantic matching.
"""

from .skill_utils import skills_match_score, normalize_skills

# Role synonyms for flexible matching
ROLE_SYNONYMS = {
    'data scientist': ['data analyst', 'ml engineer', 'ai engineer'],
    'software engineer': ['developer', 'programmer', 'software developer'],
    'finance analyst': ['financial analyst', 'finance associate'],
    'hr executive': ['hr manager', 'hr specialist', 'human resources'],
    'media analyst': ['marketing analyst', 'digital marketing analyst']
}

def validate_resume_output(example, pred, trace=None):
    """
    Validation metric for DSPy optimization.

    Uses semantic matching for roles and skills, strict matching for education,
    and numerical tolerance for experience years.

    Args:
        example: Ground truth dspy.Example
        pred: Model prediction
        trace: Optional trace (unused)

    Returns:
        Score between 0 and 1
    """
    total_score = 0
    field_count = 0

    # Job Role: Enhanced matching with role synonyms
    if pred.job_role and example.job_role:
        field_count += 1
        pred_role = pred.job_role.strip().lower()
        gt_role = example.job_role.strip().lower()

        if pred_role == gt_role:
            total_score += 1.0
        else:
            # Check synonyms
            matched = False
            for main_role, synonyms in ROLE_SYNONYMS.items():
                if (main_role in gt_role or gt_role in main_role) and any(syn in pred_role for syn in synonyms):
                    total_score += 0.9
                    matched = True
                    break
                elif (main_role in pred_role or pred_role in main_role) and any(syn in gt_role for syn in synonyms):
                    total_score += 0.9
                    matched = True
                    break

            if not matched:
                # Partial word matching
                if any(word in pred_role for word in gt_role.split()) or any(word in gt_role for word in pred_role.split()):
                    total_score += 0.6

    # Skills: Enhanced semantic matching
    if pred.skills and pred.skills.strip():
        field_count += 1
        if hasattr(example, 'skills') and example.skills:
            skills_score = skills_match_score(pred.skills, example.skills)
            # Penalize over-generation to discourage generic/verbose outputs.
            pred_count = len(normalize_skills(pred.skills))
            gt_count = len(normalize_skills(example.skills))
            if pred_count > gt_count + 2:
                skills_score *= 0.9
            total_score += skills_score
        else:
            total_score += 0.1

    # Education: Strict matching
    if pred.education and example.education:
        field_count += 1
        p_edu = pred.education.strip().lower()
        g_edu = example.education.strip().lower()
        if p_edu == g_edu:
            total_score += 1.0
        elif p_edu in g_edu or g_edu in p_edu:
            total_score += 0.9
        elif any(word in p_edu for word in g_edu.split()) or any(word in g_edu for word in p_edu.split()):
            total_score += 0.7

    # Experience: Numerical matching with tolerance
    if pred.experience_years and example.experience_years:
        field_count += 1
        try:
            gt_exp = float(str(example.experience_years).strip())
            pred_exp = float(str(pred.experience_years).strip())
            if abs(gt_exp - pred_exp) < 0.05:
                total_score += 1.0
            elif abs(gt_exp - pred_exp) < 0.2:
                total_score += 0.8
            elif abs(gt_exp - pred_exp) < 0.5:
                total_score += 0.5
        except (ValueError, TypeError):
            pass

    return total_score / field_count if field_count > 0 else 0


# Alias for backward compatibility
enhanced_validate_resume_output = validate_resume_output
