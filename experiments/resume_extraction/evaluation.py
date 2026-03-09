"""
Resume Extraction Evaluation

Detailed evaluation with per-field accuracy for resume extraction.
"""

from typing import List
import dspy
from shared.evaluation import EvaluationResult
from .skill_utils import skills_match_score, normalize_skills


def detailed_evaluation(module: dspy.Module, dataset: List, name: str = "Model") -> EvaluationResult:
    """
    Perform detailed evaluation with per-field accuracy.

    Args:
        module: DSPy module to evaluate
        dataset: List of dspy.Example objects
        name: Name for this evaluation run

    Returns:
        EvaluationResult with detailed metrics
    """
    results = []
    field_scores = {
        'job_role': [],
        'skills': [],
        'education': [],
        'experience_years': []
    }
    skill_tp_total = 0
    skill_fp_total = 0
    skill_fn_total = 0
    skill_precision_scores = []
    skill_recall_scores = []
    skill_f1_scores = []

    for i, example in enumerate(dataset):
        try:
            pred = module(unstructured_text=example.unstructured_text)

            # Job role score
            job_role_score = 0
            pred_job_role = pred.job_role
            if isinstance(pred_job_role, list):
                pred_job_role = pred_job_role[0] if pred_job_role else ''
            if pred_job_role and example.job_role:
                if str(pred_job_role).strip().lower() == str(example.job_role).strip().lower():
                    job_role_score = 1

            # Skills score (aligned with optimization metric)
            skills_score = 0
            skill_precision = 0.0
            skill_recall = 0.0
            skill_f1 = 0.0
            pred_skills = pred.skills
            if isinstance(pred_skills, list):
                pred_skills = pred_skills[0] if pred_skills else ''
            if pred_skills and hasattr(example, 'skills') and example.skills:
                skills_score = skills_match_score(pred_skills, example.skills)
                gt_set = set(normalize_skills(example.skills))
                pred_set = set(normalize_skills(pred_skills))
                tp = len(gt_set.intersection(pred_set))
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                skill_tp_total += tp
                skill_fp_total += fp
                skill_fn_total += fn
                skill_precision = tp / len(pred_set) if pred_set else 0.0
                skill_recall = tp / len(gt_set) if gt_set else 0.0
                if skill_precision + skill_recall > 0:
                    skill_f1 = 2 * skill_precision * skill_recall / (skill_precision + skill_recall)
                else:
                    skill_f1 = 0.0
                skill_precision_scores.append(skill_precision)
                skill_recall_scores.append(skill_recall)
                skill_f1_scores.append(skill_f1)
            elif pred_skills and str(pred_skills).strip():
                skills_score = 0.1

            # Education score
            education_score = 0
            pred_education = pred.education
            if isinstance(pred_education, list):
                pred_education = pred_education[0] if pred_education else ''
            if pred_education and example.education:
                p_edu = str(pred_education).strip().lower()
                g_edu = str(example.education).strip().lower()
                if p_edu == g_edu or p_edu in g_edu or g_edu in p_edu:
                    education_score = 1

            # Experience score
            exp_score = 0
            try:
                gt_exp = float(str(example.experience_years).strip())
                pred_exp_str = pred.experience_years
                if isinstance(pred_exp_str, list):
                    pred_exp_str = pred_exp_str[0] if pred_exp_str else '0'
                pred_exp = float(str(pred_exp_str).strip())
                if abs(gt_exp - pred_exp) < 0.5:
                    exp_score = 1
            except (ValueError, TypeError):
                pass

            # Store scores
            field_scores['job_role'].append(job_role_score)
            field_scores['skills'].append(skills_score)
            field_scores['education'].append(education_score)
            field_scores['experience_years'].append(exp_score)

            # Store detailed result
            result = {
                'sample_id': i + 1,
                'ground_truth': {
                    'job_role': example.job_role,
                    'skills': getattr(example, 'skills', 'N/A'),
                    'education': example.education,
                    'experience_years': example.experience_years
                },
                'predicted': {
                    'job_role': pred.job_role,
                    'skills': pred.skills,
                    'education': pred.education,
                    'experience_years': pred.experience_years
                },
                'scores': {
                    'job_role': job_role_score,
                    'skills': skills_score,
                    'skills_precision': skill_precision,
                    'skills_recall': skill_recall,
                    'skills_f1': skill_f1,
                    'education': education_score,
                    'experience_years': exp_score
                },
                'overall_score': (job_role_score + skills_score + education_score + exp_score) / 4.0
            }
            results.append(result)

        except Exception as e:
            print(f"Error evaluating sample {i + 1}: {e}")
            continue

    # Calculate field-wise accuracies
    field_accuracies = {}
    for field, scores in field_scores.items():
        field_accuracies[field] = sum(scores) / len(scores) if scores else 0

    overall_accuracy = sum(r['overall_score'] for r in results) / len(results) if results else 0

    micro_precision = skill_tp_total / (skill_tp_total + skill_fp_total) if (skill_tp_total + skill_fp_total) > 0 else 0.0
    micro_recall = skill_tp_total / (skill_tp_total + skill_fn_total) if (skill_tp_total + skill_fn_total) > 0 else 0.0
    if micro_precision + micro_recall > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = 0.0

    metadata = {
        'skills_metrics': {
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': sum(skill_precision_scores) / len(skill_precision_scores) if skill_precision_scores else 0.0,
            'macro_recall': sum(skill_recall_scores) / len(skill_recall_scores) if skill_recall_scores else 0.0,
            'macro_f1': sum(skill_f1_scores) / len(skill_f1_scores) if skill_f1_scores else 0.0,
            'tp_total': skill_tp_total,
            'fp_total': skill_fp_total,
            'fn_total': skill_fn_total
        }
    }

    return EvaluationResult(
        name=name,
        results=results,
        field_accuracies=field_accuracies,
        overall_accuracy=overall_accuracy,
        total_samples=len(results),
        metadata=metadata
    )
