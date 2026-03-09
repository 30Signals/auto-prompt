"""
Resume Extraction Evaluation

Detailed evaluation with per-field accuracy for resume extraction.
"""

from typing import List
import dspy
from shared.evaluation import EvaluationResult


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

            # Skills score (with semantic matching)
            skills_score = 0
            pred_skills = pred.skills
            if isinstance(pred_skills, list):
                pred_skills = pred_skills[0] if pred_skills else ''
            if pred_skills and hasattr(example, 'skills') and example.skills:
                gt_skills = set(s.strip().lower() for s in str(example.skills).split(','))
                pred_skills_list = set(s.strip().lower() for s in str(pred_skills).split(','))

                intersection = gt_skills.intersection(pred_skills_list)

                # Semantic matching
                semantic_matches = 0
                for gt_skill in gt_skills:
                    for pred_skill in pred_skills_list:
                        if (gt_skill in pred_skill or pred_skill in gt_skill or
                            (gt_skill == 'python' and 'python' in pred_skill) or
                            (gt_skill == 'sql' and 'sql' in pred_skill) or
                            (gt_skill == 'excel' and 'excel' in pred_skill) or
                            (gt_skill == 'tensorflow' and 'tensorflow' in pred_skill) or
                            (gt_skill == 'pytorch' and 'pytorch' in pred_skill) or
                            ('financial' in gt_skill and 'financial' in pred_skill) or
                            ('machine learning' in gt_skill and 'machine learning' in pred_skill)):
                            semantic_matches += 1
                            break

                union = gt_skills.union(pred_skills_list)
                if union:
                    skills_score = max(len(intersection) / len(union), semantic_matches / len(gt_skills))
            elif pred_skills and str(pred_skills).strip():
                skills_score = 0.3

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

    return EvaluationResult(
        name=name,
        results=results,
        field_accuracies=field_accuracies,
        overall_accuracy=overall_accuracy,
        total_samples=len(results)
    )
