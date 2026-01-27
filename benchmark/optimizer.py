import dspy
from dspy.teleprompt import BootstrapFewShot
from .extractors import StudentModule

def validate_resume_output(example, pred, trace=None):
    # Job Role: Case-insensitive normalized match
    job_role_score = 0
    if pred.job_role and example.job_role:
        if pred.job_role.strip().lower() == example.job_role.strip().lower():
            job_role_score = 1
            
    # Education: Case-insensitive normalized match
    education_score = 0
    if pred.education and example.education:
        # Simple containment or exact match
        p_edu = pred.education.strip().lower()
        g_edu = example.education.strip().lower()
        if p_edu == g_edu or p_edu in g_edu or g_edu in p_edu:
            education_score = 1

    # Skills: Jaccard Similarity (Intersection over Union)
    skills_score = 0
    if pred.skills and example.skills:
        gt_skills = set(s.strip().lower() for s in example.skills.split(','))
        pred_skills = set(s.strip().lower() for s in pred.skills.split(','))
        intersection = gt_skills.intersection(pred_skills)
        union = gt_skills.union(pred_skills)
        if union:
            skills_score = len(intersection) / len(union)
            
    # Experience: Numeric threshold
    exp_score = 0
    try:
        gt_exp = float(str(example.experience_years).strip())
        pred_exp = float(str(pred.experience_years).strip())
        if abs(gt_exp - pred_exp) < 0.5:  # Slightly more lenient for optimization
            exp_score = 1
    except (ValueError, TypeError):
        pass
        
    return (job_role_score + skills_score + education_score + exp_score) / 4.0
def optimize_model(trainset):
    print("\nOptimizing Student Module with BootstrapFewShot...")
    teleprompter = BootstrapFewShot(metric=validate_resume_output, max_bootstrapped_demos=4, max_labeled_demos=4)
    student = StudentModule()
    optimized_student = teleprompter.compile(student, trainset=trainset)
    return optimized_student
