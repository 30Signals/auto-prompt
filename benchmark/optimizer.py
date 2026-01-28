import dspy
from dspy.teleprompt import BootstrapFewShot, COPRO
from .extractors import StudentModule, ResumeSignature, EnhancedResumeSignature, BASELINE_PROMPT

def enhanced_validate_resume_output(example, pred, trace=None):
    """Enhanced validation with stricter criteria and better semantic matching"""
    total_score = 0
    field_count = 0
    
    # Job Role: Enhanced matching with role synonyms
    if pred.job_role and example.job_role:
        field_count += 1
        pred_role = pred.job_role.strip().lower()
        gt_role = example.job_role.strip().lower()
        
        # Role synonyms mapping
        role_synonyms = {
            'data scientist': ['data analyst', 'ml engineer', 'ai engineer'],
            'software engineer': ['developer', 'programmer', 'software developer'],
            'finance analyst': ['financial analyst', 'finance associate'],
            'hr executive': ['hr manager', 'hr specialist', 'human resources'],
            'media analyst': ['marketing analyst', 'digital marketing analyst']
        }
        
        if pred_role == gt_role:
            total_score += 1.0
        else:
            # Check synonyms
            for main_role, synonyms in role_synonyms.items():
                if (main_role in gt_role or gt_role in main_role) and any(syn in pred_role for syn in synonyms):
                    total_score += 0.9
                    break
                elif (main_role in pred_role or pred_role in main_role) and any(syn in gt_role for syn in synonyms):
                    total_score += 0.9
                    break
            else:
                # Partial word matching
                if any(word in pred_role for word in gt_role.split()) or any(word in gt_role for word in pred_role.split()):
                    total_score += 0.6
    
    # Skills: Enhanced semantic matching with skill categories
    if pred.skills and pred.skills.strip():
        field_count += 1
        if hasattr(example, 'skills') and example.skills:
            gt_skills = set(s.strip().lower() for s in str(example.skills).split(','))
            pred_skills_list = [s.strip().lower() for s in str(pred.skills).split(',')]
            
            # Skill category mapping for better matching
            skill_categories = {
                'python': ['programming', 'coding', 'development'],
                'sql': ['database', 'data querying', 'data management'],
                'machine learning': ['ml', 'ai', 'predictive modeling', 'data science'],
                'excel': ['spreadsheet', 'data analysis', 'financial modeling'],
                'recruitment': ['hiring', 'talent acquisition', 'hr'],
                'seo': ['search optimization', 'digital marketing'],
                'java': ['programming', 'backend development', 'software development']
            }
            
            matched_skills = 0
            for gt_skill in gt_skills:
                best_match = 0
                for pred_skill in pred_skills_list:
                    # Exact match
                    if gt_skill == pred_skill:
                        best_match = 1.0
                        break
                    # Containment match
                    elif gt_skill in pred_skill or pred_skill in gt_skill:
                        best_match = max(best_match, 0.8)
                    # Category match
                    else:
                        for category, related in skill_categories.items():
                            if (category in gt_skill and any(r in pred_skill for r in related)) or \
                               (category in pred_skill and any(r in gt_skill for r in related)):
                                best_match = max(best_match, 0.7)
                
                matched_skills += best_match
            
            skills_score = matched_skills / len(gt_skills) if gt_skills else 0
            total_score += skills_score
        else:
            total_score += 0.2  # Minimal credit for inferring skills
    
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
    
    # Experience: Strict numerical matching
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

def validate_resume_output(example, pred, trace=None):
    """Legacy validation function for compatibility"""
    return enhanced_validate_resume_output(example, pred, trace)

def optimize_model(trainset):
    # Create enhanced student
    student = StudentModule()
    
    # Stage 1: Enhanced BootstrapFewShot with maximum settings
    bootstrap_teleprompter = BootstrapFewShot(
        metric=enhanced_validate_resume_output,
        max_bootstrapped_demos=20,  # Increased from 16
        max_labeled_demos=20,       # Increased from 16
        max_rounds=6,               # Increased from 4
        max_errors=10               # Allow more errors for better exploration
    )
    
    stage1_optimized = bootstrap_teleprompter.compile(student, trainset=trainset)
    
    # Stage 2: COPRO optimization for prompt refinement
    try:
        copro_teleprompter = COPRO(
            metric=enhanced_validate_resume_output,
            breadth=10,
            depth=3,
            init_temperature=1.4
        )
        final_optimized = copro_teleprompter.compile(stage1_optimized, trainset=trainset)
        return final_optimized
    except Exception:
        return stage1_optimized
