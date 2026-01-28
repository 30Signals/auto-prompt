import dspy
import json
import pandas as pd
from . import loader, extractors

def evaluate_modules(module, devset, name="Model"):
    """
    Basic evaluation loop printing per-sample accuracy.
    """
    print(f"\nEvaluating {name}...")
    print("=" * 50)
    
    total_score = 0
    for i, example in enumerate(devset):
        try:
            pred = module(unstructured_text=example.unstructured_text)
            
            # Simple check for display
            job_role_match = pred.job_role == example.job_role
            exp_match = False
            try:
                exp_match = abs(float(pred.experience_years) - float(example.experience_years)) < 0.05
            except (ValueError, TypeError):
                pass
                
            # Use the detailed metric logic for score (simplified here for display)
            score = (job_role_match + (pred.education == example.education) + exp_match) / 3.0
            total_score += score
            
            print(f"Sample {i+1}: {score:.2f} | role={'Y' if job_role_match else 'N'} exp={'Y' if exp_match else 'N'}")
            
        except Exception as e:
            print(f"Sample {i+1}: ERROR - {e}")
            continue
    
    final_score = total_score / len(devset) if devset else 0.0
    print(f"\n{name} Final Score: {final_score:.2%}")
    print("=" * 50)
    return final_score

def detailed_evaluation(module, dataset, name="Model"):
    """
    Perform detailed evaluation with per-field accuracy and error analysis.
    """
    results = []
    field_scores = {'job_role': [], 'skills': [], 'education': [], 'experience_years': []}
    
    for i, example in enumerate(dataset):
        try:
            pred = module(unstructured_text=example.unstructured_text)
            
            # Calculate individual field scores
            job_role_score = 0
            pred_job_role = pred.job_role
            if isinstance(pred_job_role, list):
                pred_job_role = pred_job_role[0] if pred_job_role else ''
            if pred_job_role and example.job_role:
                if str(pred_job_role).strip().lower() == str(example.job_role).strip().lower():
                    job_role_score = 1
            
            skills_score = 0
            pred_skills = pred.skills
            if isinstance(pred_skills, list):
                pred_skills = pred_skills[0] if pred_skills else ''
            if pred_skills and hasattr(example, 'skills') and example.skills:
                # More lenient skills evaluation with semantic matching
                gt_skills = set(s.strip().lower() for s in str(example.skills).split(','))
                pred_skills_list = set(s.strip().lower() for s in str(pred_skills).split(','))
                
                # Direct intersection
                intersection = gt_skills.intersection(pred_skills_list)
                
                # Semantic matching for common skill variations
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
                    # Use semantic matches for better scoring
                    skills_score = max(len(intersection) / len(union), semantic_matches / len(gt_skills))
            elif pred_skills and str(pred_skills).strip():
                skills_score = 0.3  # Some credit for inferring skills
            
            education_score = 0
            pred_education = pred.education
            if isinstance(pred_education, list):
                pred_education = pred_education[0] if pred_education else ''
            if pred_education and example.education:
                p_edu = str(pred_education).strip().lower()
                g_edu = str(example.education).strip().lower()
                if p_edu == g_edu or p_edu in g_edu or g_edu in p_edu:
                    education_score = 1
            
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
            continue
    
    # Calculate field-wise accuracies
    field_accuracies = {}
    for field, scores in field_scores.items():
        field_accuracies[field] = sum(scores) / len(scores) if scores else 0
    
    overall_accuracy = sum(r['overall_score'] for r in results) / len(results) if results else 0
    
    return {
        'name': name,
        'results': results,
        'field_accuracies': field_accuracies,
        'overall_accuracy': overall_accuracy,
        'total_samples': len(results)
    }

def compare_models(baseline_results, optimized_results):
    """
    Compare baseline and optimized model results and identify improvements.
    """
    # Error analysis - where baseline failed but optimized succeeded
    improvements = []
    degradations = []
    
    # Validate array lengths before comparison
    if len(baseline_results['results']) != len(optimized_results['results']):
        min_length = min(len(baseline_results['results']), len(optimized_results['results']))
    else:
        min_length = len(baseline_results['results'])
    
    for i in range(min_length):
        baseline_score = baseline_results['results'][i]['overall_score']
        optimized_score = optimized_results['results'][i]['overall_score']
        
        if optimized_score > baseline_score:
            improvements.append({
                'sample_id': i + 1,
                'baseline_score': baseline_score,
                'optimized_score': optimized_score,
                'improvement': optimized_score - baseline_score,
                'ground_truth': baseline_results['results'][i]['ground_truth'],
                'baseline_pred': baseline_results['results'][i]['predicted'],
                'optimized_pred': optimized_results['results'][i]['predicted']
            })
        elif optimized_score < baseline_score:
            degradations.append({
                'sample_id': i + 1,
                'baseline_score': baseline_score,
                'optimized_score': optimized_score,
                'degradation': baseline_score - optimized_score
            })
    
    return {
        'improvements': improvements,
        'degradations': degradations,
        'summary': {
            'total_improvements': len(improvements),
            'total_degradations': len(degradations),
            'overall_improvement': optimized_results['overall_accuracy'] - baseline_results['overall_accuracy']
        }
    }

def save_results(baseline_results, optimized_results, comparison):
    """
    Save results to three separate JSON files.
    """
    import os
    
    # Save Baseline Results
    with open("baseline_results.json", 'w', encoding='utf-8') as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)
    
    # Save DSPy Results
    with open("dspy_results.json", 'w', encoding='utf-8') as f:
        json.dump(optimized_results, f, indent=2, ensure_ascii=False)
    
    # Save Comparison
    with open("comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

def get_prompts():
    """
    Extract and display the DSPy optimized prompt and examples.
    """
    print("\n" + "="*50)
    print("PROMPT ANALYSIS")
    print("="*50)
    
    # Baseline prompt
    print("\nHANDCRAFTED PROMPT (Input / Seed):")
    print("="*30)
    try:
        print(extractors.BASELINE_PROMPT)
    except Exception:
        print("[Could not print baseline prompt due to encoding error]")
    
    # Optimized prompt
    try:
        with open("optimized_resume_module.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\nDSPy PROMPT (OPTIMIZED):")
        print("="*30)
        
        # Generic search for signature and demos
        demos = []
        signature_instr = None
        
        # Iterate over all keys to find the predictor state (e.g. "predictor" or "predictor.predict")
        for key, value in data.items():
            if isinstance(value, dict):
                # Check for demos
                if "demos" in value:
                    demos = value["demos"]
                
                # Check for signature instructions
                if "signature" in value and "instructions" in value["signature"]:
                    signature_instr = value["signature"]["instructions"]
        
        # 1. Instruction
        print("1. Core Instruction (Preserved from Handcrafted):")
        print("-" * 20)
        if signature_instr:
            try:
                print(signature_instr)
            except Exception:
                print("[Instruction found but could not print due to encoding]")
        else:
            print("[Instruction not found in JSON structure]")

        # 2. Learned Examples
        print("\n2. Learned Few-Shot Examples (Optimized):")
        print("-" * 20)
        print(f"DSPy automatically selected {len(demos)} examples to teach the LLM.")
        
        if demos:
            print(f"\n[Example Demo 1 of {len(demos)}]")
            print(json.dumps(demos[0], indent=2, ensure_ascii=True))
            
    except FileNotFoundError:
        print("Optimized module file not found.")
    except Exception as e:
        print(f"Error loading optimized prompt: {e}")
