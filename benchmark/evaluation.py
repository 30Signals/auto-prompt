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
            score = (job_role_match + (pred.skills == example.skills) + (pred.education == example.education) + exp_match) / 4.0
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
    
    print(f"\nDetailed evaluation of {name}...")
    
    for i, example in enumerate(dataset):
        try:
            pred = module(unstructured_text=example.unstructured_text)
            
            # Calculate individual field scores
            job_role_score = 0
            if pred.job_role and example.job_role:
                if pred.job_role.strip().lower() == example.job_role.strip().lower():
                    job_role_score = 1
            
            skills_score = 0
            if pred.skills and example.skills:
                gt_skills = set(s.strip().lower() for s in example.skills.split(','))
                pred_skills = set(s.strip().lower() for s in pred.skills.split(','))
                intersection = gt_skills.intersection(pred_skills)
                union = gt_skills.union(pred_skills)
                if union:
                    skills_score = len(intersection) / len(union)
            
            education_score = 0
            if pred.education and example.education:
                p_edu = pred.education.strip().lower()
                g_edu = example.education.strip().lower()
                if p_edu == g_edu or p_edu in g_edu or g_edu in p_edu:
                    education_score = 1
            
            exp_score = 0
            try:
                gt_exp = float(str(example.experience_years).strip())
                pred_exp = float(str(pred.experience_years).strip())
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
                    'skills': example.skills,
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
            print(f"Error processing sample {i+1}: {e}")
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
    print("\n" + "="*50)
    print("DETAILED COMPARISON REPORT")
    print("="*50)
    
    # Overall comparison
    print(f"\nOVERALL ACCURACY:")
    print(f"Baseline:  {baseline_results['overall_accuracy']:.2%}")
    print(f"Optimized: {optimized_results['overall_accuracy']:.2%}")
    print(f"Improvement: {optimized_results['overall_accuracy'] - baseline_results['overall_accuracy']:.2%}")
    
    # Field-wise comparison
    print(f"\nFIELD-WISE ACCURACY:")
    print(f"{'Field':<20} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12}")
    print("-" * 56)
    
    for field in ['job_role', 'skills', 'education', 'experience_years']:
        baseline_acc = baseline_results['field_accuracies'][field]
        optimized_acc = optimized_results['field_accuracies'][field]
        improvement = optimized_acc - baseline_acc
        
        print(f"{field:<20} {baseline_acc:<12.2%} {optimized_acc:<12.2%} {improvement:<12.2%}")
    
    # Error analysis - where baseline failed but optimized succeeded
    print(f"\nERROR ANALYSIS:")
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
    
    print(f"\nSamples where optimized model improved over baseline: {len(improvements)}")
    print(f"Samples where optimized model performed worse: {len(degradations)}")
    
    if improvements:
        print(f"\nTOP IMPROVEMENTS:")
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        for imp in improvements[:3]:
            print(f"\nSample {imp['sample_id']} (Improvement: {imp['improvement']:.2f})")
            print(f"  Ground Truth: {imp['ground_truth']}")
            print(f"  Baseline:     {imp['baseline_pred']}")
            print(f"  Optimized:    {imp['optimized_pred']}")
    
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
    print("\nBaseline results saved to 'baseline_results.json'")
    
    # Save DSPy Results
    with open("dspy_results.json", 'w', encoding='utf-8') as f:
        json.dump(optimized_results, f, indent=2, ensure_ascii=False)
    print("DSPy results saved to 'dspy_results.json'")
    
    # Save Comparison
    with open("comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print("Comparison results saved to 'comparison_results.json'")

def get_prompts():
    """
    Extract and display the baseline and optimized prompts.
    """
    print("\n" + "="*50)
    print("PROMPT ANALYSIS")
    print("="*50)
    
    # Baseline prompt (from signature)
    print("\nHANDCRAFTED PROMPT:")
    print("="*20)
    print(extractors.BASELINE_PROMPT)
    
    # Try to load optimized prompt
    try:
        with open("optimized_resume_module.json", 'r') as f:
            optimized_data = json.load(f)
        
        print("\nOPTIMIZED PROMPT (GENERATED):")
        print("="*20)
        
        if 'predictor' in optimized_data and 'demos' in optimized_data['predictor']:
            demos = optimized_data['predictor']['demos']
            print(f"Number of few-shot examples selected: {len(demos)}")
            if demos:
                print("\nExample Demo 1:")
                print(str(demos[0])[:200] + "...")
        
    except FileNotFoundError:
        print("Optimized module file not found.")
    except Exception as e:
        print(f"Error loading optimized prompt: {e}")
