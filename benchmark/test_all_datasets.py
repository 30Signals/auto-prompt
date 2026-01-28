import dspy
from . import config, loader, extractors, evaluation, optimizer
from pathlib import Path

def test_dataset(dataset_name, dataset_path):
    """Test a single dataset and return results"""
    print(f"\n{'='*60}")
    print(f"TESTING DATASET: {dataset_name}")
    print(f"{'='*60}")
    
    # Temporarily update config
    original_path = config.DATA_PATH
    config.DATA_PATH = dataset_path
    
    try:
        # Load data
        trainset, testset = loader.load_data()
        print(f"Train: {len(trainset)}, Test: {len(testset)}")
        
        # Baseline
        baseline = extractors.BaselineModule()
        baseline_results = evaluation.detailed_evaluation(baseline, testset, "Baseline")
        
        # DSPy optimization
        optimized_student = optimizer.optimize_model(trainset)
        optimized_results = evaluation.detailed_evaluation(optimized_student, testset, "DSPy")
        
        # Results
        print(f"\nRESULTS FOR {dataset_name}:")
        print(f"Baseline Overall: {baseline_results['overall_accuracy']:.2%}")
        print(f"DSPy Overall:     {optimized_results['overall_accuracy']:.2%}")
        print(f"Improvement:      {optimized_results['overall_accuracy'] - baseline_results['overall_accuracy']:.2%}")
        
        print(f"\nSkills Extraction:")
        print(f"Baseline: {baseline_results['field_accuracies']['skills']:.2%}")
        print(f"DSPy:     {optimized_results['field_accuracies']['skills']:.2%}")
        
        return {
            'dataset': dataset_name,
            'baseline_overall': baseline_results['overall_accuracy'],
            'dspy_overall': optimized_results['overall_accuracy'],
            'baseline_skills': baseline_results['field_accuracies']['skills'],
            'dspy_skills': optimized_results['field_accuracies']['skills']
        }
        
    except Exception as e:
        print(f"Error testing {dataset_name}: {e}")
        return None
    finally:
        # Restore original path
        config.DATA_PATH = original_path

def main():
    # Setup LLM
    llm_conf = config.get_llm_config()
    if llm_conf["provider"] == "azure":
        model_name = "azure/" + llm_conf["deployment"]
        lm = dspy.LM(
            model_name,
            api_key=llm_conf["api_key"],
            api_base=llm_conf["endpoint"],
            api_version=llm_conf["api_version"],
        )
    else:
        lm = dspy.Google(
            model=llm_conf["model"],
            api_key=llm_conf["api_key"]
        )
    dspy.settings.configure(lm=lm)
    
    # Define datasets
    base_dir = Path(__file__).parent.parent / "Data"
    datasets = [
        ("Enterprise Dataset", base_dir / "final_synthetic_100_resumes_enterprise.csv"),
        ("Final Resume Dataset", base_dir / "final_resume_dataset_enterprise.csv"),
        ("Messy Resume Dataset", base_dir / "Final_resume_messy.csv"),
        ("Long Resume Dataset", base_dir / "Resume_long1.csv")
    ]
    
    # Test all datasets
    all_results = []
    for name, path in datasets:
        if path.exists():
            result = test_dataset(name, path)
            if result:
                all_results.append(result)
        else:
            print(f"Dataset not found: {path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL DATASETS")
    print(f"{'='*80}")
    print(f"{'Dataset':<25} {'Baseline':<12} {'DSPy':<12} {'Improvement':<12}")
    print("-" * 65)
    
    for result in all_results:
        improvement = result['dspy_overall'] - result['baseline_overall']
        print(f"{result['dataset']:<25} {result['baseline_overall']:<12.2%} {result['dspy_overall']:<12.2%} {improvement:<12.2%}")

if __name__ == "__main__":
    main()