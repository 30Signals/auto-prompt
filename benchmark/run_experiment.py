import dspy
import argparse
from . import config, loader, extractors, evaluation, optimizer

def main():
    # 1. Setup LLM
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
    
    # 2. Load Data
    trainset, testset = loader.load_data()

    # 3. Baseline Evaluation
    baseline = extractors.BaselineModule()
    baseline_results = evaluation.detailed_evaluation(baseline, testset, "Baseline")

    # 4. Optimization
    optimized_student = optimizer.optimize_model(trainset)
    
    # 5. Final Evaluation
    optimized_results = evaluation.detailed_evaluation(optimized_student, testset, "DSPy")

    # 6. Results Summary
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Baseline Overall Accuracy:  {baseline_results['overall_accuracy']:.2%}")
    print(f"DSPy Overall Accuracy:      {optimized_results['overall_accuracy']:.2%}")
    print(f"Improvement:                {optimized_results['overall_accuracy'] - baseline_results['overall_accuracy']:.2%}")
    
    print(f"\nSkills Extraction:")
    print(f"Baseline:  {baseline_results['field_accuracies']['skills']:.2%}")
    print(f"DSPy:      {optimized_results['field_accuracies']['skills']:.2%}")
    print(f"Improvement: {optimized_results['field_accuracies']['skills'] - baseline_results['field_accuracies']['skills']:.2%}")
    
    # Save results
    comparison = evaluation.compare_models(baseline_results, optimized_results)
    evaluation.save_results(baseline_results, optimized_results, comparison)
    
    # Save optimized program
    optimized_student.save("optimized_resume_module.json")
    print("\nResults saved to JSON files.")

if __name__ == "__main__":
    main()