import dspy
import argparse
from . import config, loader, extractors, evaluation, optimizer

def main():
    # 1. Setup LLM
    print("Setting up LLM...")
    llm_conf = config.get_llm_config()
    
    if llm_conf["provider"] == "azure":
        # DSPy 3.1.2 uses dspy.LM with LiteLLM model strings
        # "azure/<deployment_name>"
        model_name = "azure/" + llm_conf["deployment"]
        lm = dspy.LM(
            model_name,
            api_key=llm_conf["api_key"],
            api_base=llm_conf["endpoint"],
            api_version=llm_conf["api_version"],
        )
    else:
        # Fallback or other providers
        lm = dspy.Google(
            model=llm_conf["model"],
            api_key=llm_conf["api_key"]
        )

    dspy.settings.configure(lm=lm)
    
    # 2. Load Data
    print("Loading data...")
    trainset, testset = loader.load_data()
    print(f"Train: {len(trainset)}, Test: {len(testset)}")

    # 3. Baseline Evaluation
    print("\n--- Baseline Evaluation ---")
    baseline = extractors.BaselineModule()
    baseline_score = evaluation.evaluate_modules(baseline, testset, name="Baseline")
    
    # Detailed baseline evaluation
    try:
        baseline_results = evaluation.detailed_evaluation(baseline, testset, "Baseline")
    except Exception as e:
        print(f"Error in baseline evaluation: {e}")
        baseline_results = None

    # 4. Optimization
    print("\n--- DSPy Optimization ---")
    # Optimize on TRAIN set
    optimized_student = optimizer.optimize_model(trainset)
    
    # 5. Final Evaluation
    print("\n--- Optimized Evaluation ---")
    optimized_score = evaluation.evaluate_modules(optimized_student, testset, name="Optimized Student")
    
    # Detailed optimized evaluation
    try:
        optimized_results = evaluation.detailed_evaluation(optimized_student, testset, "Optimized")
    except Exception as e:
        print(f"Error in optimized evaluation: {e}")
        optimized_results = None

    # 6. Comprehensive Analysis
    if baseline_results and optimized_results:
        comparison = evaluation.compare_models(baseline_results, optimized_results)
        evaluation.save_results(baseline_results, optimized_results, comparison)
    else:
        print("Skipping detailed analysis due to errors")
    
    # Save optimized program
    optimized_student.save("optimized_resume_module.json")
    print("\nOptimized module saved to 'optimized_resume_module.json'")
    
    print("\n✅ Experiment Completed Successfully. See baseline_results.json, dspy_results.json, and comparison_results.json for details.")

if __name__ == "__main__":
    main()
