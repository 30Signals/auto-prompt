#!/usr/bin/env python3
"""
Run All Experiments

Master script to run all experiments and generate comparison reports.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 70)
    print("AUTO-PROMPT: Running All Experiments")
    print("=" * 70)

    experiments = []

    print("\n" + "-" * 70)
    print("Running Experiment 1: Resume Extraction")
    print("-" * 70)
    try:
        from experiments.resume_extraction.run import run_experiment

        results = run_experiment(save_results=True)
        experiments.append(
            {
                "name": "Resume Extraction",
                "status": "success",
                "baseline_accuracy": results["baseline_results"].overall_accuracy,
                "optimized_accuracy": results["optimized_results"].overall_accuracy,
                "improvement": results["comparison"]["summary"]["overall_improvement"],
            }
        )
    except Exception as e:
        print(f"Error running Resume Extraction experiment: {e}")
        experiments.append({"name": "Resume Extraction", "status": "failed", "error": str(e)})

    print("\n" + "-" * 70)
    print("Running Experiment 2: Company Legal Risk")
    print("-" * 70)
    try:
        from experiments.company_legal_risk.run import run_experiment

        results = run_experiment(save_results=True)
        experiments.append(
            {
                "name": "Company Legal Risk",
                "status": "success",
                "baseline_accuracy": results["baseline_results"].overall_accuracy,
                "optimized_accuracy": results["optimized_results"].overall_accuracy,
                "improvement": results["comparison"]["summary"]["overall_improvement"],
            }
        )
    except Exception as e:
        print(f"Error running Company Legal Risk experiment: {e}")
        experiments.append({"name": "Company Legal Risk", "status": "failed", "error": str(e)})

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for exp in experiments:
        if exp["status"] == "success":
            print(f"\n{exp['name']}:")
            print(f"  Baseline: {exp['baseline_accuracy']:.2%}")
            print(f"  Optimized: {exp['optimized_accuracy']:.2%}")
            print(f"  Improvement: {exp['improvement']:.2%}")
        else:
            print(f"\n{exp['name']}: FAILED - {exp.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
