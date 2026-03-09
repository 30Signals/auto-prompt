"""
Run multiple replay evaluation trials on a fixed labeled replay dataset.
"""

import argparse
import json
from pathlib import Path

from .replay_eval import evaluate_replay


def run_multi(replay_file, optimized_module, output_root, num_runs=5):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    trial_scores = []

    for i in range(num_runs):
        trial_dir = output_root / f"trial_{i}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        result = evaluate_replay(
            replay_file=replay_file,
            optimized_module_path=optimized_module,
            output_dir=trial_dir,
        )

        baseline_score = result["baseline_results"].overall_accuracy
        optimized_score = result["optimized_results"].overall_accuracy
        delta = optimized_score - baseline_score

        trial_scores.append(
            {
                "trial": i,
                "baseline_overall": baseline_score,
                "optimized_overall": optimized_score,
                "delta": delta,
            }
        )

        print(f"Trial {i}: baseline={baseline_score:.2%}, optimized={optimized_score:.2%}, delta={delta:+.2%}")

    mean_baseline = sum(x["baseline_overall"] for x in trial_scores) / len(trial_scores)
    mean_optimized = sum(x["optimized_overall"] for x in trial_scores) / len(trial_scores)
    mean_delta = sum(x["delta"] for x in trial_scores) / len(trial_scores)

    summary = {
        "num_runs": num_runs,
        "replay_file": replay_file,
        "optimized_module": optimized_module,
        "mean_baseline_overall": mean_baseline,
        "mean_optimized_overall": mean_optimized,
        "mean_delta": mean_delta,
        "trials": trial_scores,
    }

    out_file = output_root / "multi_run_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("REPLAY MULTI-RUN SUMMARY")
    print("=" * 70)
    print(f"Runs: {num_runs}")
    print(f"Baseline mean:  {mean_baseline:.2%}")
    print(f"Optimized mean: {mean_optimized:.2%}")
    print(f"Delta mean:     {mean_delta:+.2%}")
    print(f"Saved: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-run replay evaluation")
    parser.add_argument(
        "--replay-file",
        default="experiments/company_legal_risk/data/v1/retrieval_replay_labeling_queue_v1.json",
        help="Labeled replay dataset file",
    )
    parser.add_argument(
        "--optimized-module",
        default="experiments/company_legal_risk/results/optimized_module.json",
        help="Optimized DSPy module path",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/company_legal_risk/results/replay_multi",
        help="Output root for trial artifacts",
    )
    parser.add_argument("--num-runs", type=int, default=5, help="Number of replay trials")

    args = parser.parse_args()

    run_multi(
        replay_file=args.replay_file,
        optimized_module=args.optimized_module,
        output_root=args.output_dir,
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    main()
