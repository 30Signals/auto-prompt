"""
Prompt Extraction and Storage Utilities

Utilities for saving baseline prompts and extracting optimized prompts from DSPy modules.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


def save_baseline_prompt(baseline_prompt_path: Path, results_dir: Path):
    """
    Copy baseline prompt to results directory.

    Args:
        baseline_prompt_path: Path to the baseline prompt file
        results_dir: Results directory to save prompts to
    """
    if not baseline_prompt_path.exists():
        print(f"Warning: Baseline prompt not found at {baseline_prompt_path}")
        return

    dst = results_dir / "prompts" / "baseline_prompt.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(baseline_prompt_path, dst)


def extract_optimized_prompt(optimized_module, results_dir: Path) -> Optional[str]:
    """
    Extract prompt from DSPy module and save to file.

    Args:
        optimized_module: Optimized DSPy module
        results_dir: Results directory to save prompts to

    Returns:
        Extracted prompt text, or None if extraction failed
    """
    try:
        # Save module to get JSON representation
        module_dict = optimized_module.save()

        # Parse the module structure to extract prompt components
        prompt_text = format_dspy_prompt(module_dict, optimized_module)

        # Save to file
        dst = results_dir / "prompts" / "optimized_prompt.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(prompt_text)

        return prompt_text
    except Exception as e:
        print(f"Warning: Could not extract optimized prompt: {e}")
        return None


def format_dspy_prompt(module_dict: Dict[str, Any], optimized_module) -> str:
    """
    Format DSPy module dictionary into readable prompt text.

    Args:
        module_dict: Module dictionary from save()
        optimized_module: The optimized module object

    Returns:
        Formatted prompt text
    """
    lines = []
    lines.append("# Optimized DSPy Prompt")
    lines.append("=" * 60)
    lines.append("")

    # Extract predictor information
    if hasattr(optimized_module, 'predictor'):
        predictor = optimized_module.predictor

        # Signature
        if hasattr(predictor, 'signature'):
            sig = predictor.signature
            lines.append("## Signature")
            lines.append(f"Input: {', '.join(sig.input_fields.keys())}")
            lines.append(f"Output: {', '.join(sig.output_fields.keys())}")
            lines.append("")

            # Instructions
            if hasattr(sig, 'instructions'):
                lines.append("## Instructions")
                lines.append(sig.instructions)
                lines.append("")

        # Few-shot demonstrations
        if hasattr(predictor, 'demos') and predictor.demos:
            lines.append("## Few-Shot Examples")
            lines.append(f"Number of examples: {len(predictor.demos)}")
            lines.append("")

            for i, demo in enumerate(predictor.demos[:3], 1):  # Show first 3 examples
                lines.append(f"### Example {i}")
                if hasattr(demo, 'inputs'):
                    lines.append(f"Input: {demo.inputs()}")
                if hasattr(demo, 'labels'):
                    lines.append(f"Output: {demo.labels()}")
                lines.append("")

            if len(predictor.demos) > 3:
                lines.append(f"... and {len(predictor.demos) - 3} more examples")
                lines.append("")

    # Extended rationale (from ChainOfThought)
    lines.append("## Reasoning Strategy")
    lines.append("Uses Chain-of-Thought reasoning with intermediate rationale steps")
    lines.append("")

    return "\n".join(lines)


def count_examples(prompt_text: str) -> int:
    """
    Count number of examples in a prompt.

    Args:
        prompt_text: Prompt text to analyze

    Returns:
        Number of examples
    """
    return prompt_text.count("### Example")


def generate_prompt_comparison(results_dir: Path):
    """
    Create markdown with side-by-side prompt comparison.

    Args:
        results_dir: Results directory containing prompt files
    """
    baseline_path = results_dir / "prompts" / "baseline_prompt.txt"
    optimized_path = results_dir / "prompts" / "optimized_prompt.txt"

    baseline_text = ""
    optimized_text = ""

    if baseline_path.exists():
        baseline_text = baseline_path.read_text()
    else:
        baseline_text = "Baseline prompt not found"

    if optimized_path.exists():
        optimized_text = optimized_path.read_text()
        num_examples = count_examples(optimized_text)
    else:
        optimized_text = "Optimized prompt not found"
        num_examples = 0

    comparison = f"""# Prompt Comparison

## Baseline Prompt (Handcrafted)

```
{baseline_text}
```

## Optimized Prompt (DSPy)

```
{optimized_text}
```

## Key Differences

- **Few-shot examples**: 0 (baseline) vs {num_examples} (optimized)
- **Reasoning**: Implicit (baseline) vs Explicit Chain-of-Thought (optimized)
- **Optimization**: Manual (baseline) vs Automatic via DSPy (optimized)

## Notes

The optimized prompt is generated through DSPy's two-stage optimization:
1. BootstrapFewShot: Selects effective few-shot examples from training data
2. COPRO: Refines instructions and example selection

The baseline prompt is handcrafted based on domain knowledge and best practices.
"""

    dst = results_dir / "prompts" / "prompt_comparison.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(comparison)
