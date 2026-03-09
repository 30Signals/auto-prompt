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
        # First, load from the saved JSON file which has the full structure
        module_json_path = results_dir / "optimized_module.json"
        if module_json_path.exists():
            module_dict = _load_json_with_fallback(module_json_path)
            prompt_text = format_dspy_prompt_from_json(module_dict)
        else:
            # Fallback to extracting from module object
            prompt_text = format_dspy_prompt_from_module(optimized_module)

        # Save to file
        dst = results_dir / "prompts" / "optimized_prompt.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(prompt_text, encoding="utf-8", errors="replace")

        return prompt_text
    except Exception as e:
        print(f"Warning: Could not extract optimized prompt: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_json_with_fallback(path: Path) -> Dict[str, Any]:
    """
    Load JSON with robust decoding for Windows/local code-page variability.
    """
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not decode JSON file: {path}")


def format_dspy_prompt_from_json(module_dict: Dict[str, Any]) -> str:
    """
    Format DSPy module from saved JSON structure into readable prompt text.

    Args:
        module_dict: Module dictionary loaded from JSON file

    Returns:
        Formatted prompt text
    """
    lines = []
    lines.append("# Optimized DSPy Prompt")
    lines.append("=" * 60)
    lines.append("")

    # Find the predictor key (usually "predictor.predict" or similar)
    predictor_key = None
    for key in module_dict.keys():
        if key.startswith("predictor") or "predict" in key.lower():
            predictor_key = key
            break

    if predictor_key and predictor_key in module_dict:
        predictor_data = module_dict[predictor_key]

        # Signature and Instructions
        if 'signature' in predictor_data:
            sig = predictor_data['signature']

            # Instructions
            if 'instructions' in sig and sig['instructions']:
                lines.append("## Instructions")
                lines.append("")
                lines.append("```")
                lines.append(sig['instructions'])
                lines.append("```")
                lines.append("")

            # Fields
            if 'fields' in sig and sig['fields']:
                lines.append("## Field Definitions")
                lines.append("")
                for field in sig['fields']:
                    prefix = field.get('prefix', 'Unknown')
                    desc = field.get('description', '')
                    lines.append(f"- **{prefix}** {desc}")
                lines.append("")

        # Few-shot demonstrations
        if 'demos' in predictor_data and predictor_data['demos']:
            demos = predictor_data['demos']
            lines.append("## Few-Shot Examples")
            lines.append(f"")
            lines.append(f"Total examples: **{len(demos)}**")
            lines.append("")

            # Show first 3 examples
            for i, demo in enumerate(demos[:3], 1):
                lines.append(f"### Example {i}")
                lines.append("")

                # Input text
                if 'unstructured_text' in demo:
                    text = demo['unstructured_text']
                    if len(text) > 200:
                        text = text[:200] + "..."
                    lines.append(f"**Input (truncated):**")
                    lines.append(f"```")
                    lines.append(text)
                    lines.append(f"```")
                    lines.append("")

                # Reasoning (Chain-of-Thought)
                if 'reasoning' in demo:
                    reasoning = demo['reasoning']
                    if len(reasoning) > 300:
                        reasoning = reasoning[:300] + "..."
                    lines.append(f"**Reasoning:**")
                    lines.append(f"> {reasoning}")
                    lines.append("")

                # Output fields
                lines.append("**Output:**")
                output_fields = ['job_role', 'skills', 'education', 'experience_years']
                for field in output_fields:
                    if field in demo:
                        lines.append(f"- {field}: `{demo[field]}`")
                lines.append("")

            if len(demos) > 3:
                lines.append(f"... and **{len(demos) - 3}** more examples")
                lines.append("")

    # Reasoning strategy
    lines.append("## Reasoning Strategy")
    lines.append("")
    lines.append("Uses **Chain-of-Thought** reasoning with intermediate rationale steps.")
    lines.append("Each example includes explicit reasoning to guide the model's inference process.")
    lines.append("")

    return "\n".join(lines)


def format_dspy_prompt_from_module(optimized_module) -> str:
    """
    Fallback: Format DSPy module directly from object attributes.

    Args:
        optimized_module: The optimized module object

    Returns:
        Formatted prompt text
    """
    lines = []
    lines.append("# Optimized DSPy Prompt")
    lines.append("=" * 60)
    lines.append("")

    # Try to extract predictor information
    predictor = None
    if hasattr(optimized_module, 'predictor'):
        predictor = optimized_module.predictor
    elif hasattr(optimized_module, 'predict'):
        predictor = optimized_module.predict

    if predictor:
        # Signature
        if hasattr(predictor, 'signature'):
            sig = predictor.signature
            lines.append("## Signature")

            if hasattr(sig, 'input_fields'):
                lines.append(f"Input: {', '.join(sig.input_fields.keys())}")
            if hasattr(sig, 'output_fields'):
                lines.append(f"Output: {', '.join(sig.output_fields.keys())}")
            lines.append("")

            # Instructions
            if hasattr(sig, 'instructions') and sig.instructions:
                lines.append("## Instructions")
                lines.append("")
                lines.append("```")
                lines.append(sig.instructions)
                lines.append("```")
                lines.append("")

        # Few-shot demonstrations
        if hasattr(predictor, 'demos') and predictor.demos:
            lines.append("## Few-Shot Examples")
            lines.append(f"Number of examples: {len(predictor.demos)}")
            lines.append("")

            for i, demo in enumerate(predictor.demos[:3], 1):
                lines.append(f"### Example {i}")
                if hasattr(demo, 'inputs'):
                    lines.append(f"Input: {demo.inputs()}")
                if hasattr(demo, 'labels'):
                    lines.append(f"Output: {demo.labels()}")
                lines.append("")

            if len(predictor.demos) > 3:
                lines.append(f"... and {len(predictor.demos) - 3} more examples")
                lines.append("")

    # Reasoning strategy
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
        baseline_text = baseline_path.read_text(encoding="utf-8", errors="replace")
    else:
        baseline_text = "Baseline prompt not found"

    if optimized_path.exists():
        optimized_text = optimized_path.read_text(encoding="utf-8", errors="replace")
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
    dst.write_text(comparison, encoding="utf-8", errors="replace")
