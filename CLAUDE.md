# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DSPy-based resume information extraction optimization project that demonstrates automated prompt engineering vs handcrafted prompts. The system extracts structured data (job role, skills, education, experience) from unstructured resume text and compares baseline (static handcrafted prompt) vs DSPy-optimized (few-shot learning with reasoning chains) approaches.

## Environment Setup

The project supports two LLM providers configured via `.env`:

**Azure OpenAI** (primary):
```bash
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview  # optional, has default
```

**Google Gemini** (fallback):
```bash
GEMINI_API_KEY=your_key
```

The system auto-selects Azure if available, otherwise falls back to Gemini (gemini-1.5-flash).

## Core Commands

**Install dependencies:**
```bash
pip install dspy-ai pandas python-dotenv
```

**Run full experiment** (baseline + optimization + comparison):
```bash
python -m benchmark.run_experiment
```

**Run interactive demo** (test both models with custom resume text):
```bash
python -m benchmark.live_demo
```

**Test data loading** (verify CSV format):
```bash
python -m benchmark.loader
```

## Architecture

### Pipeline Flow

1. **Data Loading** (`loader.py`): Loads CSV, splits into 20 train / 30 test examples
2. **Baseline Evaluation** (`extractors.py::BaselineModule`): Runs handcrafted prompt directly via `dspy.settings.lm(prompt)` with manual JSON parsing
3. **Optimization** (`optimizer.py`): Two-stage optimization:
   - Stage 1: BootstrapFewShot (20 demos, 6 rounds) generates few-shot examples
   - Stage 2: COPRO refines prompts (breadth=10, depth=3)
4. **DSPy Evaluation** (`extractors.py::StudentModule`): Runs optimized ChainOfThought predictor with reasoning
5. **Comparison** (`evaluation.py`): Field-wise accuracy, error analysis, improvement metrics

### Key Modules

**`config.py`**: Centralized configuration
- Data paths (default: `Data/Resume_long1.csv`)
- Train/test split sizes (20/30)
- LLM provider selection via `get_llm_config()`

**`extractors.py`**: Core extraction modules
- `BaselineModule`: Direct LLM call with handcrafted prompt from `handcrafted_prompt.txt`
- `StudentModule`: DSPy ChainOfThought with `EnhancedResumeSignature` (includes reasoning field)
- Both use the same base prompt; DSPy adds few-shot examples during optimization

**`optimizer.py`**: DSPy optimization logic
- `enhanced_validate_resume_output()`: Sophisticated validation with:
  - Role synonym matching (e.g., "data scientist" ≈ "data analyst")
  - Skill category matching (e.g., "python" ≈ "programming")
  - Semantic similarity scoring
- Two-stage optimization: BootstrapFewShot → COPRO

**`evaluation.py`**: Metrics and comparison
- `detailed_evaluation()`: Per-field and overall accuracy
- `compare_models()`: Identifies improvements/degradations per sample
- Saves to `baseline_results.json`, `dspy_results.json`, `comparison_results.json`

**`handcrafted_prompt.txt`**: Static baseline prompt
- Instructs LLM to infer job roles from work activities
- Infer skills from descriptions
- Calculate experience from scattered dates
- Output strict JSON only

### Data Format

CSV must contain columns: `resume_id`, `unstructured_text`, `job_role`, `skills`, `education`, `experience_years`

Examples are converted to `dspy.Example` objects with `unstructured_text` as input and the rest as labels.

## Output Files

- `baseline_results.json`: Static prompt results with per-sample predictions
- `dspy_results.json`: Optimized model results
- `comparison_results.json`: Side-by-side comparison with improvements/degradations
- `optimized_resume_module.json`: **The optimized DSPy prompt with learned few-shot examples** (inspect this to see how DSPy transformed the base prompt)

## Common Patterns

**Adding a new field to extract:**
1. Update CSV columns and ground truth labels
2. Add field to `ResumeSignature` and `EnhancedResumeSignature` in `extractors.py`
3. Update validation logic in `optimizer.py::enhanced_validate_resume_output()`
4. Update evaluation scoring in `evaluation.py::detailed_evaluation()`

**Switching datasets:**
1. Update `DATA_PATH` in `config.py` or pass new path
2. Ensure CSV has required columns
3. Adjust `TRAIN_SIZE` and `TEST_SIZE` if needed

**Tuning optimization:**
- Modify BootstrapFewShot parameters in `optimizer.py` (demos, rounds, errors)
- Adjust COPRO parameters (breadth, depth, temperature)
- Customize validation metric weights in `enhanced_validate_resume_output()`

## DSPy-Specific Notes

- The baseline uses raw LLM calls (`dspy.settings.lm(prompt)`) to ensure no DSPy optimizations leak in
- DSPy optimization happens via `BootstrapFewShot` which learns from training examples
- The optimized module is saved with `.save()` and contains the full prompt + demos
- `ChainOfThought` adds reasoning before extraction, improving complex inference tasks
- The same handcrafted prompt seeds both approaches; DSPy enhances it automatically
