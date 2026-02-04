# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A sandbox for experimenting with automatic prompt generation and optimization. The repository runs structured experiments comparing different prompt strategies (handcrafted vs automated optimization) across various tasks.

**Current Status:** Phase 1 complete - code reorganized with publication tools. See `.claude/tasks.md` for detailed task tracking and roadmap.

## Repository Structure

```
auto-prompt/
├── experiments/                    # Individual experiments
│   └── resume_extraction/          # Experiment 1: Resume parsing
│       ├── config.py               # Experiment configuration
│       ├── modules.py              # DSPy modules (Baseline, Student)
│       ├── loader.py               # Data loading
│       ├── metrics.py              # Validation metrics
│       ├── evaluation.py           # Detailed evaluation
│       ├── run.py                  # Main experiment runner
│       ├── prompts/                # Prompt templates
│       ├── data/                   # Experiment data (optional)
│       └── results/                # Output JSON files
├── shared/                         # Reusable utilities
│   ├── llm_providers/              # LLM configuration (Azure, Gemini)
│   ├── evaluation/                 # Generic metrics
│   ├── optimization/               # DSPy optimizer wrappers
│   └── visualization/              # Plotting utilities
├── case_study/                     # Publication materials
│   ├── writeup.md                  # Case study draft
│   ├── analysis.ipynb              # Analysis notebook
│   └── figures/                    # Generated plots
├── scripts/                        # Utility scripts
│   └── run_all_experiments.py      # Run all experiments
├── benchmark/                      # Legacy code (deprecated)
└── Data/                           # Shared datasets
```

## Environment Setup

Configure LLM provider in `.env`:

```bash
# Azure OpenAI (primary)
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# OR Google Gemini (fallback)
GEMINI_API_KEY=your_key
```

## Core Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run specific experiment
python -m experiments.resume_extraction.run

# Run all experiments
python scripts/run_all_experiments.py

# Legacy commands (still work, but prefer above)
python -m benchmark.run_experiment
python -m benchmark.live_demo
```

## Adding New Experiments

1. Create directory: `experiments/your_experiment/`
2. Required files:
   - `config.py` - Paths, parameters
   - `modules.py` - DSPy signatures and modules
   - `loader.py` - Data loading
   - `metrics.py` - Validation metric for optimization
   - `evaluation.py` - Detailed evaluation
   - `run.py` - Main entry point
3. Add to `scripts/run_all_experiments.py`

## Shared Utilities

### LLM Providers (`shared/llm_providers`)
```python
from shared.llm_providers import setup_dspy_lm
setup_dspy_lm()  # Auto-configures DSPy with Azure or Gemini
```

### Optimization (`shared/optimization`)
```python
from shared.optimization import run_two_stage_optimization
optimized = run_two_stage_optimization(student, trainset, metric)
```

### Evaluation (`shared/evaluation`)
```python
from shared.evaluation import compare_results, EvaluationResult
comparison = compare_results(baseline_results, optimized_results)
```

### Visualization (`shared/visualization`)
```python
from shared.visualization import plot_field_comparison, save_figure
fig = plot_field_comparison(field_data)
save_figure(fig, "output/plot")  # Saves PNG and PDF
```

## DSPy Patterns

**Baseline module** (raw LLM call, no optimization):
```python
class BaselineModule(dspy.Module):
    def forward(self, input_text):
        response = dspy.settings.lm(prompt)
        return dspy.Prediction(...)
```

**Student module** (optimizable with ChainOfThought):
```python
class StudentModule(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(MySignature)
    def forward(self, input_text):
        return self.predictor(input_text=input_text)
```

**Validation metric** (for optimization):
```python
def validate(example, pred, trace=None) -> float:
    # Return score between 0 and 1
    return score
```

## Output Files

Each experiment saves to `experiments/{name}/results/`:
- `baseline_results.json` - Baseline predictions and scores
- `dspy_results.json` - Optimized model results
- `comparison_results.json` - Improvements/degradations analysis
- `optimized_module.json` - Saved DSPy module with learned demos

## Data Format

Resume extraction expects CSV with columns:
- `resume_id`, `unstructured_text`, `job_role`, `skills`, `education`, `experience_years`

Other experiments define their own formats in their `loader.py`.

## Case Study

The `case_study/` directory contains publication materials:
- `writeup.md` - Case study template with methodology and results sections
- `analysis.ipynb` - Jupyter notebook for generating figures and statistics
- `figures/` - Generated visualizations (PNG + PDF)

## Task Tracking

The project uses `.claude/tasks.md` to track progress across phases:

**Phase 1: Foundation** ✅ Complete (PR #3)
- Code reorganization
- Visualization tools
- Publication templates

**Phase 2: Statistical Rigor** 🔄 Next
- Multiple runs with seeds
- Confidence intervals
- Ablation studies

**Phase 3: New Experiments** 📋 Planned
- Medical entity extraction
- Product categorization
- Legal contract analysis
- Cross-domain validation

**Phase 4: Publication** 📝 Planned
- Complete writeup
- Generate all figures
- Statistical analysis
- Reproducibility documentation

Refer to `.claude/tasks.md` for detailed checklists and current status.

## Development Workflow

1. **Check tasks:** Review `.claude/tasks.md` for current priorities
2. **Create branch:** Work on feature branches for PRs
3. **Run experiments:** Use `python -m experiments.<name>.run`
4. **Generate figures:** Use visualization utilities in `shared/`
5. **Update tasks:** Mark items complete in `.claude/tasks.md`
6. **Commit:** Include task references in commit messages
