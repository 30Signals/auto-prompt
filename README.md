# auto-prompt

Automatic prompt optimization for structured extraction.

This repository contains experiments testing whether automated prompt optimization 
can outperform handcrafted prompts on real extraction tasks. Our first experiment 
achieved a **+26.4 point accuracy improvement** on resume extraction — validated 
across 5 seeded runs with p < 0.0001.

> **Read the full case study:** [We Stopped Writing Prompts. Accuracy Jumped 26 Points.](SUBSTACK_LINK)

---

## Purpose

Prompts tend to evolve through trial and error, intuition, and copy-paste.
This repo exists to make that process explicit and testable.

`auto-prompt` is a place to:

- experiment with prompt assembly strategies
- test prompt transformations and optimizations
- measure impact across tasks and datasets
- develop intuition backed by evidence

---

## What this repo is

- an experimentation playground
- a prompt R&D lab
- a place to test hypotheses about prompt design
- a source of learnings that may later inform production systems

---

## Current Experiments

### Experiment 1: Automatic Prompt Optimization for Resume Extraction

**Goal:** Compare handcrafted prompts vs automated optimization for structured data extraction from unstructured text.

**Hypothesis:** Automated prompt optimization (DSPy) outperforms static handcrafted prompts on complex inference tasks.

**Experiment Structure:**

```
experiments/resume_extraction/
├── config.py                  # Configuration and LLM setup
├── modules.py                 # Baseline and DSPy modules
├── loader.py                  # Data loading (20 train, 30 test)
├── metrics.py                 # Validation metrics
├── evaluation.py              # Detailed evaluation
├── run.py                     # Main entry point
└── prompts/baseline.txt       # Handcrafted prompt
```

## Result table 

# Results (5-Run Aggregate):
| Metric | Handcrafted Baseline | Auto-Optimized | Delta |
|--------|---------------------|----------------|-------|
| Overall Accuracy (Mean) | 56.80% | 83.19% | **+26.39 pts** |
| 95% Confidence Interval | [54.01%, 59.58%] | [78.58%, 87.81%] | — |
| Job Role Accuracy | 18.67% | 81.33% | +62.67 pts |
| Skills F1 Score | 6.84% | 49.81% | +42.96 pts |
| p-value | — | — | 0.0000 |
| Cohen's d | — | — | 8.603 |

---

## Quick Start

### 1. Prerequisites

- Python 3.9+
- Azure OpenAI or Google Gemini API key

### 2. Setup

**Clone and create virtual environment:**

```bash
git clone https://github.com/30Signals/auto-prompt.git
cd auto-prompt

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Configure API keys:**

Create a `.env` file in the project root:

```bash
# Azure OpenAI (recommended)
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# OR Google Gemini (alternative)
GEMINI_API_KEY=your_gemini_key_here
```

### 3. Run Experiments

**Run resume extraction experiment:**

```bash
# New modular structure (recommended)
python -m experiments.resume_extraction.run

```

**Run all experiments:**

```bash
python scripts/run_all_experiments.py
```

**Interactive demo:**

```bash
python -m benchmark.live_demo
```

### 4. View Results

Results are saved in `experiments/resume_extraction/results/`:
- `baseline_results.json` - Static handcrafted prompt results
- `dspy_results.json` - DSPy optimized results
- `comparison_results.json` - Side-by-side comparison
- `optimized_module.json` - Saved DSPy module with learned examples

**Output includes:**
- Per-field accuracy (job_role, skills, education, experience)
- Overall accuracy comparison
- Improvement analysis
- Statistical metrics

---

## Methodology

Experiments in this repo follow a structured approach:

1. **Define hypothesis** - What are we testing?
2. **Create baseline** - Simple, handcrafted approach
3. **Implement optimization** - Automated or systematic improvement
4. **Measure impact** - Quantitative comparison
5. **Document learnings** - What worked, what didn't, and why

---

## Technical Details

**DSPy Optimization Pipeline:**
- Stage 1: BootstrapFewShot (generates demonstration examples from training data)
- Stage 2: COPRO (prompt instruction refinement)
- Validation: Enhanced semantic matching with role/skill synonyms

**Evaluation:**
- Per-field accuracy tracking (job_role, skills, education, experience)
- Error analysis (improvements and degradations)
- Output artifacts: `baseline_results.json`, `dspy_results.json`, `comparison_results.json`, `optimized_resume_module.json`

**Key Implementation Detail:** The baseline uses raw LLM calls to ensure no DSPy optimizations leak in, while the optimized version uses ChainOfThought with the same seed prompt enhanced by learned examples.

---

## Future Enhancements (TODO)

- [ ] **MIPROv2 Optimizer Ablation** - Compare MIPROv2 vs BootstrapFewShot+COPRO for optimization quality
- [ ] **Temperature Ablation** - Test impact of LLM temperature (0.0, 0.3, 0.7, 1.0) on optimization and inference
- [ ] **Cross-domain Validation** - Train on resumes, test on cover letters to measure generalization

---

## Contributing

This is an experimental repository. Contributions can include:
- New prompt optimization experiments
- Alternative datasets or tasks
- Different optimization strategies
- Analysis tools and visualizations

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.
