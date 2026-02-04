# auto-prompt

**A sandbox for experimenting with automatic prompt generation and optimization.**

This repository is used to run structured experiments that test how different prompt-generation strategies affect LLM behavior. The focus is not on building a reusable library, but on learning what works, what breaks, and why.

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

### Experiment 1: DSPy-Based Resume Information Extraction

**Goal:** Compare handcrafted prompts vs automated optimization for structured data extraction from unstructured text.

**Hypothesis:** Automated prompt optimization (DSPy) outperforms static handcrafted prompts on complex inference tasks.

**Results:**

| Approach | Overall Accuracy | Skills Extraction |
|----------|------------------|-------------------|
| Handcrafted Prompt (Static) | 56.58% | 6.33% |
| DSPy Automated Optimization | 75.87% | 50.17% |

**Key Findings:**
- Automated optimization achieved 19.29% improvement in overall accuracy
- Skills extraction improved 8x through few-shot learning
- DSPy's BootstrapFewShot automatically generated 16 demonstration examples
- ChainOfThought reasoning improved implicit inference (role and skill detection)

**Experiment Structure:**

```
benchmark/
├── run_experiment.py          # Main entry point
├── handcrafted_prompt.txt     # Baseline prompt
├── extractors.py              # Baseline and DSPy modules
├── loader.py                  # Data loading (20 train, 30 test)
├── optimizer.py               # BootstrapFewShot + COPRO optimization
└── evaluation.py              # Metrics and comparison
```

**Setup:**

```bash
# Install dependencies
pip install dspy-ai pandas python-dotenv

# Configure LLM provider in .env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=...
# OR
GEMINI_API_KEY=...
```

**Run the experiment:**

```bash
# Full pipeline: baseline → optimization → comparison
python -m benchmark.run_experiment

# Interactive demo: test both models with custom input
python -m benchmark.live_demo
```

**Output:** Side-by-side comparison with per-field accuracy, improvements, and saved results in JSON files.

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

## Contributing

This is an experimental repository. Contributions can include:
- New prompt optimization experiments
- Alternative datasets or tasks
- Different optimization strategies
- Analysis tools and visualizations

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.