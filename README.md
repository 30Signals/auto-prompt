# Resume Information Extraction: Manual vs Automated Optimization

**A pure product demo comparing Handcrafted Prompts vs DSPy Automation.**

## 🎯 The Business Story

Companies spend countless hours writing and maintaining prompts for extracting structured data from unstructured documents.
- **The Problem**: Humans write prompts. Data changes. Prompts break.
- **The Reality**: Maintaining prompts manually is brittle and does not scale.
- **The Solution**: **DSPy** automatically writes and optimizes prompts for you.

This experiment proves that **Zero Manual Prompt Engineering** > **Experts Handcrafting Prompts**.

## 📊 Results Summary

We ran a strict head-to-head comparison on a dataset of 100 synthetic resumes.

| Approach | Strategy | Overall Accuracy | Experience Precision |
|----------|----------|------------------|----------------------|
| **Baseline** | Handcrafted Prompt (Static) | **82.5%** | **30.0%** (Fails on specific formats) |
| **DSPy** | Automated Optimization | **100.0%** | **100.0%** (Learns from data) |

**Key Insight**: The handcrafted prompt failed to capture precise experience years (e.g., `4.42` vs `4`), a common edge case. DSPy automatically learned this requirement just by passing the training data, without any code changes.

## 🏗️ Project Structure

The project is simplified to focus on the core comparison logic.

```plain
optimizing-ie-pipelines/
├── benchmark/
│   ├── run_experiment.py     # Main entry point
│   ├── handcrafted_prompt.txt # The actual human-written prompt file
│   ├── extractors.py         # Defines both Baseline and DSPy modules
│   ├── loader.py             # Loads and splits data (20 train, 30 test)
│   ├── optimizer.py          # DSPy optimization logic
│   └── evaluation.py         # Metrics and comparison logic
├── Data/
│   └── synthetic_100_resumes_realistic.csv
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- Azure OpenAI or Google Gemini API Key.

### 2. Setup
Clone the repo and install dependencies:
```bash
pip install dspy-ai pandas python-dotenv
```

Configure your keys in `.env`:
```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=...
# OR
GEMINI_API_KEY=...
```

### 3. Run the Experiment
Execute the full pipeline. This will run the baseline, perform optimization, and show the comparison.

```bash
python -m benchmark.run_experiment
```

### 4. What You Will See (Detailed Comparison)
The tool will output a side-by-side comparison of the handcrafted prompt vs the DSPy-optimized version.

```text
DETAILED COMPARISON REPORT
==================================================

OVERALL ACCURACY:
Baseline:  82.50%
Optimized: 100.00%
Improvement: 17.50%

FIELD-WISE ACCURACY:
Field                Baseline     Optimized    Improvement 
--------------------------------------------------------
job_role             100.00%      100.00%      0.00%       
skills               100.00%      100.00%      0.00%       
education            100.00%      100.00%      0.00%       
experience_years     30.00%       100.00%      70.00%      
```

## 📂 Output Files
- `baseline_results.json`: Results from the handcrafted prompt (Static Baseline)
- `dspy_results.json`: Results from the optimized module (DSPy)
- `comparison_results.json`: Improvement metrics and detailed analysis

## 📋 Technology Stack
- **DSPy**: For declarative self-optimizing language models.
- **Pandas**: For data handling.
- **Python**: Core logic.

## 🤝 Contributing
This is a demonstration project. Feel free to fork and test with your own datasets by replacing the file in `Data/`.