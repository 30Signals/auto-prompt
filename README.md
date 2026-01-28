# Resume Information Extraction: Manual vs Automated Optimization

**A pure product demo comparing Handcrafted Prompts vs DSPy Automation.**

## 🎯 The Business Story

Companies spend countless hours writing and maintaining prompts for extracting structured data from unstructured documents.
- **The Problem**: Humans write prompts. Data changes. Prompts break.
- **The Reality**: Maintaining prompts manually is brittle and does not scale.
- **The Solution**: **DSPy** automatically writes and optimizes prompts for you.

This experiment proves that **Zero Manual Prompt Engineering** > **Experts Handcrafting Prompts**.

## 📊 Results Summary

We ran a strict head-to-head comparison on an extremely challenging enterprise dataset of 100 synthetic resumes with **complex information extraction** requirements.

| Approach | Strategy | Overall Accuracy | Skills Extraction |
|----------|----------|------------------|-------------------|
| **Baseline** | Handcrafted Prompt (Static) | **56.58%** | **6.33%** (Poor inference) |
| **DSPy** | Automated Optimization | **75.87%** | **50.17%** (8x better!) |

**Complex Enterprise Challenges**:
- **Scattered information**: Timeline, education, and work activities randomly placed in messy text
- **Implicit skill inference**: Must infer skills from work descriptions ("optimized inference pipelines" → Python, TensorFlow)
- **Complex timeline parsing**: Calculate experience from scattered date ranges
- **Job role mapping**: Infer roles from activities across multiple domains
- **Enterprise complexity**: Realistic messy resume format with distractors

**DSPy's Key Advantage**: Starts with the same handcrafted prompt but automatically optimizes it with 16 few-shot examples, learning complex inference patterns that the static baseline cannot handle.

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
│   └── final_synthetic_100_resumes_enterprise.csv
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

### 4. Mode 2: Live Demo (User Mode)
Interactive mode to test real-world generalization. Paste *any* resume text to see how both models perform side-by-side.

```bash
python -m benchmark.live_demo
```

### 5. What You Will See (Detailed Comparison)
The tool will output a side-by-side comparison of the handcrafted prompt vs the DSPy-optimized version.

```text
DETAILED COMPARISON REPORT
==================================================

OVERALL ACCURACY:
Baseline:  56.58%
DSPy:      75.87%
Improvement: 19.29%

FIELD-WISE ACCURACY:
Field                Baseline     DSPy         Improvement 
--------------------------------------------------------
job_role             20.00%       53.33%       33.33%      
skills               6.33%        50.17%       43.83%      
education            100.00%      100.00%      0.00%       
experience_years     100.00%      100.00%      0.00%      
```

## 📂 Output Files
- `baseline_results.json`: Results from the handcrafted prompt (Static Baseline)
- `dspy_results.json`: Results from the optimized module (DSPy)
- `comparison_results.json`: Improvement metrics and detailed analysis
- `optimized_resume_module.json`: **DSPy's optimized prompt with 16 few-shot examples**

## 🔍 How DSPy Optimization Works

**Both models start with the same handcrafted prompt**, but:

1. **Baseline**: Uses the handcrafted prompt exactly as-is (static)
2. **DSPy**: Takes the same handcrafted prompt and automatically optimizes it by:
   - Learning from 20 training examples
   - Generating 16 few-shot examples through BootstrapFewShot
   - Adding reasoning chains for complex inference tasks
   - Creating an optimized prompt saved in `optimized_resume_module.json`

**View the Optimized Prompt**: Check `optimized_resume_module.json` to see how DSPy transformed the basic handcrafted prompt into a sophisticated few-shot prompt with examples and reasoning.

## 📋 Technology Stack
- **DSPy**: For declarative self-optimizing language models.
- **Pandas**: For data handling.
- **Python**: Core logic.

## 🤝 Contributing
This is a demonstration project. Feel free to fork and test with your own datasets by replacing the file in `Data/`.