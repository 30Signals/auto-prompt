# Deep Dive: How the Resume Parsing Pipeline Works

This document explains the exact technical workflow of the experiment, from start to finish.

## 1. High-Level Architecture

The goal is to compare two approaches to extracting structured data (JSON) from unstructured resumes:
1.  **Manual Approach (Baseline)**: A human writes a prompt text file. We feed it to the LLM.
2.  **Automated Approach (DSPy)**: We give DSPy the *same* text file as a starting point (seed) + 20 training examples. It "programs" the LLM to be better.

## 2. The Execution Flow (`benchmark/run_experiment.py`)

The script `run_experiment.py` orchestrates the entire process in 4 main stages.

### Stage 1: Data Preparation (`loader.py`)
- **Input**: A CSV file (`Data/synthetic_100_resumes_realistic.csv`).
- **Splitting**: we strictly separate the data:
    - **Train Set (20 docs)**: Used *only* by DSPy to learn.
    - **Test Set (30 docs)**: Used to evaluate *both* models. The models have never seen these resumes.
- **Object**: Data is converted into `dspy.Example` objects containing the raw text and ground truth labels.

### Stage 2: The Baseline Run (Static & Untrained)
This represents the "Manual" approach.
- **NO TRAINING**: The Baseline is never trained or optimized. It is fixed.
- **Process**:
    1.  **Read File**: We read `benchmark/handcrafted_prompt.txt` exactly as written.
    2.  **Inference (Testing)**: We send this prompt + resume to the LLM to get a prediction.
    3.  **Result**: We measure how well this static prompt performs.
    *   *Note*: We use the LLM as an engine to process the text, but the prompt itself never changes.

### Stage 3: DSPy Optimization (Automated Learning)
This is where the "Training" happens, but only for the **New DSPy Module**, not the Baseline.
1.  **The Seed**: We use the *content* of `handcrafted_prompt.txt` as the starting instruction for DSPy.
2.  **Optimization Loop**:
    - DSPy takes this instruction and runs it on the **Training Set (20 docs)**.
    - It tries to solve them. If it gets a correct answer (verified by our logic), it saves that example.
    - **The Baseline is NOT updated**. A *new* prompt object is built.
3.  **Result**: A **Student Module** that has learned from the data. This module contains the original instruction PLUS new "few-shot" examples it discovered.

### Stage 4: Evaluation & Comparison (`evaluation.py`)
We run both the **Baseline** and the **Optimized DSPy Module** on the **Test Set**.

**Scoring Logic**:
- **Job Role / Education**: Strict case-insensitive match.
- **Skills**: Jaccard Similarity (How much overlap between predicted skills and actual skills).
- **Experience Years**: Numeric Tolerance.
    - *Constraint*: The value must be within `0.5` years of the truth.
    - *Baseline Failure*: The manual prompt often says "4 years" for "4.5 years", failing this check.
    - *DSPy Success*: DSPy sees examples of "4.5" in training and learns to be precise.

## Summary of Files

| File | Purpose |
| :--- | :--- |
| `run_experiment.py` | The main script. Runs the show. |
| `loader.py` | Loads CSV, splits 20/30. |
| `extractors.py` | Defines the `BaselineModule` (Manual) and `StudentModule` (DSPy). |
| `handcrafted_prompt.txt` | Your manual instructions. |
| `optimizer.py` | Runs the `BootstrapFewShot` learning loop. |
| `evaluation.py` | Calculates accuracy and prints the comparison table. |
