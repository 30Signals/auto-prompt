# Case Study: Automated Prompt Optimization with DSPy

**Status:** Draft

## Executive Summary

This case study demonstrates how automated prompt optimization can significantly outperform handcrafted prompts for complex information extraction tasks. Using DSPy's BootstrapFewShot and COPRO optimizers, we achieved [X]% improvement in overall accuracy compared to a carefully designed static prompt.

**Key Findings:**
- Automated optimization achieved [X]% accuracy vs [Y]% for handcrafted prompts
- Skills extraction improved [X]x through few-shot learning
- The same base prompt was enhanced automatically with learned demonstrations
- Complex inference tasks (role detection, skill inference) benefited most

---

## Problem Statement

### The Challenge

Extracting structured information from unstructured text is a common enterprise task:
- Resume parsing for HR systems
- Contract clause extraction for legal review
- Medical record structuring for healthcare analytics

Traditional approaches require:
1. Expert-crafted prompts (time-consuming)
2. Iterative refinement (error-prone)
3. Domain-specific tuning (doesn't scale)

### Research Question

**Can automated prompt optimization outperform carefully handcrafted prompts for information extraction?**

---

## Methodology

### Experimental Setup

**Task:** Extract structured fields from unstructured resume text
- Job role (inferred from work activities)
- Skills (inferred from descriptions)
- Education level
- Years of experience (calculated from dates)

**Dataset:** [N] synthetic resumes with ground truth labels
- Train: 20 samples (for optimization)
- Test: 30 samples (for evaluation)

**Baseline:** Handcrafted prompt designed by domain expert
- Clear instructions for each field
- Explicit inference guidelines
- JSON output constraints

**Optimized:** DSPy two-stage optimization
- Stage 1: BootstrapFewShot (20 demos, 6 rounds)
- Stage 2: COPRO (breadth=10, depth=3)

### Evaluation Metrics

- **Overall Accuracy:** Average of field-level scores
- **Field-wise Accuracy:** Per-field exact/semantic match
- **Improvement Analysis:** Per-sample comparison

---

## Results

### Overall Performance

| Approach | Overall Accuracy | Skills | Job Role | Education | Experience |
|----------|-----------------|--------|----------|-----------|------------|
| Baseline (Handcrafted) | [X]% | [X]% | [X]% | [X]% | [X]% |
| DSPy (Optimized) | [X]% | [X]% | [X]% | [X]% | [X]% |
| **Improvement** | [+X]% | [+X]% | [+X]% | [+X]% | [+X]% |

### Key Observations

1. **Skills extraction improved dramatically**
   - Baseline struggled with implicit skill inference
   - DSPy learned patterns from demonstrations

2. **Job role detection benefited from reasoning**
   - ChainOfThought added explicit reasoning step
   - Better mapping from activities to roles

3. **Structured fields were already solved**
   - Education and experience parsing worked well in baseline
   - Less room for improvement

---

## Analysis

### What DSPy Optimized

The optimized prompt includes:
1. **Original instructions** (preserved from handcrafted)
2. **[N] learned demonstrations** (automatically selected)
3. **Reasoning template** (added by ChainOfThought)

### Why Optimization Worked

1. **Demonstration selection:** DSPy identified the most informative examples
2. **Pattern learning:** Few-shot examples taught implicit inference rules
3. **Reasoning chains:** Explicit reasoning improved complex tasks

### Limitations

- Optimization requires labeled training data
- API costs scale with optimization rounds
- Results may vary with different random seeds

---

## Conclusions

### For Practitioners

1. **Start with a reasonable base prompt** - optimization enhances, not replaces
2. **Invest in quality training data** - DSPy learns from examples
3. **Use ChainOfThought for inference tasks** - explicit reasoning helps

### For Researchers

1. **Automated optimization is competitive** - even with small datasets (20 examples)
2. **Different fields benefit differently** - complex inference gains most
3. **Two-stage optimization adds value** - BootstrapFewShot + COPRO > either alone

---

## Reproducibility

### Code

```bash
# Clone repository
git clone https://github.com/[org]/auto-prompt.git
cd auto-prompt

# Install dependencies
pip install -r requirements.txt

# Configure LLM provider
cp .env.example .env
# Edit .env with your API keys

# Run experiment
python -m experiments.resume_extraction.run
```

### Data

- Dataset: `Data/Resume_long1.csv`
- Format: CSV with columns `resume_id`, `unstructured_text`, `job_role`, `skills`, `education`, `experience_years`

### Environment

- Python 3.9+
- DSPy 2.4+
- LLM: Azure OpenAI GPT-4 / Google Gemini

---

## Appendix

### A. Handcrafted Prompt

```
[Include full baseline prompt]
```

### B. Optimized Prompt Structure

```
[Include example of optimized prompt with demonstrations]
```

### C. Sample Predictions

| Sample | Ground Truth | Baseline | DSPy |
|--------|-------------|----------|------|
| 1 | ... | ... | ... |

---

*Last updated: [Date]*
