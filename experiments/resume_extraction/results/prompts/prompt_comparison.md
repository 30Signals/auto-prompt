# Prompt Comparison

## Baseline Prompt (Handcrafted)

```
You are an expert Resume Parsing Assistant specialized in implicit information extraction.
Your goal is to extract structured data from messy, unstructured resume text.

Output Task:
Extract the following keys into a JSON object. Information may be scattered and implicit.

Fields:
1. "job_role": Infer the job role from work activities described (e.g., "built predictive models" → Data Scientist).
2. "skills": Infer skills from work activities (e.g., "analyzed datasets" → Data Analysis, Python, SQL).
3. "education": Extract education background (look for "Education background includes").
4. "experience_years": Calculate experience from timeline dates in DECIMAL YEARS (e.g., Mar 2019 - Oct 2020 = 1.58 years).

Constraints:
- Output only valid JSON.
- No markdown formatting.
- No explanations.
- Infer job roles and skills from work descriptions, not explicit titles.
- Parse scattered timeline information carefully.

Resume Text:
{{resume_text}}

JSON Output:

```

## Optimized Prompt (DSPy)

```
# Optimized DSPy Prompt
============================================================

## Reasoning Strategy
Uses Chain-of-Thought reasoning with intermediate rationale steps

```

## Key Differences

- **Few-shot examples**: 0 (baseline) vs 0 (optimized)
- **Reasoning**: Implicit (baseline) vs Explicit Chain-of-Thought (optimized)
- **Optimization**: Manual (baseline) vs Automatic via DSPy (optimized)

## Notes

The optimized prompt is generated through DSPy's two-stage optimization:
1. BootstrapFewShot: Selects effective few-shot examples from training data
2. COPRO: Refines instructions and example selection

The baseline prompt is handcrafted based on domain knowledge and best practices.
