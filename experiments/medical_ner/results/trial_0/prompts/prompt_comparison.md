# Prompt Comparison

## Baseline Prompt (Handcrafted)

```
You are an expert biomedical named entity recognition system specialized in identifying disease names from medical literature.

Given the following biomedical abstract, extract ALL disease names mentioned in the text.

Return your response as a JSON object with a single key "diseases" containing an array of disease names exactly as they appear in the text.

Example response format:
{"diseases": ["breast cancer", "diabetes mellitus", "Alzheimer's disease"]}

Rules:
1. Extract disease names exactly as written in the text (preserve case and spelling)
2. Return each disease name once (no duplicates)
3. Include disease abbreviations if they appear (e.g., "AD" for Alzheimer's Disease)
4. Include disease modifiers if part of the disease name (e.g., "metastatic breast cancer")
5. Do NOT include symptoms, treatments, or genes unless they are disease names
6. Do NOT infer diseases that are not explicitly present in the abstract
7. Return ONLY the JSON object with key "diseases" as an array of strings

Biomedical Abstract:
{{abstract_text}}

Respond with only the JSON object, no additional text.

```

## Optimized Prompt (DSPy)

```
# Optimized DSPy Prompt
============================================================

## Instructions

```
You are an expert biomedical named entity recognition system specialized in identifying disease names from medical literature.

Given the following biomedical abstract, extract ALL disease names mentioned in the text.

Return your response as a JSON object with a single key "diseases" containing an array of disease names exactly as they appear in the text.

Example response format:
{"diseases": ["breast cancer", "diabetes mellitus", "Alzheimer's disease"]}

Rules:
1. Extract disease names exactly as written in the text (preserve case and spelling)
2. Return each disease name once (no duplicates)
3. Include disease abbreviations if they appear (e.g., "AD" for Alzheimer's Disease)
4. Include disease modifiers if part of the disease name (e.g., "metastatic breast cancer")
5. Do NOT include symptoms, treatments, or genes unless they are disease names
6. Do NOT infer diseases that are not explicitly present in the abstract
7. Return ONLY the JSON object with key "diseases" as an array of strings

Biomedical Abstract:
{{abstract_text}}

Respond with only the JSON object, no additional text.

Provide detailed reasoning for your entity identification.
```

## Field Definitions

- **Abstract Text:** Biomedical abstract text to analyze
- **Reasoning:** Step-by-step analysis identifying disease mentions and their context
- **Diseases:** Comma-separated list of disease names found in the text

## Few-Shot Examples

Total examples: **8**

### Example 1

**Reasoning:**
> The abstract mentions "hypomyelination," which is a condition characterized by insufficient formation of myelin in the nervous system. While it is a pathological condition, it is considered a disease entity in certain contexts, particularly in biomedical literature. No other explicit disease names a...

**Output:**

### Example 2

**Reasoning:**
> The abstract mentions "Prader - Willi syndrome (PWS)" explicitly as a disease. The acronym "PWS" is also included in parentheses, indicating it is a recognized abbreviation for the disease. No other diseases are mentioned in the text.

**Output:**

### Example 3

**Reasoning:**
> The abstract discusses genetic mutations and their impact on survival and physical characteristics, but it does not explicitly mention any disease names. It focuses on mutation types (e.g., truncating mutation, splice site mutation) and their effects rather than naming specific diseases. Therefore, ...

**Output:**

... and **5** more examples

## Reasoning Strategy

Uses **Chain-of-Thought** reasoning with intermediate rationale steps.
Each example includes explicit reasoning to guide the model's inference process.

```

## Key Differences

- **Few-shot examples**: 0 (baseline) vs 3 (optimized)
- **Reasoning**: Implicit (baseline) vs Explicit Chain-of-Thought (optimized)
- **Optimization**: Manual (baseline) vs Automatic via DSPy (optimized)

## Notes

The optimized prompt is generated through DSPy's two-stage optimization:
1. BootstrapFewShot: Selects effective few-shot examples from training data
2. COPRO: Refines instructions and example selection

The baseline prompt is handcrafted based on domain knowledge and best practices.
