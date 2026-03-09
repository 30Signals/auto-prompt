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

## Reasoning Strategy

Uses **Chain-of-Thought** reasoning with intermediate rationale steps.
Each example includes explicit reasoning to guide the model's inference process.

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
