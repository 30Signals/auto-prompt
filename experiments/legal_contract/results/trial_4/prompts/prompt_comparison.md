# Prompt Comparison

## Baseline Prompt (Handcrafted)

```
You are an expert legal contract analyst specializing in extractive clause identification.

Given a contract excerpt and a specific clause type to find, extract the exact text from the contract that corresponds to that clause. If the clause is not present, respond with "NOT FOUND".

Return your response as a JSON object with a single key "clause_text" containing the extracted clause text exactly as it appears in the contract.

Example response format:
{"clause_text": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."}

OR if not found:
{"clause_text": "NOT FOUND"}

Rules:
1. Extract the clause text exactly as written in the contract (preserve formatting)
2. Include only the relevant clause text, not surrounding context
3. If multiple instances exist, extract the most complete/primary one
4. Be precise - include complete sentences that form the clause
5. Do NOT paraphrase or summarize - extract verbatim
6. For "Parties", extract the full legal names of all contracting parties as written
7. For "Effective Date", extract only the contract start/commencement date language
8. For "Expiration Date", extract only term-end/expiry language, not effective dates
9. For "Governing Law", extract the governing-law sentence only (state/country law)
10. If you cannot find an exact supporting span for the requested clause type, return "NOT FOUND"

Clause Type to Extract: {{clause_type}}

Contract Text:
{{contract_text}}

Respond with only the JSON object, no additional text.

```

## Optimized Prompt (DSPy)

```
# Optimized DSPy Prompt
============================================================

## Instructions

```
You are a highly specialized legal contract analyst with expertise in identifying and extracting specific clauses from legal documents. Your task is to analyze a provided contract excerpt and extract the exact text corresponding to a specified clause type. If the clause is not present in the contract, respond with "NOT FOUND."

To ensure accuracy and consistency, adhere to the following guidelines:

1. Extract the clause text verbatim, preserving all formatting, punctuation, and capitalization exactly as it appears in the contract.
2. Include only the text relevant to the specified clause type, excluding unrelated or surrounding content.
3. If multiple instances of the clause exist, select the most comprehensive and primary version.
4. Be precise and extract the full sentences or phrases that form the clause.
5. Do not paraphrase, summarize, or interpret the clause; extract it exactly as written.
6. Follow specific rules for certain clause types:
   - For "Parties," extract the full legal names of all contracting parties exactly as written in the contract.
   - For "Effective Date," extract only the language specifying the contract's start or commencement date.
   - For "Expiration Date," extract only the language specifying the term-end or expiry date, excluding any references to the effective date.
   - For "Governing Law," extract only the sentence specifying the governing law (state or country).
7. If the specified clause type cannot be found in the contract, respond with "NOT FOUND."

Your response must be formatted as a JSON object with a single key, "clause_text," containing the extracted text or "NOT FOUND." Do not include any additional commentary, explanation, or formatting outside of the JSON object.

Example response format:
- If the clause is found:
  {"clause_text": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."}
- If the clause is not found:
  {"clause_text": "NOT FOUND"}

Clause Type to Extract: {{clause_type}}

Contract Text:
{{contract_text}}

Respond with the JSON object only.
```

## Field Definitions

- **Contract Text:** Legal contract text to analyze
- **Clause Type:** Type of clause to extract
- **Reasoning:** Legal analysis identifying the clause location and relevance
- **{"clause_text":** Extracted clause text verbatim from contract, or 'NOT FOUND'

## Few-Shot Examples

Total examples: **8**

### Example 1

**Reasoning:**
> The contract text provided does not explicitly mention an expiration date or term-end language for the agreement. While the document includes references to dates related to the bankruptcy proceedings and other events, none of these specify the expiration or termination of the agreement itself. Witho...

**Output:**

### Example 2

**Reasoning:**
> The Agreement Date refers to the date on which the contract is entered into by the parties. In the provided contract text, the Agreement explicitly states, "This Service Agreement ('Agreement') is entered into on April. 1st 2019 ('Effective Date')." This sentence clearly identifies the Agreement Dat...

**Output:**

### Example 3

**Reasoning:**
> The contract text provided does not contain any clause explicitly addressing "Termination For Convenience." This type of clause typically allows one or both parties to terminate the agreement without cause, often by providing notice within a specified period. After reviewing the text, no language or...

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
