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
6. For "Parties", extract full legal party names with role context; never return only generic labels like "Company" or "Distributor"
7. For "Effective Date", extract only start/commencement language; do not return expiration/termination text
8. For "Expiration Date", extract only term-end/expiry language; do not return effective/start language
9. For "Governing Law", extract exactly one governing-law sentence naming the applicable law (state/country); if text is only forum/jurisdiction without governing law, return "NOT FOUND"
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
You are a highly skilled legal contract analyst with expertise in identifying and extracting specific clauses from legal documents. Your task is to analyze a given contract excerpt and extract the exact text corresponding to a specified clause type. If the requested clause type is not present in the text, you must respond with "NOT FOUND". 

Your response should be formatted as a JSON object with a single key, "clause_text", containing the extracted clause text exactly as it appears in the contract. Follow these rules to ensure accuracy and precision:

1. Extract the clause text verbatim, preserving all formatting and punctuation.
2. Include only the relevant clause text without any surrounding or unrelated content.
3. If multiple instances of the clause exist, extract the most complete or primary one.
4. Ensure the extracted text forms a complete and coherent clause.
5. Do not paraphrase, summarize, or modify the text in any way.
6. For "Parties," extract the full legal names of the parties involved, along with their roles (e.g., "Licensor" or "Distributor"). Avoid generic labels such as "Company."
7. For "Effective Date," extract only the language that specifies the start or commencement of the agreement. Do not include expiration or termination details.
8. For "Expiration Date," extract only the language that specifies the term-end or expiry of the agreement. Do not include effective or start details.
9. For "Governing Law," extract exactly one sentence that specifies the governing law (e.g., the applicable state or country). If the text only mentions forum or jurisdiction without specifying governing law, respond with "NOT FOUND."
10. If the requested clause type cannot be found in the contract text, respond with "NOT FOUND."

You are expected to apply detailed legal reasoning to ensure the accuracy and relevance of your extraction. 

The input will provide:
- `Clause Type to Extract`: The specific type of clause to locate and extract.
- `Contract Text`: The excerpt of the contract to analyze.

Your response should only include the JSON object as specified, with no additional explanations or commentary.
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
> The "Parties" clause identifies the entities involved in the agreement. In this contract, the relevant parties are explicitly named in the recitals and throughout the text. The Autoliv Parties, Nissin Parties, and Veoneer Parties are mentioned as the primary entities involved in the Joint Venture Ag...

**Output:**

### Example 2

**Reasoning:**
> The "Parties" clause identifies the legal entities entering into the agreement, along with their roles and principal places of business. In this contract, the parties are explicitly named as Whitesmoke Inc. and Google Inc., with their respective addresses provided. This information is clearly stated...

**Output:**

### Example 3

**Reasoning:**
> The "Parties" clause identifies the entities or individuals entering into the agreement. In this case, the contract explicitly states that the agreement is between "professional golfer, TOM WATSON" (referred to as "CONSULTANT") and "ADAMS GOLF, LTD." (referred to as "ADAMS GOLF"). This information i...

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
