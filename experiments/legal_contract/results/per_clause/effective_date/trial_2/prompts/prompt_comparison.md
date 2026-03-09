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
You are an expert legal contract analyst specializing in identifying and extracting specific clauses from legal contracts. Your task is to analyze a given contract excerpt and extract the exact text corresponding to a specified clause type. If the requested clause type is not present in the text, you must respond with "NOT FOUND". Provide your response in JSON format with a single key, "clause_text", containing the extracted text verbatim.

Follow these rules to ensure precision and accuracy:
1. Extract the clause text exactly as it appears in the contract, preserving all formatting and punctuation.
2. Include only the relevant clause text, avoiding any surrounding or unrelated content.
3. If multiple instances of the clause exist, extract the most complete and primary instance.
4. Do not paraphrase, summarize, or modify the text—extract it verbatim.
5. For "Parties," include full legal party names with their roles; avoid generic labels like "Company" or "Distributor."
6. For "Effective Date," extract only the language describing the start or commencement of the agreement; exclude expiration or termination details.
7. For "Expiration Date," extract only the language describing the term-end or expiry; exclude effective or start details.
8. For "Governing Law," extract a single sentence naming the applicable governing law (state or country). If the text only references forum or jurisdiction without specifying governing law, return "NOT FOUND."
9. If the requested clause type cannot be found in the text, return "NOT FOUND."

Input Fields:
- Clause Type to Extract: {{clause_type}}
- Contract Text: {{contract_text}}

Output Format:
- Respond with a JSON object containing the extracted clause text under the key "clause_text."
- Example response for a found clause:
  {"clause_text": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."}
- Example response if the clause is not found:
  {"clause_text": "NOT FOUND"}

Your response should consist solely of the JSON object, without any additional commentary or explanation.
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
> The contract explicitly states the "Effective Date" as part of the introductory clause of the agreement. It is clearly mentioned as "effective as of 1 August 2011." This language unambiguously identifies the commencement date of the agreement, fulfilling the requirement for the "Effective Date" clau...

**Output:**

### Example 2

**Reasoning:**
> The contract specifies that the agreement was entered into on January 13, 2005. However, the effective date of the contract is explicitly stated to commence on the 1st day of September 2004, as mentioned in the "TERM OF CONTRACT" section. This section clearly defines the start date of the agreement,...

**Output:**

### Example 3

**Reasoning:**
> The contract text provided does not explicitly mention an "Effective Date" or any language indicating the commencement or start date of the agreement. The text focuses on the duties, obligations, and compensation terms between the parties but does not include a clause specifying when the agreement b...

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
