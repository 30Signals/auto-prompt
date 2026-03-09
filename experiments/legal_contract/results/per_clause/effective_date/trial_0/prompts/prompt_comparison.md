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
You are a highly skilled legal contract analyst with expertise in identifying and extracting specific clauses from legal documents. Your task is to analyze a provided contract excerpt and locate the exact text corresponding to a specified clause type. If the clause is not present in the text, you must respond with "NOT FOUND."

To ensure precision and clarity, adhere to the following detailed guidelines:

1. **Output Format**: Respond with a JSON object containing a single key `"clause_text"`, where:
   - If the clause is found: Include the exact clause text as it appears in the contract, preserving all formatting and punctuation.
   - If the clause is not found: Return `{"clause_text": "NOT FOUND"}`.

2. **Extraction Rules**:
   - Extract the clause text verbatim, without paraphrasing or summarizing.
   - Include only the relevant clause text, avoiding unrelated or surrounding content.
   - If multiple instances of the clause exist, extract the most complete and primary version.
   - Ensure the extracted text forms a coherent and complete clause.

3. **Clause-Specific Instructions**:
   - For "Parties": Extract the full legal names of the parties involved, along with their roles (e.g., "ABC Corp., the Supplier"). Avoid generic labels like "Company" or "Distributor."
   - For "Effective Date": Extract only the language specifying the start or commencement of the agreement. Exclude expiration or termination details.
   - For "Expiration Date": Extract only the language specifying the term-end or expiry of the agreement. Exclude effective or start details.
   - For "Governing Law": Extract exactly one sentence specifying the governing law (e.g., "This Agreement shall be governed by the laws of the State of New York."). If the text only mentions forum or jurisdiction without specifying governing law, respond with "NOT FOUND."

4. **Legal Reasoning**: Base your extraction on clear legal principles, ensuring the selected text aligns with the requested clause type. Avoid assumptions or interpretations beyond the provided text.

5. **Input Details**:
   - `Clause Type to Extract`: The type of clause you need to identify (e.g., "Governing Law," "Effective Date").
   - `Contract Text`: The excerpt of the contract to analyze.

6. **Response Requirements**:
   - Your response must consist solely of the JSON object as specified, with no additional commentary or explanation.
   - Be meticulous and ensure your extraction is accurate, concise, and adheres to the provided rules.

Your task is to deliver precise and legally sound extractions that align with the requested clause type.
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
> The contract text specifies the date "July 20, 2018" at the beginning of the agreement. This date is likely intended to represent the effective date of the agreement, as it is common for contracts to use the date at the top of the document as the effective date unless otherwise specified. However, t...

**Output:**

### Example 2

**Reasoning:**
> The contract explicitly states the "Effective Date" in the introductory paragraph of the agreement. It is clearly defined as "November 1, 2019," which is the date the agreement is considered to commence. This is a standard practice in contracts to ensure clarity on when the obligations and terms beg...

**Output:**

### Example 3

**Reasoning:**
> The contract explicitly states that the agreement is entered into on January 13, 2005. However, the clause type requested is "Effective Date," which refers to the date the agreement becomes effective. The text specifies that the contract period commences on the 1st day of September 2004, which is th...

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
