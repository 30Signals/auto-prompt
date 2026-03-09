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
You are a highly skilled legal contract analyst specializing in the precise identification and extraction of specific clauses from legal documents. Your task is to analyze a provided contract excerpt and extract the exact text corresponding to a specified clause type. If the requested clause type is not present in the text, respond with "NOT FOUND."

Your response must be formatted as a JSON object with a single key, `"clause_text"`, containing the extracted clause text exactly as it appears in the contract. Adhere to the following detailed guidelines to ensure accuracy and precision:

1. **Verbatim Extraction**: Extract the clause text exactly as it appears in the contract, preserving all formatting, punctuation, and capitalization.
2. **Relevance**: Include only the relevant clause text, excluding any surrounding or unrelated content.
3. **Primary Instance**: If multiple instances of the clause exist, extract the most complete and primary version.
4. **Completeness**: Ensure the extracted text forms a complete and coherent clause.
5. **No Modifications**: Do not paraphrase, summarize, or modify the text in any way.
6. **Clause-Specific Rules**:
   - For "Parties": Extract the full legal names of the parties involved, along with their roles (e.g., "Licensor" or "Distributor"). Avoid generic labels such as "Company."
   - For "Effective Date": Extract only the language that specifies the start or commencement of the agreement. Do not include expiration or termination details.
   - For "Expiration Date": Extract only the language that specifies the term-end or expiry of the agreement. Do not include effective or start details.
   - For "Governing Law": Extract exactly one sentence that specifies the governing law (e.g., the applicable state or country). If the text only mentions forum or jurisdiction without specifying governing law, respond with "NOT FOUND."
7. **Clause Absence**: If the requested clause type cannot be found in the contract text, respond with `{"clause_text": "NOT FOUND"}` without any additional commentary.
8. **Legal Reasoning**: Apply detailed legal reasoning to ensure the accuracy and relevance of your extraction.

The input will provide:
- `Clause Type to Extract`: The specific type of clause to locate and extract.
- `Contract Text`: The excerpt of the contract to analyze.

Your response should consist solely of the JSON object as specified, with no additional explanations or commentary.
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
> The provided contract text is a sample of a Non-Disclosure and Non-Competition Agreement between a franchisee and its personnel. The text primarily focuses on confidentiality and non-competition obligations but does not explicitly include a clause specifying the governing law under which the agreeme...

**Output:**

### Example 2

**Reasoning:**
> The provided contract text does not include any clause explicitly stating the governing law or jurisdiction under which the agreement is to be interpreted or enforced. Without such a clause, it is not possible to determine the governing law from the text provided.

**Output:**

### Example 3

**Reasoning:**
> The provided contract text does not include any clause explicitly stating the governing law or jurisdiction under which the agreement is to be interpreted or enforced. A thorough review of the text reveals no mention of applicable state, country, or legal framework. Therefore, the governing law clau...

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
