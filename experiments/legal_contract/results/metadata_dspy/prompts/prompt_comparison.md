# Prompt Comparison

## Baseline Prompt

```text
# Baseline Metadata Prompt (Signature)

Extract contract metadata fields from contract text.

    Rules:
    - Return concise normalized values only, never section numbers like "Section 19" and never raw clause paragraphs.
    - Use "NOT FOUND" only when the contract truly lacks the clause/value.
    - Agreement Date / Effective Date: return an exact normalized date like "June 08, 2010" when present; if the contract only says the date is not specified or references an undefined commencement date, return "NOT FOUND".
    - Expiration Date: return ONLY one of these normalized forms: exact date like "March 18, 2021"; "X-Year (Y months) Initial Term"; that same phrase with ", Auto-Renewal" appended if present; "Co-terminous with Related Agreement"; "Event-Based Termination"; or "NOT FOUND".
    - Governing Law: return only jurisdiction name (e.g., "Delaware", "Texas, United States", "England and Wales"), not full sentence; do not return venue, arbitration forum, or court names.
    - Indemnification: return a normalized indemnification result only if explicit indemnify / hold harmless / defend language exists; otherwise return "NOT FOUND" or a narrow "not indemnification" label when the clause is clearly one of those categories.
    - Limitation Of Liability: return a concise normalized summary like "Yes | Excludes: ... | Cap: ... | Exceptions: ..." or "NOT FOUND".
    - Non-Compete: return a concise normalized summary like "Yes | Restricted Party: ... | Duration: ... | Scope: ..." or "NOT FOUND".
    - Parties: this field is always expected; return the principal signatory/legal-entity names only, separated by " | ". Exclude role labels like Provider, Recipient, Customer, Distributor, Party, and exclude descriptive prose.
    - Termination For Convenience: return only a normalized summary like "Yes | Notice: 30 days | Either Party | Without Cause" or "NOT FOUND".

## Fields
- Agreement Date
- Effective Date
- Expiration Date
- Governing Law
- Indemnification
- Limitation Of Liability
- Non-Compete
- Parties
- Termination For Convenience

## Field Metadata
```

## Optimized Prompt

# Optimized DSPy Prompt
============================================================

## Instructions

```
You are an information extraction assistant. From the provided contract text, extract the following contract metadata fields and output only normalized, concise values.

Hard rules (follow exactly):
- Output concise normalized values only. Never include section/clause numbers (e.g., do not say “Section 19”) and never paste raw clause paragraphs.
- Use `NOT FOUND` only when the contract truly lacks that clause/value.
- Agreement Date / Effective Date:
  - If an exact date is stated, return it in the form: “Month DD, YYYY” (e.g., “June 08, 2010”).
  - If the contract says the effective/agreement date is unspecified, depends on an undefined commencement date, or is otherwise not determinable, return `NOT FOUND`.
- Expiration Date: return exactly one of the following formats (choose the best match):
  1) Exact date: “Month DD, YYYY”
  2) “X-Year (Y months) Initial Term”
  3) “X-Year (Y months) Initial Term, Auto-Renewal” (only if auto-renewal is explicitly present)
  4) “Co-terminous with Related Agreement”
  5) “Event-Based Termination”
  6) `NOT FOUND`
- Governing Law: return only the jurisdiction name (examples: “Delaware”, “Texas, United States”, “England and Wales”).
  - Do not return full sentences.
  - Do not include venue, arbitration forum, or court names.
- Indemnification:
  - Return a normalized indemnification result ONLY if explicit indemnify/hold harmless/defend language exists.
  - Otherwise return `NOT FOUND`.
  - If it is clearly about something that is not indemnification (e.g., insurance only, confidentiality only) return `NOT FOUND` (no guessing).
- Limitation Of Liability:
  - Return a concise normalized summary: `Yes | Excludes: ... | Cap: ... | Exceptions: ...`
  - If no limitation of liability clause is found, return `NOT FOUND`.
  - If components are missing, leave them out but keep the `Yes | ...` structure as appropriate.
- Non-Compete:
  - Return a concise normalized summary: `Yes | Restricted Party: ... | Duration: ... | Scope: ...`
  - If no non-compete (or sufficiently similar restriction on competition) is found, return `NOT FOUND`.
- Parties:
  - Always expected.
  - Return principal signatory/legal-entity names only, separated by ` | `.
  - Exclude role labels and descriptive prose (e.g., exclude “Provider”, “Customer”, “Party”, “Recipient”, “Distributor”, etc.).
- Termination For Convenience:
  - If present, return: `Yes | Notice: <X> days | Either Party | Without Cause`
  - Use “Either Party” only if both sides can terminate for convenience; otherwise specify the applicable side(s) (e.g., “Only Company” / “Only Licensor”).
  - If not present, return `NOT FOUND`.

Output requirements:
- Respond with the corresponding output fields only.
- Do not add explanations or any extra text.
- If multiple candidates exist, prefer the explicit, most specific, and controlling language.
```

## Field Definitions

- **Contract Text:** Full contract text
- **Reasoning:** Step-by-step extraction rationale
- **Agreement Date:** Agreement Date as normalized exact date like June 08, 2010, or NOT FOUND
- **Effective Date:** Effective Date as normalized exact date like June 08, 2010, or NOT FOUND
- **Expiration Date:** Expiration Date only in normalized schema: exact date, X-Year (Y months) Initial Term, optional Auto-Renewal, Co-terminous with Related Agreement, Event-Based Termination, or NOT FOUND
- **Governing Law:** Governing Law jurisdiction only in normalized form, or NOT FOUND
- **Indemnification:** Normalized indemnification result only, else NOT FOUND
- **Limitation Of Liability:** Concise normalized limitation-of-liability summary, or NOT FOUND
- **Non Compete:** Concise normalized non-compete summary, or NOT FOUND
- **Parties:** Principal party names only, separated by |, or NOT FOUND
- **Termination For Convenience:** Normalized termination-for-convenience summary with notice/party if present, else NOT FOUND

## Few-Shot Examples

Total examples: **8**

### Example 1

**Reasoning:**
> Agreement date appears as “dated as of October 24, 2005”. Effective date is not separately stated as “effective as of”; therefore use NOT FOUND. Term length/expiration date is not clearly provided in the supplied excerpt; termination includes a 6-month notice right “for any reason,” but no stated en...

**Output:**
- agreement_date: `October 24, 2005`
- effective_date: `NOT FOUND`
- expiration_date: `3-Year (36 months) Initial Term, Auto-Renewal`
- governing_law: `Delaware`
- indemnification: `NOT FOUND`
- limitation_of_liability: `Yes | Excludes: Consequential, Incidental, Punitive, Exemplary, Indirect, Special | Remedy: Sole Remedy`
- non_compete: `NOT FOUND`
- parties: `Johnson Matthey Catalog Company, Inc | Nanophase Technologies Corporation`
- termination_for_convenience: `NOT FOUND`

### Example 2

**Reasoning:**
> Agreement date is not explicitly stated beyond the “effective as of” language; treat that as the agreement/effective date. Term is 5 years from execution with renewal by agreement; expiration date is not a fixed date, so use the normalized “X-Year (Y months) Initial Term” form. Governing law is not ...

**Output:**
- agreement_date: `October 01, 2008`
- effective_date: `October 01, 2008`
- expiration_date: `5-Year (60 months) Initial Term`
- governing_law: `United States`
- indemnification: `Yes | Explicit indemnification / defense / hold harmless obligations`
- limitation_of_liability: `Yes | Excludes: Consequential, Incidental, Exemplary, Indirect, Special, Lost Profits, Loss of Data | Exceptions: IP Infringement, Indemnification, Personal Injury/Death`
- non_compete: `NOT FOUND`
- parties: `Dynamic Hearing Pty Ltd | IntriCon Corporation`
- termination_for_convenience: `NOT FOUND`

### Example 3

**Reasoning:**
> Effective/Agreement date is stated on the cover page. Governing law is not explicitly shown in the provided excerpts. Indemnification language exists and is selected broadly. Limitation of liability appears partially (mutual waiver of certain damages), but key cap language is not fully present in th...

**Output:**
- agreement_date: `December 23, 2019`
- effective_date: `December 23, 2019`
- expiration_date: `5-Year (60 months) Initial Term, Auto-Renewal`
- governing_law: `New York`
- indemnification: `Yes | Explicit indemnification / defense / hold harmless obligations`
- limitation_of_liability: `NOT FOUND`
- non_compete: `Yes | Restricted Party: Party A | Duration: during the term | Scope: Non-Endorsement, Non-Solicitation`
- parties: `Legacy Education Alliance Holdings, Inc | T&B Seminars, Inc`
- termination_for_convenience: `NOT FOUND`

... and **5** more examples

## Reasoning Strategy

Uses **Chain-of-Thought** reasoning with intermediate rationale steps.
Each example includes explicit reasoning to guide the model's inference process.


## Key Differences

- **Few-shot examples**: 0 (baseline) vs 3 shown / 8 total (optimized)
- **Reasoning**: Implicit baseline extraction vs explicit Chain-of-Thought in optimized DSPy prompt
- **Optimization**: Baseline metadata signature vs automatic DSPy optimization (BootstrapFewShot + COPRO)

## Notes

The optimized prompt is generated through DSPy's two-stage optimization:
1. BootstrapFewShot: Selects effective few-shot examples from training data
2. COPRO: Refines instructions and example selection

The baseline prompt shown here is the metadata baseline signature used by `run_metadata_dspy.py`.
