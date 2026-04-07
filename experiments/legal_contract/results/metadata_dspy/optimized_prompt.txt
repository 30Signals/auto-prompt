# Optimized DSPy Prompt
============================================================

## Instructions

```
You are an extraction engine. Read the provided contract text and extract the following metadata fields. Output only the corresponding output fields (no explanations, no extra text). Always return normalized, concise values.

General rules (apply to every field):
- Return values only—never include section/clause numbers (e.g., no “Section 19”) and never paste raw clause paragraphs.
- If the contract truly does not contain the requested information, return exactly: "NOT FOUND".
- Do not infer missing terms from typical templates.
- Prefer exact wording where it identifies a value (dates, jurisdictions, named parties), but normalize formatting (especially dates).
- If a field is present but ambiguous/undefined (e.g., commencement date referenced but not defined), treat as missing and return "NOT FOUND" (or the appropriate "NOT FOUND" option mandated below).

Field-specific rules:

1) Parties (always required):
- Extract the principal signatory/legal entity names from the agreement.
- Output as: Entity1 | Entity2 | ...
- Exclude role labels and descriptive prose (e.g., exclude “Company”, “Provider”, “Customer”, “Party”, “Distributor”, “Recipient”, etc.).
- If multiple signatories exist, include all principal legal entities separated by " | ".

2) Agreement Date / Effective Date:
- If an explicit agreement date or effective date is stated, return an exact normalized date, formatted like: "June 08, 2010".
- If the contract says the date is unspecified, or references an undefined commencement/effective date, return "NOT FOUND".

3) Expiration Date (must match exactly one of the following formats):
Return ONLY one of these:
- An exact normalized date, e.g., "March 18, 2021"
- "X-Year (Y months) Initial Term"
- The same as the prior item with ", Auto-Renewal" appended if auto-renewal is stated, e.g., "2-Year (24 months) Initial Term, Auto-Renewal"
- "Co-terminous with Related Agreement" (if explicitly co-terminous)
- "Event-Based Termination" (if explicitly terminates upon an event)
- "NOT FOUND" otherwise

4) Governing Law:
- Return only the jurisdiction name (no full sentences).
- Examples: "Delaware", "Texas, United States", "England and Wales"
- Do NOT include venue, arbitration forum, court names, or dispute-resolution institutions.

5) Indemnification:
- Return a normalized indemnification result ONLY if the contract explicitly includes indemnify / hold harmless / defend language.
- If it contains language that is clearly not an indemnification obligation, return either "NOT FOUND" or a narrow label such as "not indemnification" (choose the narrow label only when it is clearly in a different category).
- Otherwise return "NOT FOUND".

6) Limitation Of Liability:
- If present, return a concise normalized summary in this shape:
  - "Yes | Excludes: ... | Cap: ... | Exceptions: ..."
- If not present, return "NOT FOUND".
- Keep it short; summarize categories without quoting.

7) Non-Compete:
- If present, return:
  - "Yes | Restricted Party: ... | Duration: ... | Scope: ..."
- If not present, return "NOT FOUND".
- Summarize duration and scope without quoting.

8) Termination For Convenience:
- Return only when the contract expressly permits termination for convenience / without cause / for any reason.
- Output format:
  - "Yes | Notice: <duration> | Either Party | Without Cause"
- If neither termination-for-convenience right exists, return "NOT FOUND".
- Do NOT treat breach/default/insolvency/bankruptcy/force majeure/failure-to-cure termination rights as termination for convenience.

Important output constraints:
- Output must strictly contain only the requested fields with their values.
- Never include extra commentary.
- If multiple date formats exist, use the one that is explicitly the agreement/effective/expiration date.

Now extract from the contract text accordingly.
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
- **Contract text:** Normalized termination-for-convenience summary with notice/party if present, else NOT FOUND

## Few-Shot Examples

Total examples: **8**

### Example 1

**Reasoning:**
> The contract is titled “DISTRIBUTOR AGREEMENT” dated October 24, 2005 and states it is governed by Delaware law. The term appears to be established via termination provisions (including termination “by either party for any reason by providing six (6) months prior written notice”), and no specific fi...

**Output:**
- agreement_date: `October 24, 2005`
- effective_date: `NOT FOUND`
- expiration_date: `3-Year (36 months) Initial Term, Auto-Renewal`
- governing_law: `Delaware`
- indemnification: `NOT FOUND`
- limitation_of_liability: `Yes | Excludes: Consequential, Incidental, Punitive, Exemplary, Indirect, Special, Lost Profits, Lost Revenue, Loss of Use, Business Interruption | Remedy: Sole Remedy, Exclusive Remedy`
- non_compete: `NOT FOUND`
- parties: `JOHNSON MATTHEY CATALOG COMPANY, INC | Nanophase Technologies Corporation`
- termination_for_convenience: `NOT FOUND`

### Example 2

**Reasoning:**
> Agreement cover text gives the Commencement/Effective date. Term clause states 5-year initial term. Governing law section is not clearly present, so governing law is NOT FOUND. Indemnity exists (IntriCon indemnifies Dynamic Hearing; also mutual concept and exclusions). Limitation of liability contai...

**Output:**
- agreement_date: `NOT FOUND`
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
> Agreement date appears as “12-23-2019, 2019” with an Effective Date defined as that same date. Term/end date and liability/covenants are not fully visible due to redactions/excerpting, so expiration is NOT FOUND. Governing law is not fully shown in the provided excerpt, so it is NOT FOUND.

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

## Runtime Demonstrations (3)

These examples were captured from the actual DSPy runtime prompt history. They make the optimized behavior visible even when the compiled DSPy object does not expose a rewritten instruction string directly.

### Runtime System Instructions

```text
Your input fields are:
1. `contract_text` (str): Full contract text
Your output fields are:
1. `agreement_date` (str): Agreement Date as normalized exact date like June 08, 2010, or NOT FOUND
2. `effective_date` (str): Effective Date as normalized exact date like June 08, 2010, or NOT FOUND
3. `expiration_date` (str): Expiration Date only in normalized schema: exact date, X-Year (Y months) Initial Term, optional Auto-Renewal, Co-terminous with Related Agreement, Event-Based Termination, or NOT FOUND
4. `governing_law` (str): Governing Law jurisdiction only in normalized form, or NOT FOUND
5. `indemnification` (str): Normalized indemnification result only, else NOT FOUND
6. `limitation_of_liability` (str): Concise normalized limitation-of-liability summary, or NOT FOUND
7. `non_compete` (str): Concise normalized non-compete summary, or NOT FOUND
8. `parties` (str): Principal party names only, separated by |, or NOT FOUND
9. `termination_for_convenience` (str): Normalized termination-for-convenience summary with notice/party if present, else NOT FOUND
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## contract_text ## ]]
{contract_text}

[[ ## agreement_date ## ]]
{agreement_date}

[[ ## effective_date ## ]]
{effective_date}

[[ ## expiration_date ## ]]
{expiration_date}

[[ ## governing_law ## ]]
{governing_law}

[[ ## indemnification ## ]]
{indemnification}

[[ ## limitation_of_liability ## ]]
{limitation_of_liability}

[[ ## non_compete ## ]]
{non_compete}

[[ ## parties ## ]]
{parties}

[[ ## termination_for_convenience ## ]]
{termination_for_convenience}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Extract contract metadata fields from contract text.
        
        Rules:
        - Return concise normalized values only, never section numbers like "Section 19" and never raw clause paragraphs.
        - Use "NOT FOUND" only when the contract truly lacks the clause/value.
        - Agreement Date / Effective Date: return an exact normalized date like "June 08, 2010" when present; if the contract only says the date is not specified or references an undefined commencement date, return "NOT FOUND".
        - Expiration Date: return ONLY one of these normalized forms: exact date like "March 18, 2021"; "X-Year (Y months) Initial Term"; that same phrase with ", Auto-Renewal" appended if present; "Co-terminous with Related Agreement"; "Event-Based Termination"; or "NOT FOUND".
        - Governing Law: return only jurisdiction name (e.g., "Delaware", "Texas, United States
... [truncated]
```

### Runtime Example 1

**User Input**
```text
[[ ## contract_text ## ]]
[Intro and Parties]
Exhibit 1

Execution Copy

JOINT VENTURE AGREEMENT

BETWEEN

TATE & LYLE FERMENTATION PRODUCTS LTD.

AND

IGENE BIOTECHNOLOGY, INC.

Table of Contents

1.

2.

DEFINITIONS

CREATION OF THE JOINT VENTURE AND ITS STRUCTURE

Page

1

4

3.

4.

PARTIES' CONTRIBUTIONS TO THE JOINT VENTURE AND FINANCING

PARTIES' INTERESTS IN THE JOINT VENTURE

4

6

5.

6.

[INTENTIONALLY OMITTED]

TAX CONSIDERATIONS

6

6

7.

8.

PARTIES' WITHDRAWAL FROM THE FIELD OF AGREEMENT

FUTURE PRODUCTS AND DEVELOPMENTS

8

8

9.

10.

MANAGEMENT AND CONTROL OF THE JOINT VENTURE

DAY-TO-DAY MANAGEMENT OF THE JOINT VENTURE

10

12

11. SERVICES TO BE PROVIDED TO THE JOINT VENTURE BY THE PARTIES 12

12. RIGHT OF FIRST REFUSAL; TERMINATION OF THE JOINT VENTURE AFTER THE EFFECTIVE DATE 14

13. LIABILITIES 15

14. WARRANTIES BY THE PARTIES 16

15. CONDITIONS PRECEDENT; TERMINATION OF AGREEMENT 17





16. OPERATIONS UNTIL TRANSFER 18

17. ASSIGNMENT; PLEDGE OF JOINT VENTURE INTEREST 19

18. COSTS, FEES AND TAXES 19

19. NOTICES 19

20. LAW 20

21. MISCELLANEOUS PROVISIONS 20

22. DISPUTE RESOLUTION PROCEDURES; LIQUIDATION 23

23. JOINT VENTURE TO BE BOUND 24

JOINT VENTURE AGREEMENT

This Joint Venture Agreement (as amended, restated, supplemented or otherwise modified in accordance herewith, this "Agreement") is entered into as of March 18, 2003 and is by and between Tate & Lyle Fermentation Products Ltd., a corporation organized under the laws of England ("T&L") and a subsidiary of Tate & Lyle PLC, and Igene Biotechnology, Inc., a Maryland corporation ("Igene"; collectively with T&L, the "Parties").

RECITALS

A. Igene owns certain technology and expertise related to the manufacture and sale of Astaxanthin and derivative products, and T&L and its subsidiar
... [truncated]
```

**Assistant Output**
```text
[[ ## agreement_date ## ]]
March 18, 2003

[[ ## effective_date ## ]]
NOT FOUND

[[ ## expiration_date ## ]]
NOT FOUND

[[ ## governing_law ## ]]
NOT FOUND

[[ ## indemnification ## ]]
NOT FOUND

[[ ## limitation_of_liability ## ]]
NOT FOUND

[[ ## non_compete ## ]]
NOT FOUND

[[ ## parties ## ]]
Tate & Lyle Fermentation Products Ltd. | Igene Biotechnology, Inc.

[[ ## termination_for_convenience ## ]]
NOT FOUND

[[ ## completed ## ]]
```

### Runtime Example 2

**User Input**
```text
[[ ## contract_text ## ]]
[Intro and Parties]
EXHIBIT B

COOPERATION AGREEMENT

This Agreement dated March 13, 2014 is by and between JANA Partners LLC ("JANA") and URS Corporation (the "Company"). In consideration of and reliance upon the mutual covenants and agreements contained herein, and for other good and valuable consideration, the receipt and sufficiency of which is hereby acknowledged, the parties hereto agree as follows:

1. Representations and Warranties of the Company. The Company represents and warrants to JANA that this Agreement has been duly authorized, executed and delivered by the Company, and is a valid and binding obligation of the Company, enforceable against the Company in accordance with its terms.

2. Representations and Warranties of JANA. JANA represents and warrants to the Company that this Agreement has been duly authorized, executed and delivered by JANA, and is a valid and binding obligation of JANA, enforceable against JANA in accordance with its terms. As of the date of this Agreement, JANA beneficially owns 6,745,623 shares of the Company's common stock and has voting authority over such shares.

3. Board Nomination and Other Company Matters.

(a) In accordance with the Company's By­Laws and Delaware law, the Company agrees that, effective as of the next meeting (the "March Board Meeting") of the Board of Directors of the Company (the "Board"), which shall be held not later than March 27, 2014, and prior to taking any other formal action at such meeting, the Board will:

(1) increase the size of the Board to fourteen (14) members; and

(2) appoint Diane C. Creel, William H. Schumann, III, David N. Siegel and V. Paul Unruh (together, the "JANA Nominees") (other than in the case of the refusal or inability of any such person to serve, in w
... [truncated]
```

**Assistant Output**
```text
[[ ## agreement_date ## ]]
March 13, 2014

[[ ## effective_date ## ]]
NOT FOUND

[[ ## expiration_date ## ]]
Co-terminous with Related Agreement

[[ ## governing_law ## ]]
Delaware

[[ ## indemnification ## ]]
NOT FOUND

[[ ## limitation_of_liability ## ]]
NOT FOUND

[[ ## non_compete ## ]]
NOT FOUND

[[ ## parties ## ]]
JANA Partners LLC | URS Corporation

[[ ## termination_for_convenience ## ]]
NOT FOUND

[[ ## completed ## ]]
```

### Runtime Example 3

**User Input**
```text
[[ ## contract_text ## ]]
[Intro and Parties]
EXHIBIT 99.2 Page 1 of 3 DISTRIBUTOR AGREEMENT Agreement made this 19t h day of March, 2020 Between: Co-Diagnostics, Inc. (herein referred to as "Principal") And PreCheck Health Services, Inc. (herein referred to as "Distributor"). In consideration of the mutual terms, conditions and covenants hereinafter set forth, Principal and Distributor acknowledge and agree to the following descriptions and conditions: DESCRIPTION OF PRINCIPAL The Principal is a company located in Utah, United States and is in the business of research and development of reagents. The Principal markets and sells it products globally through direct sales and distributors. DESCRIPTION OF DISTRIBUTOR The Distributor is a company operating or planning to operate in the United States of America, Latin America, Europe and Russia. The Distributor represents that the Distributor or a subsidiary of the Distributor is or will be fully licensed and registered in the Territory and will provide professional distribution services for the products of the Principal. CONDITIONS: 1. The Principal appoints the Distributor as a non-exclusive distributor, to sell Principal's qPCR infectious disease kits, Logix Smart COVID-19 PCR diagnostic test and Co-Dx Box™ instrument (the "Products"). The Products are described on Exhibit A to this Agreement. 2. The Principal grants Distributor non- exclusive rights to sell these products within the countries of Romania (the "Territory"), which may be amended by mutual written agreement.

Source: PRECHECK HEALTH SERVICES, INC., 8-K, 3/20/2020





3. The Distributor accepts the appointment and shall use its commercially reasonable efforts to promote, market and sell the Products within the Territory, devote such time and attention as may
... [truncated]
```

**Assistant Output**
```text
[[ ## agreement_date ## ]]
March 19, 2020

[[ ## effective_date ## ]]
NOT FOUND

[[ ## expiration_date ## ]]
March 18, 2021

[[ ## governing_law ## ]]
Texas

[[ ## indemnification ## ]]
NOT FOUND

[[ ## limitation_of_liability ## ]]
NOT FOUND

[[ ## non_compete ## ]]
Yes | Restricted Party: Distributor | Duration: During the term of the Agreement | Scope: No competitive products within the Territory (Romania)

[[ ## parties ## ]]
Co-Diagnostics, Inc. | PreCheck Health Services, Inc.

[[ ## termination_for_convenience ## ]]
Yes | Notice: 30 days | Either Party | Without cause

[[ ## completed ## ]]
```
