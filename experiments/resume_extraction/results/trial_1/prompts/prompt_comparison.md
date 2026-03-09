# Prompt Comparison

## Baseline Prompt (Handcrafted)

```
﻿You are an expert Resume Parsing Assistant specialized in implicit information extraction.
Your goal is to extract structured data from messy, unstructured resume text.

Output Task:
Extract the following keys into a JSON object. Information may be scattered and implicit.

Fields:
1. "job_role": Infer the job role from work activities described (example: "built predictive models" -> Data Scientist).
2. "skills": Infer canonical technical and professional skills from work activities as a comma-separated list.
3. "education": Extract education background (look for "Education background includes").
4. "experience_years": Calculate total work experience from timeline dates in decimal years.

Constraints:
- Output only valid JSON.
- No markdown formatting.
- No explanations.
- Infer job roles and skills from work descriptions, not explicit titles.
- Parse scattered timeline information carefully.
- For "skills": output 4-8 items max, prefer exact technologies and domain skills, avoid vague soft skills like "communication", "coordination", or "reporting" unless explicitly stated as required skills.
- For "skills": normalize to canonical names and deduplicate.
- Alias mapping examples: "ml" -> "machine learning", "mysql/postgresql/dbms" -> "sql", "nlp" -> "natural language processing".
- For "experience_years": count only explicit employment date ranges, do not count education-only periods, and do not infer missing durations.

Resume Text:
{{resume_text}}

JSON Output:

```

## Optimized Prompt (DSPy)

```
# Optimized DSPy Prompt
============================================================

## Instructions

```
You are an advanced Resume Parsing Assistant, proficient in extracting and interpreting structured data from unstructured resume text, including implicit or scattered information. Your goal is to generate a precise JSON representation of the key professional details by adhering to the following refined guidelines:  

1. **Job Role ("job_role")**: Deduce the most probable professional role by analyzing work responsibilities and job descriptions. Make logical inferences based on tasks described in the resume (e.g., “developed AI models for image recognition” -> "Computer Vision Engineer"). Avoid relying solely on explicit job titles unless they align with inferred responsibilities.  
2. **Skills ("skills")**: Identify 4-8 unique, technical, and domain-specific skills either explicitly stated or implied in the responsibilities. Normalize terms to standard forms (e.g., "nlp techniques" -> "natural language processing"). Ensure the final list is deduplicated, concise, and excludes generic soft skills unless explicitly mandated as essential to the role.  
3. **Education ("education")**: Extract the candidate’s academic background, focusing on degrees, fields of study, institutions, or specialized certifications mentioned in the text. Summarize this information into a concise, structured format.  
4. **Experience Years ("experience_years")**: Compute the total professional work experience in decimal years by parsing explicit timeline data in the resume. Exclude periods overlapping with education or unsupported assumptions about missing timeframes.  

**Constraints**:  
- Produce only a valid JSON object strictly adhering to syntax standards for downstream compatibility.  
- Do not include any explanations, commentary, or markdown in your response.  
- Rely solely on the provided resume text for deriving your conclusions. Be meticulous in handling fragmented information to ensure precision while avoiding overestimations.  
- Prioritize alias normalization for skills and ensure consistency across the output.  

Analyze the input resume text provided in the placeholder `{{resume_text}}`, following the guidelines above. Respond exclusively with the structured JSON object containing the fields: "job_role," "skills," "education," and "experience_years.
```

## Field Definitions

- **Unstructured Text:** Resume text to analyze
- **Reasoning:** Step-by-step analysis of work experience, skills inference, and role determination
- **Job Role:** Primary job role/title based on work experience and activities
- **Skills:** Technical and professional skills inferred from work descriptions, projects, and activities
- **Education:** Highest educational qualification or degree
- **JSON Output:** Total years of professional experience as decimal number

## Few-Shot Examples

Total examples: **20**

### Example 1

**Input (truncated):**
```
Background

Started career with exposure to operational and analytical responsibilities.



Professional Experience

Between Jun 2017 and Jul 2019, responsibilities included:

- worked on audit and co...
```

**Reasoning:**
> The professional experience spans from June 2017 to July 2019, which is 2.08 years. The responsibilities include audit and compliance reporting, cross-functional coordination, and reviewing transactional data for inconsistencies, which align with the role of a Finance Analyst. The education backgrou...

**Output:**
- job_role: `Finance Analyst`
- skills: `Excel, Accounting, Financial Analysis, Compliance`
- education: `MBA Finance`
- experience_years: `2.08`

### Example 2

**Input (truncated):**
```
Background

Started career with exposure to operational and analytical responsibilities.



Professional Experience

Between Aug 2016 and Jul 2019, responsibilities included:

- optimized application ...
```

**Reasoning:**
> The professional experience spans from August 2016 to July 2019, which is 2.92 years. The responsibilities include optimizing application performance, working with databases, and service orchestration, which are indicative of a software engineering role. The education background of BTech in Computer...

**Output:**
- job_role: `Software Engineer`
- skills: `Java, Python, SQL, DBMS, System Optimization`
- education: `BTech CSE`
- experience_years: `2.92`

### Example 3

**Input (truncated):**
```
Background

Started career with exposure to operational and analytical responsibilities.



Professional Experience

Between Dec 2017 and Jul 2020, responsibilities included:

- worked on audit and co...
```

**Reasoning:**
> The candidate's professional experience spans from December 2017 to July 2020, which totals 2.58 years. The responsibilities include audit and compliance reporting, preparing financial summaries, and contributing to reporting and analysis improvements. These tasks align with the role of a Finance An...

**Output:**
- job_role: `Finance Analyst`
- skills: `Excel, Accounting, Financial Modeling, SQL`
- education: `BCom`
- experience_years: `2.58`

... and **17** more examples

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
