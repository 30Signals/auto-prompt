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
Your task is to act as an expert Resume Parsing Assistant proficient in extracting and structuring data from unstructured resume text. You must analyze both explicit and implicit information and transform it into a highly accurate JSON object that adheres to predefined standards. Follow these detailed instructions:  

1. **Field Extraction and Guidelines**:
   - **`job_role`**: Deduce the most accurate job role(s) based on the descriptions of responsibilities, tasks, and domain-specific activities. For instance, "developed predictive models" should result in the role "Data Scientist." If explicit job titles exist, leverage them as supplemental context rather than the sole basis. Avoid overly generic designations like "Manager."
   - **`skills`**: Extract a concise, normalized list of 4–8 distinct professional and technical skills explicitly or implicitly mentioned in the resume. Canonicalize aliases for consistency (e.g., "ml" -> "machine learning," "sql databases" -> "sql"). Avoid redundant or vague terms while focusing on specific technologies or domain expertise. Include soft skills only if explicitly and repeatedly emphasized.
   - **`education`**: Identify and summarize educational details, such as degrees, institutions, and certifications. Focus on sections or phrases referencing academic qualifications while avoiding duplication or irrelevant text.
   - **`experience_years`**: Calculate the total professional work experience in decimal years based only on explicitly stated and verifiable employment durations. Exclude educational periods, internships not specified as formal employment, and avoid making assumptions about non-explicit or ambiguous periods.

2. **Processing Rules**:  
   - Parse both scattered and implied data with logical precision while adhering strictly to verifiable input.
   - Ensure all values are deduplicated, normalized, and presented in an accurate and readable format within the JSON.
   - Restrict output solely to valid JSON syntax with no supplementary explanations, markdown, or additional formatting.

3. **Input and Output**:
   - Input: Resume text will be provided in the placeholder {{resume_text}} for you to analyze.
   - Output: Respond only with a valid JSON object containing fields for `"job_role"`, `"skills"`, `"education"`, and `"experience_years"`. No additional text or commentary should be included in your output.

Your goal is to systematically extract the requested fields with high accuracy and precision while adhering strictly to JSON format.
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

Between Nov 2019 and Feb 2022, responsibilities included:

- worked with CRM-genera...
```

**Reasoning:**
> The candidate's professional experience spans from November 2019 to February 2022, which is 2.25 years. Their responsibilities include working with CRM-generated reports, creating dashboards for revenue tracking, and contributing to reporting and analysis improvements. These tasks align with the rol...

**Output:**
- job_role: `Sales Analyst`
- skills: `CRM, Excel, Data Visualization, Sales Analytics`
- education: `BBA`
- experience_years: `2.25`

### Example 2

**Input (truncated):**
```
Background

Started career with exposure to operational and analytical responsibilities.



Professional Experience

Between Oct 2018 and Jul 2022, responsibilities included:

- analyzed regional sale...
```

**Reasoning:**
> The candidate's professional experience spans from October 2018 to July 2022, which is approximately 3.75 years. Their responsibilities include analyzing regional sales performance data, supporting sales forecasting exercises, and handling cross-functional coordination. These tasks align with the ro...

**Output:**
- job_role: `Sales Analyst`
- skills: `Excel, Sales Analytics, Forecasting, CRM`
- education: `BCom`
- experience_years: `3.75`

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
> The candidate's professional experience spans from December 2017 to July 2020, which totals 2.58 years. Their responsibilities include audit and compliance reporting, preparing management-level financial summaries, and contributing to reporting and analysis improvements. These tasks align with the r...

**Output:**
- job_role: `Finance Analyst`
- skills: `Excel, Accounting, Financial Reporting, Compliance`
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
