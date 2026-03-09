"""
Company Legal Risk DSPy Modules
"""

import json
from pathlib import Path

import dspy

PROMPTS_DIR = Path(__file__).parent / "prompts"

#load baseline prompt
def load_baseline_prompt():
    prompt_path = PROMPTS_DIR / "baseline.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()
    

# DSPy Signatures
def _strip_fences(text):
    value = str(text or "").strip()
    if value.startswith("```json"):
        value = value[7:]
    if value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    return value.strip()

# Normalization helpers
def normalize_risk(value):
    label = str(value or "").strip().upper()
    if label in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        return label
    if "CRIT" in label:
        return "CRITICAL"
    if "HIGH" in label:
        return "HIGH"
    if "MED" in label:
        return "MEDIUM"
    return "LOW"

# Normalize decision to YES/NO with leniency for various negative forms
def normalize_decision(value):
    label = str(value or "").strip().upper()
    if label in {"NO", "N", "REJECT", "DO NOT WORK"}:
        return "NO"
    if label in {"YES", "Y", "APPROVE", "WORK"}:
        return "YES"
    # Default conservative for ambiguous outputs.
    return "NO"

# Normalize key findings to a comma-separated list, handling various output formats
def normalize_findings(value):
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return ", ".join(parts)
    text = str(value or "").strip()
    return text.replace("\n", ", ")


RISK_KEYWORDS = {
    "CRITICAL": [
        "criminal",
        "indict",
        "doj",
        "department of justice",
        "fraud",
        "money laundering",
        "bribery",
        "sanction",
        "antitrust lawsuit",
        "class action",
    ],
    "HIGH": [
        "lawsuit",
        "regulator",
        "regulatory",
        "probe",
        "investigation",
        "fine",
        "penalty",
        "settlement",
        "violation",
        "litigation",
        "compliance",
    ],
}


def conservative_risk_floor(search_context):
    text = str(search_context or "").lower()
    critical_hits = sum(1 for kw in RISK_KEYWORDS["CRITICAL"] if kw in text)
    high_hits = sum(1 for kw in RISK_KEYWORDS["HIGH"] if kw in text)
    if critical_hits >= 3:
        return "CRITICAL"
    if critical_hits >= 2 or high_hits >= 4:
        return "HIGH"
    return "LOW"


def _risk_index(level):
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(level, 0)


def apply_conservative_policy(risk_level, decision, search_context):
    normalized_risk = normalize_risk(risk_level)
    normalized_decision = normalize_decision(decision)
    floor = conservative_risk_floor(search_context)

    if _risk_index(normalized_risk) < _risk_index(floor):
        normalized_risk = floor

    # Critical risk should not produce a YES recommendation.
    if normalized_risk == "CRITICAL":
        normalized_decision = "NO"

    # If policy floor indicates at least HIGH risk, avoid approving.
    if floor in {"HIGH", "CRITICAL"} and normalized_decision == "YES":
        normalized_decision = "NO"

    # Keep risk and decision coherent.
    if normalized_decision == "YES" and normalized_risk == "CRITICAL":
        normalized_risk = "MEDIUM"

    return normalized_risk, normalized_decision

# Build search context from retrieved results, concatenating relevant fields while respecting a max character limit.
class CompanyLegalRiskSignature(dspy.Signature):
    """Assess legal and compliance risk for a company using retrieved evidence."""

    company_name = dspy.InputField(desc="Target company name")
    search_context = dspy.InputField(desc="Consolidated search snippets about lawsuits, compliance, and news")
    risk_level = dspy.OutputField(desc="One of LOW, MEDIUM, HIGH, CRITICAL")
    should_work_with_company = dspy.OutputField(desc="YES or NO")
    summary = dspy.OutputField(desc="Short recommendation summary")
    key_findings = dspy.OutputField(desc="Comma-separated key risk findings")

# Load and set baseline prompt as signature docstring
class EnhancedCompanyLegalRiskSignature(dspy.Signature):
    """Assess legal and compliance risk with explicit reasoning over evidence."""

    company_name = dspy.InputField(desc="Target company name")
    search_context = dspy.InputField(desc="Consolidated search snippets about lawsuits, compliance, and news")
    reasoning = dspy.OutputField(desc="Step-by-step legal and compliance risk reasoning")
    risk_level = dspy.OutputField(desc="One of LOW, MEDIUM, HIGH, CRITICAL")
    should_work_with_company = dspy.OutputField(desc="YES or NO")
    summary = dspy.OutputField(desc="Short recommendation summary")
    key_findings = dspy.OutputField(desc="Comma-separated key risk findings")

BASELINE_PROMPT = load_baseline_prompt()
CONSERVATIVE_APPENDIX = """
Additional policy:
- If evidence mentions lawsuits, regulatory enforcement, major fines, fraud probes, sanctions, antitrust, AML/KYC failures, or class actions, risk must be at least HIGH and recommendation should be NO.
- Use CRITICAL for multiple severe legal signals or ongoing major government enforcement.
- Include 3-5 concrete key findings copied closely from evidence wording.
"""
CompanyLegalRiskSignature.__doc__ = BASELINE_PROMPT

# The enhanced signature includes an additional field for reasoning, and its docstring encourages detailed legal-risk reasoning.

EnhancedCompanyLegalRiskSignature.__doc__ = (
    BASELINE_PROMPT
    + "\n\nInclude explicit legal-risk reasoning before final decision.\n"
    + CONSERVATIVE_APPENDIX
)

# Helper to build search context from retrieved results, concatenating relevant fields while respecting a max character limit.
class BaselineModule(dspy.Module):
    def __init__(self):
        super().__init__()
        
    # The forward method takes company_name and search_context as inputs, prepares the prompt, calls the LLM, and parses the response to produce the required outputs.

    def forward(self, company_name, search_context):
        prompt = (BASELINE_PROMPT + "\n\n" + CONSERVATIVE_APPENDIX).replace("{{company_name}}", company_name)
        prompt = prompt.replace("{{search_context}}", search_context)

        response = dspy.settings.lm(prompt)
        text = response[0] if isinstance(response, list) else response
        text = _strip_fences(text)

        risk_level = "LOW"
        decision = "YES"
        summary = ""
        findings = ""

        # Attempt to parse the LLM response as JSON to extract structured outputs; if parsing fails, return the raw text as the summary.

        try:
            payload = json.loads(text)
            risk_level = normalize_risk(payload.get("risk_level", "LOW"))
            decision = normalize_decision(payload.get("should_work_with_company", "NO"))
            summary = str(payload.get("summary", "")).strip()
            findings = normalize_findings(payload.get("key_findings", ""))
        except json.JSONDecodeError:
            summary = text

        risk_level, decision = apply_conservative_policy(risk_level, decision, search_context)
        # Return a Prediction object with the normalized risk level, decision, summary, and key findings extracted from the LLM response.

        return dspy.Prediction(
            risk_level=risk_level,
            should_work_with_company=decision,
            summary=summary,
            key_findings=findings,
        )

# The StudentModule uses a Chain of Thought predictor with the enhanced signature, allowing it to generate detailed reasoning in addition to the final outputs.
class StudentModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(EnhancedCompanyLegalRiskSignature)

    # The forward method calls the predictor with the company name and search context, and then normalizes the outputs to ensure consistent formatting for risk level, decision, summary, and key findings.
    
    def forward(self, company_name, search_context):
        result = self.predictor(company_name=company_name, search_context=search_context)
        risk_level, decision = apply_conservative_policy(
            result.risk_level,
            result.should_work_with_company,
            search_context,
        )
        return dspy.Prediction(
            risk_level=risk_level,
            should_work_with_company=decision,
            summary=str(result.summary).strip(),
            key_findings=normalize_findings(result.key_findings),
        )
