"""
Resume Extraction DSPy Modules

Defines the baseline and optimized modules for resume information extraction.
"""

import dspy
import json
from pathlib import Path

# Load baseline prompt
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_baseline_prompt():
    """Load the baseline prompt from text file."""
    prompt_path = PROMPTS_DIR / "baseline.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


# DSPy Signatures
class ResumeSignature(dspy.Signature):
    """Resume parsing signature."""
    unstructured_text = dspy.InputField(desc="Resume text to analyze")

    job_role = dspy.OutputField(desc="Primary job role/title based on work experience and activities")
    skills = dspy.OutputField(desc="Technical and professional skills inferred from work descriptions, projects, and activities")
    education = dspy.OutputField(desc="Highest educational qualification or degree")
    experience_years = dspy.OutputField(desc="Total years of professional experience as decimal number")


class EnhancedResumeSignature(dspy.Signature):
    """Enhanced resume parsing with detailed reasoning."""
    unstructured_text = dspy.InputField(desc="Resume text to analyze")
    reasoning = dspy.OutputField(desc="Step-by-step analysis of work experience, skills inference, and role determination")

    job_role = dspy.OutputField(desc="Primary job role/title based on work experience and activities")
    skills = dspy.OutputField(desc="Technical and professional skills inferred from work descriptions, projects, and activities")
    education = dspy.OutputField(desc="Highest educational qualification or degree")
    experience_years = dspy.OutputField(desc="Total years of professional experience as decimal number")


# Load and set baseline prompt as signature docstring
BASELINE_PROMPT = load_baseline_prompt()
ResumeSignature.__doc__ = BASELINE_PROMPT
EnhancedResumeSignature.__doc__ = BASELINE_PROMPT + "\n\nProvide detailed reasoning for your analysis before extracting information."


class BaselineModule(dspy.Module):
    """Baseline module using strictly the handcrafted prompt text."""

    def __init__(self):
        super().__init__()

    def forward(self, unstructured_text):
        # Prepare prompt from file
        prompt = BASELINE_PROMPT.replace("{{resume_text}}", unstructured_text)

        # Call LLM directly (raw generation, no DSPy optimizations)
        response = dspy.settings.lm(prompt)

        # Parse JSON response
        cleaned_response = response[0] if isinstance(response, list) else response
        if hasattr(cleaned_response, 'strip'):
            cleaned_response = cleaned_response.strip()
        else:
            cleaned_response = str(cleaned_response).strip()

        # Handle markdown wrappers
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        try:
            data = json.loads(cleaned_response.strip())
        except json.JSONDecodeError:
            data = {}

        return dspy.Prediction(
            job_role=data.get('job_role', ''),
            skills=data.get('skills', ''),
            education=data.get('education', ''),
            experience_years=str(data.get('experience_years', ''))
        )


class StudentModule(dspy.Module):
    """Enhanced DSPy module with multi-stage reasoning."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(EnhancedResumeSignature)

    def forward(self, unstructured_text):
        result = self.predictor(unstructured_text=unstructured_text)
        return dspy.Prediction(
            job_role=result.job_role,
            skills=result.skills,
            education=result.education,
            experience_years=result.experience_years
        )
