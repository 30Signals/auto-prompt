"""
Medical NER DSPy Modules

Defines the baseline and optimized modules for disease entity extraction.
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


def normalize_diseases_output(value):
    """Normalize model output to a strict comma-separated disease list."""
    if value is None:
        return ""

    # Already a list-like output
    if isinstance(value, list):
        items = [str(v).strip() for v in value]
        return ", ".join([v for v in items if v])

    text = str(value).strip()
    if not text:
        return ""

    # Remove markdown fences
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # If model returned JSON, extract diseases field when present
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = parsed.get("diseases", "")
        else:
            text = parsed
    except (json.JSONDecodeError, TypeError):
        pass

    if isinstance(text, list):
        items = [str(v).strip() for v in text]
        return ", ".join([v for v in items if v])

    text = str(text).strip()
    if not text:
        return ""

    # Normalize common separators/labels from free-form outputs
    text = text.replace("\n", ",")
    text = text.replace(";", ",")
    text = text.replace("|", ",")
    text = text.replace("Diseases:", "")
    text = text.replace("diseases:", "")

    parts = [p.strip(" -*\t\r") for p in text.split(",")]
    parts = [p for p in parts if p]
    return ", ".join(parts)


# DSPy Signatures
class DiseaseExtractionSignature(dspy.Signature):
    """Extract disease entities from biomedical text."""
    abstract_text = dspy.InputField(desc="Biomedical abstract text to analyze")
    diseases = dspy.OutputField(desc="Comma-separated list of disease names found in the text")


class EnhancedDiseaseExtractionSignature(dspy.Signature):
    """Extract disease entities with detailed reasoning."""
    abstract_text = dspy.InputField(desc="Biomedical abstract text to analyze")
    reasoning = dspy.OutputField(desc="Step-by-step analysis identifying disease mentions and their context")
    diseases = dspy.OutputField(desc="Comma-separated list of disease names found in the text")


# Load and set baseline prompt as signature docstring
BASELINE_PROMPT = load_baseline_prompt()
DiseaseExtractionSignature.__doc__ = BASELINE_PROMPT
EnhancedDiseaseExtractionSignature.__doc__ = BASELINE_PROMPT + "\n\nProvide detailed reasoning for your entity identification."


class BaselineModule(dspy.Module):
    """Baseline module using strictly the handcrafted prompt text."""

    def __init__(self):
        super().__init__()

    def forward(self, abstract_text):
        # Prepare prompt from file
        prompt = BASELINE_PROMPT.replace("{{abstract_text}}", abstract_text)

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
            diseases_list = data.get('diseases', [])
            diseases = normalize_diseases_output(diseases_list)
        except json.JSONDecodeError:
            diseases = normalize_diseases_output(cleaned_response)

        return dspy.Prediction(diseases=diseases)


class StudentModule(dspy.Module):
    """Enhanced DSPy module with multi-stage reasoning for NER."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(EnhancedDiseaseExtractionSignature)

    def forward(self, abstract_text):
        result = self.predictor(abstract_text=abstract_text)
        return dspy.Prediction(diseases=normalize_diseases_output(result.diseases))
