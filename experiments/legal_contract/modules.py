"""
Legal Contract Analysis DSPy Modules

Defines the baseline and optimized modules for legal clause extraction.
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
class ClauseExtractionSignature(dspy.Signature):
    """Extract specific clause from legal contract."""
    contract_text = dspy.InputField(desc="Legal contract text to analyze")
    clause_type = dspy.InputField(desc="Type of clause to extract (e.g., 'Governing Law', 'Termination')")
    clause_text = dspy.OutputField(desc="Extracted clause text verbatim from contract, or 'NOT FOUND'")


class EnhancedClauseExtractionSignature(dspy.Signature):
    """Extract specific clause with legal reasoning."""
    contract_text = dspy.InputField(desc="Legal contract text to analyze")
    clause_type = dspy.InputField(desc="Type of clause to extract")
    reasoning = dspy.OutputField(desc="Legal analysis identifying the clause location and relevance")
    clause_text = dspy.OutputField(desc="Extracted clause text verbatim from contract, or 'NOT FOUND'")


# Load and set baseline prompt as signature docstring
BASELINE_PROMPT = load_baseline_prompt()
ClauseExtractionSignature.__doc__ = BASELINE_PROMPT
EnhancedClauseExtractionSignature.__doc__ = BASELINE_PROMPT + "\n\nProvide detailed legal reasoning for your extraction."


class BaselineModule(dspy.Module):
    """Baseline module using strictly the handcrafted prompt text."""

    def __init__(self):
        super().__init__()

    def forward(self, contract_text, clause_type):
        # Prepare prompt from file
        prompt = BASELINE_PROMPT.replace("{{contract_text}}", contract_text)
        prompt = prompt.replace("{{clause_type}}", clause_type)

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
            clause_text = data.get('clause_text', 'NOT FOUND')
        except json.JSONDecodeError:
            clause_text = cleaned_response.strip() if cleaned_response else "NOT FOUND"

        return dspy.Prediction(clause_text=clause_text)


class StudentModule(dspy.Module):
    """Enhanced DSPy module with legal reasoning for clause extraction."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(EnhancedClauseExtractionSignature)

    def forward(self, contract_text, clause_type):
        result = self.predictor(contract_text=contract_text, clause_type=clause_type)
        return dspy.Prediction(clause_text=result.clause_text)
