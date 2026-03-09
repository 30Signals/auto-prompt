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


CLAUSE_KEYWORDS = {
    "Parties": ["between", "party", "parties", "by and between", "agreement is made"],
    "Agreement Date": ["agreement date", "dated", "date of this agreement", "as of"],
    "Effective Date": ["effective date", "effective as of", "commences", "commencement"],
    "Expiration Date": ["expiration", "expires", "term ends", "termination date", "end date"],
    "Governing Law": ["governing law", "laws of", "governed by", "construed in accordance"],
    "Termination For Convenience": ["termination for convenience", "terminate for convenience", "at any time"],
    "Limitation Of Liability": ["limitation of liability", "liable", "liability shall not exceed"],
    "Indemnification": ["indemnify", "indemnification", "hold harmless"],
    "Non-Compete": ["non-compete", "noncompete", "not compete", "competitive activity"],
    "Confidentiality": ["confidentiality", "confidential information", "non-disclosure", "nondisclosure"],
}


def _normalize_clause_output(value):
    if value is None:
        return "NOT FOUND"
    text = str(value).strip()
    if not text:
        return "NOT FOUND"
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return text if text else "NOT FOUND"


def _is_short_generic_party_output(text):
    words = [w for w in str(text).strip().lower().replace(".", " ").split() if w]
    # Treat only single-token generic labels as invalid. Two-word spans can be
    # legitimate legal entities in this dataset (e.g., "Acme Corp").
    if len(words) != 1:
        return False
    generic = {"company", "distributor", "seller", "buyer", "party", "parties", "customer", "vendor"}
    return any(w in generic for w in words)


def _enforce_clause_constraints(clause_type, clause_text):
    """
    Apply lightweight clause-specific constraints to reduce false positives.
    """
    text = _normalize_clause_output(clause_text)
    if text == "NOT FOUND":
        return text

    ct = str(clause_type or "").strip()
    lower = text.lower()

    if ct == "Governing Law":
        # Keep this permissive: many valid clauses use varied wording and can
        # include jurisdiction language in the same sentence.
        has_law_signal = any(
            token in lower
            for token in (
                "governed by",
                "governing law",
                "laws of",
                "law of",
                "construed in accordance",
                "interpreted in accordance",
            )
        )
        if not has_law_signal:
            return "NOT FOUND"
        return text

    if ct == "Parties":
        if _is_short_generic_party_output(text):
            return "NOT FOUND"

    if ct == "Effective Date":
        # Avoid returning explicit expiry/term-end text for Effective Date.
        if any(token in lower for token in ("expire", "expires", "termination date", "term ends", "end date")):
            return "NOT FOUND"

    if ct == "Expiration Date":
        # Avoid returning only commencement/effective start phrasing.
        if ("effective date" in lower or "commence" in lower) and not any(
            token in lower for token in ("expire", "expires", "termination", "end of", "term shall")
        ):
            return "NOT FOUND"

    return text


def _targeted_contract_context(contract_text, clause_type, max_chars=3200):
    """
    Select the most relevant contract segments for the requested clause type.
    This reduces noise for long contracts and improves exact span extraction.
    """
    if not contract_text:
        return ""
    ct = str(clause_type or "").strip()
    effective_max_chars = max_chars
    if ct in {"Parties", "Governing Law", "Expiration Date"}:
        effective_max_chars = max(max_chars, 4200)

    if len(contract_text) <= effective_max_chars:
        return contract_text

    clause_key = clause_type.split("(")[0].strip()
    keywords = [k.lower() for k in CLAUSE_KEYWORDS.get(clause_key, [])]
    chunks = [c.strip() for c in contract_text.split("\n\n") if c.strip()]
    if not chunks:
        return contract_text[:effective_max_chars]

    scored = []
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        hits = sum(chunk_lower.count(k) for k in keywords)
        # Small bonus for section headers that often carry clause names.
        header_bonus = 1 if idx > 0 and len(chunk) < 180 and any(k in chunk_lower for k in keywords) else 0
        score = hits + header_bonus
        scored.append((score, idx, chunk))

    # Pick top chunks with any keyword hits. If none, use leading context fallback.
    hit_chunks = [x for x in scored if x[0] > 0]
    if not hit_chunks:
        return contract_text[:effective_max_chars]

    hit_chunks.sort(key=lambda x: (-x[0], x[1]))
    selected = []
    used = set()
    total_len = 0
    for _, idx, chunk in hit_chunks:
        # Include neighboring chunk for boundary continuity.
        for neighbor in (idx - 1, idx, idx + 1):
            if neighbor < 0 or neighbor >= len(chunks) or neighbor in used:
                continue
            candidate = chunks[neighbor]
            if total_len + len(candidate) + 2 > effective_max_chars:
                continue
            selected.append(candidate)
            used.add(neighbor)
            total_len += len(candidate) + 2
        if total_len >= effective_max_chars * 0.9:
            break

    return "\n\n".join(selected) if selected else contract_text[:effective_max_chars]


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
        focused_context = _targeted_contract_context(contract_text, clause_type)

        # Prepare prompt from file
        prompt = BASELINE_PROMPT.replace("{{contract_text}}", focused_context)
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
            clause_text = _enforce_clause_constraints(
                clause_type,
                data.get('clause_text', 'NOT FOUND'),
            )
        except json.JSONDecodeError:
            clause_text = _enforce_clause_constraints(clause_type, cleaned_response)

        return dspy.Prediction(clause_text=clause_text)


class StudentModule(dspy.Module):
    """Enhanced DSPy module with legal reasoning for clause extraction."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(EnhancedClauseExtractionSignature)

    def forward(self, contract_text, clause_type):
        focused_context = _targeted_contract_context(contract_text, clause_type)
        result = self.predictor(contract_text=focused_context, clause_type=clause_type)
        return dspy.Prediction(
            clause_text=_enforce_clause_constraints(clause_type, result.clause_text)
        )

