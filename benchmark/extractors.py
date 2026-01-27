import dspy
import os

def load_baseline_prompt():
    """Load the baseline prompt from text file"""
    prompt_path = os.path.join(os.path.dirname(__file__), 'handcrafted_prompt.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Extract just the prompt part (after the header)
    lines = content.split('\n')
    prompt_start = None
    for i, line in enumerate(lines):
        if 'You are an expert HR resume parser' in line:
            prompt_start = i
            break
    if prompt_start:
        return '\n'.join(lines[prompt_start:])
    return content

class ResumeSignature(dspy.Signature):
    """Resume parsing signature that uses the baseline prompt"""
    unstructured_text = dspy.InputField(desc="Resume text")
    
    job_role = dspy.OutputField(desc="Job role")
    skills = dspy.OutputField(desc="Skills as comma-separated list")
    education = dspy.OutputField(desc="Highest degree")
    experience_years = dspy.OutputField(desc="Experience in years as number")

# Load the baseline prompt once
BASELINE_PROMPT = load_baseline_prompt()

# Set the signature docstring to the loaded prompt
ResumeSignature.__doc__ = BASELINE_PROMPT

class BaselineModule(dspy.Module):
    """Baseline module using strictly the handcrafted prompt text"""
    def __init__(self):
        super().__init__()
        # No dspy.Predict here, we use the raw LM
        
    def forward(self, unstructured_text):
        # 1. Prepare Prompt strictly from file
        prompt = BASELINE_PROMPT.replace("{{resume_text}}", unstructured_text)
        
        # 2. Call LLM directly (Raw Generation)
        # dspy.settings.lm is the configured LM (Azure/Gemini)
        response = dspy.settings.lm(prompt)
        
        # 3. Parse JSON (Manual extraction to ensure "nothing else")
        # We expect the LLM to output JSON as per instructions
        import json
        import re
        
        # Basic cleanup to handle potential markdown wrappers if the LLM ignores instructions
        cleaned_response = response[0] if isinstance(response, list) else response
        cleaned_response = cleaned_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        try:
            data = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Fallback for malformed JSON - return empties
            data = {}
            
        # 4. Wrap in Prediction for compatibility with Evaluate
        return dspy.Prediction(
            job_role=data.get('job_role', ''),
            skills=data.get('skills', ''),
            education=data.get('education', ''),
            experience_years=str(data.get('experience_years', ''))
        )

class StudentModule(dspy.Module):
    """DSPy module that uses ChainOfThought for automatic optimization"""
    def __init__(self):
        super().__init__()
        # ChainOfThought automatically optimizes the prompt with reasoning
        self.predictor = dspy.ChainOfThought(ResumeSignature)
    
    def forward(self, unstructured_text):
        return self.predictor(unstructured_text=unstructured_text)