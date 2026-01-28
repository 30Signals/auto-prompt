import dspy
import os

def load_baseline_prompt():
    """Load the baseline prompt from text file"""
    prompt_path = os.path.join(os.path.dirname(__file__), 'handcrafted_prompt.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

class ResumeSignature(dspy.Signature):
    """Resume parsing signature that uses the baseline prompt"""
    unstructured_text = dspy.InputField(desc="Resume text to analyze")
    
    job_role = dspy.OutputField(desc="Primary job role/title based on work experience and activities")
    skills = dspy.OutputField(desc="Technical and professional skills inferred from work descriptions, projects, and activities")
    education = dspy.OutputField(desc="Highest educational qualification or degree")
    experience_years = dspy.OutputField(desc="Total years of professional experience as decimal number")

class EnhancedResumeSignature(dspy.Signature):
    """Enhanced resume parsing with detailed reasoning"""
    unstructured_text = dspy.InputField(desc="Resume text to analyze")
    reasoning = dspy.OutputField(desc="Step-by-step analysis of work experience, skills inference, and role determination")
    
    job_role = dspy.OutputField(desc="Primary job role/title based on work experience and activities")
    skills = dspy.OutputField(desc="Technical and professional skills inferred from work descriptions, projects, and activities")
    education = dspy.OutputField(desc="Highest educational qualification or degree")
    experience_years = dspy.OutputField(desc="Total years of professional experience as decimal number")

# Load the baseline prompt once
BASELINE_PROMPT = load_baseline_prompt()

# Set the signature docstring to the loaded prompt
ResumeSignature.__doc__ = BASELINE_PROMPT
EnhancedResumeSignature.__doc__ = BASELINE_PROMPT + "\n\nProvide detailed reasoning for your analysis before extracting information."

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
        
        # Basic cleanup to handle potential markdown wrappers if the LLM ignores instructions
        cleaned_response = response[0] if isinstance(response, list) else response
        if hasattr(cleaned_response, 'strip'):
            cleaned_response = cleaned_response.strip()
        else:
            cleaned_response = str(cleaned_response).strip()
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
    """Enhanced DSPy module with multi-stage reasoning"""
    def __init__(self):
        super().__init__()
        # Use enhanced signature with reasoning
        self.predictor = dspy.ChainOfThought(EnhancedResumeSignature)
    
    def forward(self, unstructured_text):
        result = self.predictor(unstructured_text=unstructured_text)
        return dspy.Prediction(
            job_role=result.job_role,
            skills=result.skills,
            education=result.education,
            experience_years=result.experience_years
        )