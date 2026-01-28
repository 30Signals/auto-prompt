import dspy
import json
import textwrap
from . import config, extractors

def get_multiline_input():
    """Reads multiline input from user until they type 'END' or press Ctrl+D/Z."""
    print("\n" + "="*60)
    print("PASTE RESUME TEXT BELOW (Type 'END' on a new line to submit):")
    print("="*60)
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)

def main():
    # 1. Setup LLM (Quietly)
    llm_conf = config.get_llm_config()
    if llm_conf["provider"] == "azure":
        model_name = "azure/" + llm_conf["deployment"]
        lm = dspy.LM(model_name, api_key=llm_conf["api_key"], api_base=llm_conf["endpoint"], api_version=llm_conf["api_version"])
    else:
        lm = dspy.Google(model=llm_conf["model"], api_key=llm_conf["api_key"])
    dspy.settings.configure(lm=lm)

    # 2. Load Models
    print("\nLoading models...")
    baseline = extractors.BaselineModule()
    
    dspy_student = extractors.StudentModule()
    try:
        dspy_student.load("optimized_resume_module.json")
        print("✅ DSPy Optimized Module Loaded.")
    except Exception as e:
        print(f"⚠️ Could not load optimized module: {e}")
        print("Running unoptimized version for DSPy.")

    # 3. Interactive Loop
    while True:
        resume_text = get_multiline_input()
        if not resume_text.strip():
            print("Exiting...")
            break
            
        print("\nProcessing... (Running Baseline & DSPy)")
        
        # Run Baseline
        try:
            base_pred = baseline(unstructured_text=resume_text)
            base_json = {
                "job_role": base_pred.job_role,
                "skills": base_pred.skills,
                "education": base_pred.education,
                "experience_years": base_pred.experience_years
            }
        except Exception as e:
            base_json = {"error": str(e)}

        # Run DSPy
        try:
            dspy_pred = dspy_student(unstructured_text=resume_text)
            dspy_json = {
                "job_role": dspy_pred.job_role,
                "skills": dspy_pred.skills,
                "education": dspy_pred.education,
                "experience_years": dspy_pred.experience_years
            }
        except Exception as e:
            dspy_json = {"error": str(e)}

        # 4. Side-by-Side Output
        print("\n" + "="*100)
        print(f"{'BASELINE (Handcrafted)':<50} | {'DSPy (Optimized)':<50}")
        print("="*100)
        
        base_str = json.dumps(base_json, indent=2)
        dspy_str = json.dumps(dspy_json, indent=2)
        
        base_lines = base_str.split('\n')
        dspy_lines = dspy_str.split('\n')
        
        max_len = max(len(base_lines), len(dspy_lines))
        
        for i in range(max_len):
            b_line = base_lines[i] if i < len(base_lines) else ""
            d_line = dspy_lines[i] if i < len(dspy_lines) else ""
            print(f"{b_line:<50} | {d_line:<50}")
            
        # 5. Save to JSON
        output_entry = {
            "timestamp": str(dspy.datetime.datetime.now()),
            "input_text": resume_text,
            "ground_truth": "N/A (User Input)",
            "baseline": base_json,
            "dspy": dspy_json
        }
        
        filename = "live_demo_results.json"
        try:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    if not isinstance(history, list): history = []
            except (FileNotFoundError, json.JSONDecodeError):
                history = []
            
            history.append(output_entry)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to '{filename}'")
        except Exception as e:
            print(f"\n⚠️ Could not save JSON: {e}")

        print("="*100 + "\n")

if __name__ == "__main__":
    main()
