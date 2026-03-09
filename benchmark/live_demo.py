import dspy
import json
import textwrap
import datetime
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
            "timestamp": str(datetime.datetime.now()),
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

# import dspy
# import json
# from . import config, extractors
# import datetime
# import re
# from typing import Optional, Tuple, List

# MONTH_MAP = {
#     "jan": 1, "january": 1,
#     "feb": 2, "february": 2,
#     "mar": 3, "march": 3,
#     "apr": 4, "april": 4,
#     "may": 5,
#     "jun": 6, "june": 6,
#     "jul": 7, "july": 7,
#     "aug": 8, "august": 8,
#     "sep": 9, "sept": 9, "september": 9,
#     "oct": 10, "october": 10,
#     "nov": 11, "november": 11,
#     "dec": 12, "december": 12,
# }

# DATE_RANGE_PATTERN = re.compile(
#     r"(?P<smon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
#     r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
#     r"\s+(?P<syear>\d{4})\s*(?:-|to|–|—)\s*"
#     r"(?:(?P<emon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
#     r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
#     r"(?P<eyear>\d{4})|(?P<present>present|current|now))",
#     flags=re.IGNORECASE,
# )

# SECTION_SPLIT_PATTERN = re.compile(
#     r"(?i)\b(PROJECTS|CERTIFICATES|SKILLS|ACHIEVEMENTS|EDUCATION)\b"
# )

# SKILL_ALIAS = {
#     "ml": "machine learning",
#     "ai": "machine learning",
#     "dbms": "sql",
#     "mysql": "sql",
#     "postgresql": "sql",
#     "ms excel": "excel",
# }

# GENERIC_SKILLS = {
#     "communication",
#     "coordination",
#     "reporting",
#     "documentation",
#     "teamwork",
#     "collaboration",
# }

# WORK_HINTS = {
#     "intern",
#     "internship",
#     "engineer",
#     "developer",
#     "analyst",
#     "experience",
#     "role",
#     "remote",
#     "full time",
#     "part time",
# }

# EDUCATION_HINTS = {
#     "education",
#     "btech",
#     "mtech",
#     "b.e",
#     "b.e.",
#     "bachelor",
#     "master",
#     "institute",
#     "college",
#     "school",
#     "certificate",
#     "secondary",
#     "higher secondary",
# }


# def _ym_to_index(year: int, month: int) -> int:
#     return year * 12 + month


# def _parse_month_year(month_str: str, year_str: str) -> Optional[Tuple[int, int]]:
#     month = MONTH_MAP.get(month_str.strip().lower())
#     if not month:
#         return None
#     try:
#         year = int(year_str)
#     except ValueError:
#         return None
#     return year, month


# def estimate_experience_years(resume_text: str) -> Optional[float]:
#     work_text = _extract_work_experience_text(resume_text)
#     target_text = work_text if work_text else resume_text

#     ranges: List[Tuple[int, int]] = []
#     now = datetime.datetime.now()
#     now_idx = _ym_to_index(now.year, now.month)

#     for match in DATE_RANGE_PATTERN.finditer(target_text):
#         if _should_skip_date_range(target_text, match.start(), match.end()):
#             continue
#         start = _parse_month_year(match.group("smon"), match.group("syear"))
#         if not start:
#             continue
#         start_idx = _ym_to_index(start[0], start[1])

#         if match.group("present"):
#             end_idx = now_idx
#         else:
#             end = _parse_month_year(match.group("emon"), match.group("eyear"))
#             if not end:
#                 continue
#             end_idx = _ym_to_index(end[0], end[1])

#         if end_idx < start_idx:
#             continue

#         ranges.append((start_idx, end_idx))

#     if not ranges:
#         return None

#     ranges.sort(key=lambda x: x[0])
#     merged = []
#     cur_start, cur_end = ranges[0]
#     for start, end in ranges[1:]:
#         if start <= cur_end + 1:
#             cur_end = max(cur_end, end)
#         else:
#             merged.append((cur_start, cur_end))
#             cur_start, cur_end = start, end
#     merged.append((cur_start, cur_end))

#     total_months = sum((end - start + 1) for start, end in merged)
#     return round(total_months / 12.0, 2)


# def _should_skip_date_range(text: str, start_idx: int, end_idx: int) -> bool:
#     # Use nearby context to drop education date ranges and keep work-like ranges.
#     left = max(0, start_idx - 140)
#     right = min(len(text), end_idx + 140)
#     context = text[left:right].lower()

#     has_work_hint = any(h in context for h in WORK_HINTS)
#     has_edu_hint = any(h in context for h in EDUCATION_HINTS)

#     if has_edu_hint and not has_work_hint:
#         return True
#     return False


# def _extract_work_experience_text(resume_text: str) -> str:
#     match = re.search(r"(?i)\bWORK EXPERIENCE\b", resume_text)
#     if not match:
#         return ""
#     start = match.end()
#     tail = resume_text[start:]
#     stop = SECTION_SPLIT_PATTERN.search(tail)
#     return tail[:stop.start()] if stop else tail


# def normalize_skills(skills_value) -> str:
#     if skills_value is None:
#         return ""
#     if isinstance(skills_value, list):
#         raw = [str(s).strip().lower() for s in skills_value]
#     else:
#         raw = [s.strip().lower() for s in str(skills_value).split(",")]

#     normalized = []
#     seen = set()
#     for skill in raw:
#         if not skill:
#             continue
#         skill = " ".join(skill.replace("-", " ").split())
#         skill = SKILL_ALIAS.get(skill, skill)
#         if skill in GENERIC_SKILLS:
#             continue
#         if skill not in seen:
#             seen.add(skill)
#             normalized.append(skill)

#     return ", ".join(normalized)


# def apply_postprocessing(output_json: dict, resume_text: str) -> dict:
#     if "error" in output_json:
#         return output_json

#     cleaned = dict(output_json)
#     cleaned["skills"] = normalize_skills(cleaned.get("skills", ""))

#     derived_years = estimate_experience_years(resume_text)
#     if derived_years is not None:
#         cleaned["experience_years"] = f"{derived_years:.2f}"
#         cleaned["experience_years_source"] = "timeline_parser"
#     else:
#         cleaned["experience_years_source"] = "llm_only"

#     return cleaned

# def get_multiline_input():
#     """Reads multiline input from user until they type 'END' or press Ctrl+D/Z."""
#     print("\n" + "="*60)
#     print("PASTE RESUME TEXT BELOW (Type 'END' on a new line to submit):")
#     print("="*60)
#     lines = []
#     while True:
#         try:
#             line = input()
#             if line.strip().upper() == 'END':
#                 break
#             lines.append(line)
#         except EOFError:
#             break
#     return "\n".join(lines)

# def main():
#     # 1. Setup LLM (Quietly)
#     llm_conf = config.get_llm_config()
#     if llm_conf["provider"] == "azure":
#         model_name = "azure/" + llm_conf["deployment"]
#         lm = dspy.LM(model_name, api_key=llm_conf["api_key"], api_base=llm_conf["endpoint"], api_version=llm_conf["api_version"])
#     else:
#         lm = dspy.Google(model=llm_conf["model"], api_key=llm_conf["api_key"])
#     dspy.settings.configure(lm=lm)

#     # 2. Load Models
#     print("\nLoading models...")
#     baseline = extractors.BaselineModule()
    
#     dspy_student = extractors.StudentModule()
#     try:
#         dspy_student.load("optimized_resume_module.json")
#         print("✅ DSPy Optimized Module Loaded.")
#     except Exception as e:
#         print(f"⚠️ Could not load optimized module: {e}")
#         print("Running unoptimized version for DSPy.")

#     # 3. Interactive Loop
#     while True:
#         resume_text = get_multiline_input()
#         if not resume_text.strip():
#             print("Exiting...")
#             break
            
#         print("\nProcessing... (Running Baseline & DSPy)")
        
#         # Run Baseline
#         try:
#             base_pred = baseline(unstructured_text=resume_text)
#             base_json = {
#                 "job_role": base_pred.job_role,
#                 "skills": base_pred.skills,
#                 "education": base_pred.education,
#                 "experience_years": base_pred.experience_years
#             }
#             base_json = apply_postprocessing(base_json, resume_text)
#         except Exception as e:
#             base_json = {"error": str(e)}

#         # Run DSPy
#         try:
#             dspy_pred = dspy_student(unstructured_text=resume_text)
#             dspy_json = {
#                 "job_role": dspy_pred.job_role,
#                 "skills": dspy_pred.skills,
#                 "education": dspy_pred.education,
#                 "experience_years": dspy_pred.experience_years
#             }
#             dspy_json = apply_postprocessing(dspy_json, resume_text)
#         except Exception as e:
#             dspy_json = {"error": str(e)}

#         # 4. Side-by-Side Output
#         print("\n" + "="*100)
#         print(f"{'BASELINE (Handcrafted)':<50} | {'DSPy (Optimized)':<50}")
#         print("="*100)
        
#         base_str = json.dumps(base_json, indent=2)
#         dspy_str = json.dumps(dspy_json, indent=2)
        
#         base_lines = base_str.split('\n')
#         dspy_lines = dspy_str.split('\n')
        
#         max_len = max(len(base_lines), len(dspy_lines))
        
#         for i in range(max_len):
#             b_line = base_lines[i] if i < len(base_lines) else ""
#             d_line = dspy_lines[i] if i < len(dspy_lines) else ""
#             print(f"{b_line:<50} | {d_line:<50}")
            
#         # 5. Save to JSON
#         output_entry = {
#             "timestamp": str(datetime.datetime.now()),
#             "input_text": resume_text,
#             "ground_truth": "N/A (User Input)",
#             "baseline": base_json,
#             "dspy": dspy_json
#         }
        
#         filename = "live_demo_results.json"
#         try:
#             try:
#                 with open(filename, 'r', encoding='utf-8') as f:
#                     history = json.load(f)
#                     if not isinstance(history, list): history = []
#             except (FileNotFoundError, json.JSONDecodeError):
#                 history = []
            
#             history.append(output_entry)
            
#             with open(filename, 'w', encoding='utf-8') as f:
#                 json.dump(history, f, indent=2, ensure_ascii=False)
#             print(f"\n✅ Results saved to '{filename}'")
#         except Exception as e:
#             print(f"\n⚠️ Could not save JSON: {e}")

#         print("="*100 + "\n")

# if __name__ == "__main__":
#     main()
