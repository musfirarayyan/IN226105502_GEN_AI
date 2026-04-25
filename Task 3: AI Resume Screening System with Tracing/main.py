import os
import json
import re
from dotenv import load_dotenv

from chains.extract_chain import extract_chain

load_dotenv()


# -------- SAFE JSON PARSER --------
def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        try:
            matches = re.findall(r'\{.*?\}', str(text), re.DOTALL)
            for m in matches:
                try:
                    return json.loads(m)
                except:
                    continue
            return {}
        except:
            return {}


# -------- SIMPLE MATCH FUNCTION --------
def simple_match(resume_skills, jd_text):
    jd_skills = ["Python", "Machine Learning", "Deep Learning", "NLP", "SQL", "TensorFlow"]

    matched = []
    missing = []

    for skill in jd_skills:
        if skill.lower() in [s.lower() for s in resume_skills]:
            matched.append(skill)
        else:
            missing.append(skill)

    return {
        "matched_skills": matched,
        "missing_skills": missing
    }


# -------- SIMPLE SCORE FUNCTION --------
def simple_score(match_data):
    total = len(match_data["matched_skills"]) + len(match_data["missing_skills"])

    if total == 0:
        return {"score": 0}

    score = int((len(match_data["matched_skills"]) / total) * 100)

    return {"score": score}


# -------- SIMPLE EXPLANATION FUNCTION --------
def simple_explain(score, match_data):
    return f"""
Candidate Score: {score['score']}%

Matched Skills: {', '.join(match_data['matched_skills'])}
Missing Skills: {', '.join(match_data['missing_skills'])}

More matched skills increase the score, while missing skills reduce it.
"""


# -------- MAIN PROCESS FUNCTION --------
def process_resume(resume_path, jd_path):

    # Read files
    with open(resume_path, "r", encoding="utf-8") as f:
        resume = f.read()

    with open(jd_path, "r", encoding="utf-8") as f:
        jd = f.read()

    print("\n--- RESUME CONTENT ---")
    print(resume)

    # ================= STEP 1 =================
    print("\n--- STEP 1: Extraction ---")

    extracted = extract_chain.invoke({"resume": resume})
    print("RAW OUTPUT:\n", extracted)

    extracted = safe_json_parse(extracted)
    print("PARSED OUTPUT:\n", extracted)

    if not extracted:
        extracted = {"skills": [], "experience": "", "tools": []}

    # ================= STEP 2 =================
    print("\n--- STEP 2: Matching ---")

    match_result = simple_match(extracted.get("skills", []), jd)
    print(match_result)

    # ================= STEP 3 =================
    print("\n--- STEP 3: Scoring ---")

    score = simple_score(match_result)
    print(score)

    # ================= STEP 4 =================
    print("\n--- STEP 4: Explanation ---")

    explanation = simple_explain(score, match_result)
    print(explanation)


# ================= MAIN =================
if __name__ == "__main__":
    process_resume("data/resume_strong.txt", "data/job_description.txt")