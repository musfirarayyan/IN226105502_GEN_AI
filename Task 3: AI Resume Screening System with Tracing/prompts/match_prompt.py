from langchain_core.prompts import PromptTemplate

match_prompt = PromptTemplate.from_template("""
Compare resume data with job description.

STRICT RULES:
- Only compare skills
- Case insensitive matching
- Do NOT return empty if match exists

Resume Skills:
{skills}

Job Description:
{job_description}

Extract skills from job description and compare.

Return ONLY JSON:
{{
  "matched_skills": [...],
  "missing_skills": [...]
}}
""")