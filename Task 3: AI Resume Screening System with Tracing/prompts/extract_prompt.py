from langchain_core.prompts import PromptTemplate

extract_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
Extract information from the resume.

Return ONLY valid JSON.

Resume:
{resume}

Output:
{{
  "skills": ["Python", "Machine Learning"],
  "experience": "3 years",
  "tools": ["TensorFlow", "SQL"]
}}
"""
)
