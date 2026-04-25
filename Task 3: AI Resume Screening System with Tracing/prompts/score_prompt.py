from langchain_core.prompts import PromptTemplate

score_prompt = PromptTemplate(
    input_variables=["match_data"],
    template="""
Based on matching data, assign a score (0-100).

Rules:
- More matched skills → higher score
- More missing skills → lower score

Match Data:
{match_data}

Output:
{{
  "score": number
}}
"""
)