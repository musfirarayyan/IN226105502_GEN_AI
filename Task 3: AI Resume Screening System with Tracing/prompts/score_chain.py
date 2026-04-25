from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from prompts.score_prompt import score_prompt

pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

chain = score_prompt | llm | StrOutputParser()

score_chain = chain
