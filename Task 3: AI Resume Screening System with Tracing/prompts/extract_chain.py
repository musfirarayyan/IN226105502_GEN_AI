from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from prompts.extract_prompt import extract_prompt

pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=256
)


llm = HuggingFacePipeline(pipeline=pipe)

chain = extract_prompt | llm | StrOutputParser()

extract_chain = chain