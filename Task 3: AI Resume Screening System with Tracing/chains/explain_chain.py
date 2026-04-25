from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from prompts.explain_prompt import explain_prompt


pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

chain = explain_prompt | llm | StrOutputParser()

explain_chain = chain
