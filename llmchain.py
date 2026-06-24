import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from read_key import hf_provider, hf_token

llm_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

endpoint = HuggingFaceEndpoint(
    repo_id=llm_model,
    provider=hf_provider,
    huggingfacehub_api_token=hf_token,
    task="conversational",
    temperature=0.9,
    max_new_tokens=100,
)

llm = ChatHuggingFace(llm=endpoint)

prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe "
    "a company that makes {product}?"
)

chain = prompt | llm | StrOutputParser()

product = "Queen Size Sheet Set"
result = chain.invoke({"product": product})
print(result)
