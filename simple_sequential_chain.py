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

# Chain 1: generate a company name from a product
prompt_1 = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}? "
    "Reply with only the company name, nothing else."
)

chain_1 = prompt_1 | llm | StrOutputParser()

# Chain 2: generate a 20-word description for the company name
prompt_2 = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company: {company_name}"
)

chain_2 = prompt_2 | llm | StrOutputParser()

product = "Queen Size Sheet Set"

company_name = chain_1.invoke({"product": product})
print(company_name)

description = chain_2.invoke({"company_name": company_name})
print(description)
