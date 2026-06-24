import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from read_key import hf_provider, hf_token

llm_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

endpoint = HuggingFaceEndpoint(
    repo_id=llm_model,
    provider=hf_provider,
    huggingfacehub_api_token=hf_token,
    task="conversational",
    temperature=0,
    max_new_tokens=512,
)

llm = ChatHuggingFace(llm=endpoint)

# ---------------------------------------------------------------------------
# Python Agent
# ---------------------------------------------------------------------------
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool

agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True,
)

customer_list = [
    ["Harrison", "Chase"],
    ["Lang", "Chain"],
    ["Dolly", "Too"],
    ["Elle", "Elem"],
    ["Geoff", "Fusion"],
    ["Trance", "Former"],
    ["Jen", "Ayai"],
]

print("=" * 70)
print("PYTHON AGENT — Sort customers by last name then first name")
print("=" * 70)

agent.run(
    f"Sort these customers by last name and then first name "
    f"and print the output: {customer_list}"
)
