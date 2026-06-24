import gc
import os
import sys
from io import StringIO

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

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

@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the printed output. Use this to run Python code."""
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    try:
        exec(code, {})  # noqa: S102
        output = buffer.getvalue()
    except Exception as e:
        output = f"Error: {e}"
    finally:
        sys.stdout = old_stdout
    return output or "(no output)"


tools = [python_repl]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can execute Python code to solve tasks."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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

try:
    agent_executor.invoke({
        "input": (
            f"Sort these customers by last name and then first name "
            f"and print the output: {customer_list}"
        )
    })
finally:
    gc.collect()
    if hasattr(endpoint, 'client') and hasattr(endpoint.client, 'close'):
        endpoint.client.close()
