import gc
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from datetime import date

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
# Custom tool: return today's date
# ---------------------------------------------------------------------------

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())


tools = [time]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use available tools when needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print("=" * 70)
print("CUSTOM TOOL AGENT — Ask about today's date")
print("=" * 70)

try:
    result = agent_executor.invoke({"input": "whats the date today?"})
    print(result)
except Exception:
    print("exception on external access")
finally:
    gc.collect()
    if hasattr(endpoint, 'client') and hasattr(endpoint.client, 'close'):
        endpoint.client.close()
