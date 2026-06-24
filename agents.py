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
    max_new_tokens=30,
)

llm = ChatHuggingFace(llm=endpoint)

# ---------------------------------------------------------------------------
# Calculator tool — replaces llm-math, no external packages needed
# ---------------------------------------------------------------------------
from langchain_core.tools import tool
import math

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    return str(eval(expression, {"__builtins__": {}}, allowed))  # noqa: S307

# Wikipedia tool
import wikipedia as _wiki_module
_wiki_module.set_lang("en")
_wiki_module.set_user_agent("LangChain-course-bot/1.0 (educational use)")

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# New LangGraph-based agent API (langchain 1.x)
from langchain.agents import create_agent

agent = create_agent(llm, [calculator, wikipedia])

print("=" * 70)
print("AGENT — Math question")
print("=" * 70)
result = agent.invoke({"messages": [("user", "What is the 25% of 300?")]})
print(result["messages"][-1].content)

print("=" * 70)
print("AGENT — Wikipedia question")
print("=" * 70)
result = agent.invoke({
    "messages": [(
        "user",
        "Tom M. Mitchell is an American computer scientist "
        "and the Founders University Professor at Carnegie Mellon University (CMU)"
        "what book did he write?"
    )]
})
print(result["messages"][-1].content)



