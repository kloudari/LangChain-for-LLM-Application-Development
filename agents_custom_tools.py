import gc
import math
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
# Tool 1: return today's date
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


# ---------------------------------------------------------------------------
# Tool 2: safe math expression evaluator
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.
    Supports standard arithmetic and math functions (sqrt, pow, sin, cos, etc.).
    Example input: '2 ** 10' or 'sqrt(144)'."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))  # noqa: S307
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ---------------------------------------------------------------------------
# Tool 3: word and character counter
# ---------------------------------------------------------------------------

@tool
def text_stats(text: str) -> str:
    """Returns word count and character count for the given text.
    Input should be the text to analyse."""
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    return (
        f"Words: {words}, "
        f"Characters (with spaces): {chars}, "
        f"Characters (without spaces): {chars_no_spaces}"
    )


# ---------------------------------------------------------------------------
# Tool 4: Celsius ↔ Fahrenheit converter
# ---------------------------------------------------------------------------

@tool
def celsius_to_fahrenheit(celsius: str) -> str:
    """Convert a temperature from Celsius to Fahrenheit.
    Input should be a numeric string, e.g. '100'."""
    try:
        c = float(celsius)
        f = c * 9 / 5 + 32
        return f"{c}°C = {f}°F"
    except ValueError:
        return "Invalid input: please provide a numeric value."


@tool
def fahrenheit_to_celsius(fahrenheit: str) -> str:
    """Convert a temperature from Fahrenheit to Celsius.
    Input should be a numeric string, e.g. '212'."""
    try:
        f = float(fahrenheit)
        c = (f - 32) * 5 / 9
        return f"{f}°F = {round(c, 2)}°C"
    except ValueError:
        return "Invalid input: please provide a numeric value."


# ---------------------------------------------------------------------------
# Tool 5: string reverser
# ---------------------------------------------------------------------------

@tool
def reverse_string(text: str) -> str:
    """Reverse the characters in a string.
    Input should be the string to reverse."""
    return text[::-1]


# ---------------------------------------------------------------------------
# Agent setup — all custom tools
# ---------------------------------------------------------------------------

tools = [time, calculator, text_stats, celsius_to_fahrenheit, fahrenheit_to_celsius, reverse_string]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use available tools when needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def run(question: str) -> None:
    print("-" * 70)
    print(f"Q: {question}")
    try:
        result = agent_executor.invoke({"input": question})
        print("A:", result.get("output", result))
    except Exception:
        print("exception on external access")


print("=" * 70)
print("CUSTOM TOOLS AGENT")
print("=" * 70)

run("whats the date today?")
run("What is the square root of 256 plus 10 percent of 500?")
run("How many words are in this sentence: The quick brown fox jumps over the lazy dog")
run("Convert 37 degrees Celsius to Fahrenheit.")
run("Convert 98.6 Fahrenheit to Celsius.")
run("Reverse the string 'LangChain'.")

gc.collect()
if hasattr(endpoint, 'client') and hasattr(endpoint.client, 'close'):
    endpoint.client.close()
