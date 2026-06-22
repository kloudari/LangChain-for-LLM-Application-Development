import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from read_key import hf_provider, hf_token


# ConversationTokenBufferMemory keeps messages up to a token budget.
# Oldest messages are dropped (in human-AI pairs) when the limit is exceeded.
# Token count is approximated as word count (1 word ≈ 1 token).
class ConversationTokenBufferMemory:
    def __init__(self, max_token_limit: int = 80):
        self.max_token_limit = max_token_limit
        self.history: list[BaseMessage] = []

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def _total_tokens(self) -> int:
        return sum(self._count_tokens(m.content) for m in self.history)

    def save_context(self, human_msg: str, ai_msg: str):
        """Add a human-AI exchange and prune from the front if over budget."""
        self.history.append(HumanMessage(content=human_msg))
        self.history.append(AIMessage(content=ai_msg))
        # Remove oldest human-AI pairs until within the token limit
        while self._total_tokens() > self.max_token_limit and len(self.history) >= 2:
            self.history = self.history[2:]

    def get_history(self) -> list[BaseMessage]:
        return self.history

    def buffer_string(self) -> str:
        tokens = self._total_tokens()
        lines = [f"{m.type.capitalize()}: {m.content}" for m in self.history]
        return "\n".join(lines) + f"\n[~{tokens} tokens in buffer]"

llm_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

endpoint = HuggingFaceEndpoint(
    repo_id=llm_model,
    provider=hf_provider,
    huggingfacehub_api_token=hf_token,
    task="conversational",
    temperature=0,
    max_new_tokens=512,
)

chat_model = ChatHuggingFace(llm=endpoint)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "The following is a friendly conversation between a human and an AI. "
        "The AI is talkative and provides lots of specific details from its context. "
        "If the AI does not know the answer to a question, it truthfully says it does not know."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | chat_model

memory = ConversationTokenBufferMemory(max_token_limit=80)

def chat(user_input: str) -> str:
    history = memory.get_history()
    messages = prompt.format_messages(input=user_input, history=history)
    print("\n> Entering new ConversationChain chain...")
    print("Prompt after formatting:")
    for m in messages:
        print(f"{m.type.capitalize()}: {m.content}")
    response = chain.invoke({"input": user_input, "history": history})
    print(f"AI: {response.content}")
    print("\n> Finished chain.")

    # Save context; memory auto-prunes oldest pairs when over the token budget.
    memory.save_context(user_input, response.content)
    print(f"\n[Memory State (token-limited, max_token_limit=80)]:")
    print(memory.buffer_string())
    
    return response.content

print("=" * 70)
print("ConversationTokenBufferMemory Demo")
print("Keeps as much history as fits in max_token_limit tokens")
print("=" * 70)

chat("Hi, my name is Karim")
chat("What is 1+1?")
print("\n>>> Turn 3: Testing memory")
chat("What is my name?")
print("\nNote: Old messages are pruned automatically when the token limit is exceeded.")
