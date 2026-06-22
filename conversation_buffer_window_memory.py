import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from read_key import hf_provider, hf_token

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

# ConversationBufferWindowMemory keeps only the last k messages
# This prevents the conversation history from growing unbounded
class ConversationBufferWindowMemory:
    def __init__(self, k: int = 2):
        self.k = k
        self.history: list[BaseMessage] = []
    
    def add_message(self, message: BaseMessage):
        """Add a message and enforce the window size"""
        self.history.append(message)
        # Keep only the last k messages
        if len(self.history) > self.k:
            self.history = self.history[-self.k:]
    
    def get_history(self) -> list[BaseMessage]:
        """Get current history"""
        return self.history
    
    def buffer_string(self) -> str:
        """Format history as a readable string"""
        return "\n".join(f"{m.type.capitalize()}: {m.content}" for m in self.history)

memory = ConversationBufferWindowMemory(k=2)  # Keep only last 2 messages (1 exchange)

def chat(user_input: str) -> str:
    messages = prompt.format_messages(input=user_input, history=memory.get_history())
    print("\n> Entering new ConversationChain chain...")
    print("Prompt after formatting:")
    for m in messages:
        print(f"{m.type.capitalize()}: {m.content}")
    response = chain.invoke({"input": user_input, "history": memory.get_history()})
    print(f"AI: {response.content}")
    print("\n> Finished chain.")
    
    # Add messages to memory (they'll be pruned to window size)
    memory.add_message(HumanMessage(content=user_input))
    memory.add_message(AIMessage(content=response.content))
    print(f"\n[Memory State (k=2, max 2 messages)]:")
    print(memory.buffer_string())
    
    return response.content

print("=" * 70)
print("ConversationBufferWindowMemory Demo (k=2)")
print("Keeps only the last 2 messages (1 human-AI exchange)")
print("=" * 70)

chat("Hi, my name is Karim")
chat("What is 1+1?")
print("\n>>> Turn 3: Testing memory (should not remember my name)")
chat("What is my name?")
print("\nNote: The first exchange is no longer in memory due to window size k=2")
