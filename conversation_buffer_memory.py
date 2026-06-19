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

history: list[BaseMessage] = []

def chat(user_input: str) -> str:
    messages = prompt.format_messages(input=user_input, history=history)
    print("\n> Entering new ConversationChain chain...")
    print("Prompt after formatting:")
    for m in messages:
        print(f"{m.type.capitalize()}: {m.content}")
    response = chain.invoke({"input": user_input, "history": history})
    print(f"AI: {response.content}")
    print("\n> Finished chain.")
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))
    return response.content

chat("Hi, my name is Karim")
chat("What is 1+1?")
chat("What is my name?")

print("\n".join(f"{m.type}: {m.content}" for m in history))


print({"history": history})

demo_history: list[BaseMessage] = []
demo_history.append(HumanMessage(content="Hi"))
demo_history.append(AIMessage(content="What's up"))
print("\n".join(f"{m.type}: {m.content}" for m in demo_history))

print({"history": demo_history})

demo_history.append(HumanMessage(content="Not much, just hanging"))
demo_history.append(AIMessage(content="Cool"))
print({"history": demo_history})
