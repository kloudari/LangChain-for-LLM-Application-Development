import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.chains import ConversationChain  # type: ignore
from langchain_classic.memory import ConversationBufferMemory  # type: ignore

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

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=chat_model, memory=memory, verbose=True)

conversation.predict(input="Hi, my name is Karim")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")

print(memory.buffer)
