import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from read_key import hf_provider, hf_token


# ConversationSummaryMemory keeps a running LLM-generated summary of the
# conversation so far. Instead of storing raw messages, after each exchange
# it asks the LLM to update the summary with the latest turn. The summary
# is then injected into the system prompt as context.
class ConversationSummaryMemory:
    def __init__(self, llm):
        self.llm = llm
        self.summary: str = ""

    def save_context(self, human_msg: str, ai_msg: str):
        """Update the running summary with the latest human-AI exchange."""
        update_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Progressively summarize the lines of conversation provided, "
                "adding onto the previous summary returning a new summary.\n\n"
                "EXAMPLE\n"
                "Current summary:\nThe human asks what the AI thinks of artificial intelligence. "
                "The AI thinks artificial intelligence is a force for good.\n\n"
                "New lines of conversation:\n"
                "Human: Why do you think artificial intelligence is a force for good?\n"
                "AI: Because artificial intelligence will help humans reach their full potential.\n\n"
                "New summary:\nThe human asks what the AI thinks of artificial intelligence. "
                "The AI thinks artificial intelligence is a force for good because it will help "
                "humans reach their full potential.\nEND OF EXAMPLE\n\n"
                "Current summary:\n{summary}\n\n"
                "New lines of conversation:\n"
                "Human: {human_msg}\n"
                "AI: {ai_msg}\n\n"
                "New summary:"
            )),
        ])
        messages = update_prompt.format_messages(
            summary=self.summary,
            human_msg=human_msg,
            ai_msg=ai_msg,
        )
        response = self.llm.invoke(messages)
        self.summary = response.content.strip()

    def get_summary(self) -> str:
        return self.summary


llm_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

endpoint = HuggingFaceEndpoint(
    repo_id=llm_model,
    provider=hf_provider,
    huggingfacehub_api_token=hf_token,
    task="conversational",
    temperature=0,
    max_new_tokens=100,
)

chat_model = ChatHuggingFace(llm=endpoint)

memory = ConversationSummaryMemory(llm=chat_model)


def chat(user_input: str) -> str:
    summary = memory.get_summary()
    system_content = (
        "The following is a friendly conversation between a human and an AI. "
        "The AI is talkative and provides lots of specific details from its context. "
        "If the AI does not know the answer to a question, it truthfully says it does not know.\n\n"
        "Current conversation summary:\n" + summary
    )
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ]
    print("\n> Entering new ConversationChain chain...")
    print("Prompt after formatting:")
    for m in messages:
        print(f"{m.type.capitalize()}: {m.content}")

    response = chat_model.invoke(messages)
    print(f"AI: {response.content}")
    print("\n> Finished chain.")

    memory.save_context(user_input, response.content)
    print(f"\n[Memory State (summary)]:\n{memory.get_summary()}")

    return response.content


print("=" * 70)
print("ConversationSummaryMemory Demo")
print("Keeps a running LLM-generated summary instead of raw message history")
print("=" * 70)

chat("Hi, my name is Karim and I work in machine learning.")
chat("What are interesting topics in machine learning?")
print("\n>>> Turn 3: Testing memory via summary")
chat("What is my name and what do I work on?")
