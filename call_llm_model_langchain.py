import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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


def get_completion(prompt, model=llm_model):
    if model != llm_model:
        llm = HuggingFaceEndpoint(
            repo_id=model,
            provider=hf_provider,
            huggingfacehub_api_token=hf_token,
            task="conversational",
            temperature=0,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=llm).invoke(prompt).content

    return chat_model.invoke(prompt).content


customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

template_string = """Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

chain = prompt_template | chat_model | StrOutputParser()


if __name__ == "__main__":
    try:
        print(prompt_template.messages[0].prompt)
        print(prompt_template.messages[0].prompt.input_variables)
        print(chain.invoke({"style": style, "customer_email": customer_email}))
    except Exception as exc:
        print(f"API call failed: {exc}")


customer_style = """American English \
in a calm and respectful tone
"""

customer_messages = chain.invoke({"style": customer_style, "customer_email": customer_email})
print(type(customer_messages))
print(type(customer_messages[0]))
print(customer_messages[0])

# Call the LLM to translate to the style of the customer message
customer_response = chat_model.invoke(customer_messages)

print(customer_response.content)

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

service_messages = chain.invoke({"style": service_style_pirate, "customer_email": service_reply})
print(service_messages[0])

service_response = chat_model.invoke(service_messages)
print(service_response.content)
