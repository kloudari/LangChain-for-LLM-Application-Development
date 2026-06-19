import os
from read_key import client

llm_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

print(get_completion("What is 1+1?"))

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

prompt = f"""Translate (don't answer) the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(get_completion(prompt))

