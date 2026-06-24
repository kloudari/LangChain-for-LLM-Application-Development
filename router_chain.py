import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from read_key import hf_provider, hf_token

llm_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

endpoint = HuggingFaceEndpoint(
    repo_id=llm_model,
    provider=hf_provider,
    huggingfacehub_api_token=hf_token,
    task="conversational",
    temperature=0,
    max_new_tokens=256,
)

llm = ChatHuggingFace(llm=endpoint)

# ---------------------------------------------------------------------------
# Subject-specific prompt templates
# ---------------------------------------------------------------------------

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise \
and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{question}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down hard problems into their \
component parts, answer the component parts, and then put them together \
to answer the broader question.

Here is a question:
{question}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people, events, and \
contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and evaluate the past. \
You have a respect for historical evidence and the ability to make use of it \
to support your explanations and judgements.

Here is a question:
{question}"""

computerscience_template = """You are a successful computer scientist. \
You have a passion for creativity, collaboration, forward-thinking, confidence, \
strong problem-solving capabilities, understanding of theories and algorithms, \
and excellent communication skills. \
You are great at answering coding questions because you know how to solve a \
problem by describing the solution in imperative steps that a computer can \
easily interpret, while choosing solutions with a good balance between time \
and space complexity.

Here is a question:
{question}"""

# ---------------------------------------------------------------------------
# Build subject-specific chains
# ---------------------------------------------------------------------------

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "template": math_template,
    },
    {
        "name": "history",
        "description": "Good for answering history questions",
        "template": history_template,
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science and programming questions",
        "template": computerscience_template,
    },
]

subject_chains = {
    info["name"]: ChatPromptTemplate.from_template(info["template"]) | llm | StrOutputParser()
    for info in prompt_infos
}

# Default chain used when the input doesn't match any known subject
default_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question to the best of your ability.\n\n{question}"
)
default_chain = default_prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Classifier chain — determines which subject the question belongs to
# ---------------------------------------------------------------------------

subjects_description = "\n".join(
    f"- {info['name']}: {info['description']}" for info in prompt_infos
)

classifier_prompt = ChatPromptTemplate.from_template(
    "Given a user question, classify it into exactly one of the following subjects. "
    "Reply with ONLY the subject name from the list, nothing else.\n\n"
    "Subjects:\n{subjects}\n\n"
    "Question: {question}"
)

classifier_chain = (
    (lambda x: {"subjects": subjects_description, "question": x["question"]})
    | classifier_prompt
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route(inputs: dict) -> str:
    subject = inputs["subject"].strip().lower()
    for name, chain in subject_chains.items():
        if name in subject:
            print(f"[Router] → {name}")
            return chain.invoke({"question": inputs["question"]})
    print(f"[Router] → no match for '{inputs['subject']}', using default chain")
    return default_chain.invoke({"question": inputs["question"]})


# Full router chain:
#   1. Run classifier and pass the question through in parallel
#   2. Route to the appropriate sub-chain based on the classified subject
router_chain = (
    RunnableParallel(
        subject=classifier_chain,
        question=RunnablePassthrough() | (lambda x: x["question"]),
    )
    | RunnableLambda(route)
)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

questions = [
    "What is black body radiation?",
    "What is 2 + 2?",
    "Why does every cell in our body contain DNA?",
    "Who was the first president of the United States?",
    "What is the difference between a list and a tuple in Python?",
]

for question in questions:
    print(f"\nQ: {question}")
    answer = router_chain.invoke({"question": question})
    print(f"A: {answer}")
