import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from read_key import hf_provider, hf_token

llm_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

endpoint = HuggingFaceEndpoint(
    repo_id=llm_model,
    provider=hf_provider,
    huggingfacehub_api_token=hf_token,
    task="conversational",
    temperature=0.9,
    max_new_tokens=150,
)

llm = ChatHuggingFace(llm=endpoint)

# Chain 1: translate the review to English
prompt_1 = ChatPromptTemplate.from_template(
    "Translate the following review to English:\n\n{review}"
)
chain_1 = prompt_1 | llm | StrOutputParser()

# Chain 2: summarize the English review in 1 sentence
prompt_2 = ChatPromptTemplate.from_template(
    "Summarize the following review in 1 sentence:\n\n{english_review}"
)
chain_2 = prompt_2 | llm | StrOutputParser()

# Chain 3: detect the language of the original review
prompt_3 = ChatPromptTemplate.from_template(
    "What language is the following review written in? "
    "Reply with only the language name, nothing else.\n\n{review}"
)
chain_3 = prompt_3 | llm | StrOutputParser()

# Chain 4: write a follow-up reply in the detected language
prompt_4 = ChatPromptTemplate.from_template(
    "You are a customer service agent. Write a short, polite reply to the customer "
    "based on the following review summary. Write your reply in {language}. "
    "Reply with only the response message, nothing else.\n\nReview summary: {summary}"
)
chain_4 = prompt_4 | llm | StrOutputParser()

# Build the full sequential chain using LCEL
# Step 1: translate review to English, while passing the original review through
step_1 = RunnableParallel(
    english_review=chain_1,
    review=RunnablePassthrough() | (lambda x: x["review"]),
)

# Step 2: summarize + detect language in parallel, pass english_review through
step_2 = RunnableParallel(
    summary=chain_2,
    language=chain_3 | (lambda lang: lang),
    english_review=lambda x: x["english_review"],
).with_config(
    run_name="summarize_and_detect"
)

# Wire step_2 inputs: chain_2 needs english_review, chain_3 needs review
step_2 = RunnableParallel(
    summary=(lambda x: {"english_review": x["english_review"]}) | chain_2,
    language=(lambda x: {"review": x["review"]}) | chain_3,
    english_review=lambda x: x["english_review"],
)

# Step 3: write a follow-up reply
step_3 = (lambda x: {"summary": x["summary"], "language": x["language"]}) | chain_4

# Full pipeline
chain = step_1 | step_2 | (
    lambda x: {
        "english_review": x["english_review"],
        "summary": x["summary"],
        "language": x["language"],
        "followup_message": step_3.invoke(x),
    }
)

# --- Run ---
reviews = [
    (
        "Je trouve que ce produit est vraiment excellent. "
        "La qualité est au rendez-vous et la livraison a été très rapide. "
        "Je le recommande vivement à tous !"
    ),
    (
        "Ce produit est une véritable déception. "
        "La qualité est médiocre, il s'est cassé après seulement deux jours d'utilisation. "
        "Le service client n'a pas répondu à mes messages. Je ne recommande absolument pas cet achat."
    ),
]

for i, review in enumerate(reviews, start=1):
    print(f"=== Review {i} ===")
    result = chain.invoke({"review": review})
    print("Original review :", review)
    print()
    print("English review  :", result["english_review"])
    print()
    print("Summary         :", result["summary"])
    print()
    print("Language        :", result["language"])
    print()
    print("Follow-up       :", result["followup_message"])
    print()
