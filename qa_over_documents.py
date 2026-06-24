import os

from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint

from read_key import hf_provider, hf_token

llm_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
embed_model = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

endpoint = HuggingFaceEndpoint(
    repo_id=llm_model,
    provider=hf_provider,
    huggingfacehub_api_token=hf_token,
    task="conversational",
    temperature=0,
    max_new_tokens=512,
)

llm = ChatHuggingFace(llm=endpoint)

# Local sentence-transformers embeddings — no API call required
embeddings = HuggingFaceEmbeddings(model_name=embed_model)

# Load documents
file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "outdoor_clothing.csv")
loader = CSVLoader(file_path=file)

# Create in-memory vector index from the CSV loader
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])

# Query
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = index.query(query, llm=llm)
print(response)
