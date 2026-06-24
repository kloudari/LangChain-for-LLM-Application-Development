import os

import csv

from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


class CSVLoader:
    """Minimal CSV loader compatible with VectorstoreIndexCreator."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        docs = []
        with open(self.file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                content = "\n".join(f"{k}: {v}" for k, v in row.items())
                docs.append(Document(page_content=content, metadata={"row": i, "source": self.file_path}))
        return docs
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
    vectorstore_cls=InMemoryVectorStore,
    embedding=embeddings,
).from_loaders([loader])

# Query
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = index.query(query, llm=llm, chain_type="stuff")
print(response)
