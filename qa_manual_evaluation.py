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

# ---------------------------------------------------------------------------
# Hard-coded evaluation examples: (query, expected answer)
# ---------------------------------------------------------------------------
examples = [
    {
        "query": "Do the SunShield Hiking Shirt offer sun protection?",
        "expected": "Yes, the SunShield Hiking Shirt offers UPF 50+ sun protection.",
    },
    {
        "query": "What is the fill power of the DownDrift Insulated Jacket?",
        "expected": "The DownDrift Insulated Jacket has 700-fill-power down.",
    },
    {
        "query": "Which product is recommended for wet mountain conditions?",
        "expected": "The ArcticShell Waterproof Jacket is ideal for wet mountain conditions.",
    },
    {
        "query": "Can the SummitTrek Hiking Pants be converted to shorts?",
        "expected": "Yes, the SummitTrek Hiking Pants have zip-off lower legs that convert them to shorts.",
    },
    {
        "query": "What type of wool is used in the AlpineBase Merino Top?",
        "expected": "The AlpineBase Merino Top uses 18.5-micron merino wool.",
    },
]

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
print("=" * 70)
print("MANUAL EVALUATION")
print("=" * 70)

correct = 0
for i, example in enumerate(examples, start=1):
    query = example["query"]
    expected = example["expected"]

    predicted = index.query(query, llm=llm, chain_type="stuff")

    # Simple keyword-based correctness check
    key_words = [w for w in expected.lower().split() if len(w) > 4]
    matches = sum(1 for w in key_words if w in predicted.lower())
    is_correct = matches >= max(1, len(key_words) // 2)

    if is_correct:
        correct += 1

    print(f"\nExample {i}")
    print(f"  Query    : {query}")
    print(f"  Expected : {expected}")
    print(f"  Predicted: {predicted.strip()}")
    print(f"  Result   : {'CORRECT' if is_correct else 'INCORRECT'}")

print("\n" + "=" * 70)
print(f"Score: {correct}/{len(examples)} ({100 * correct // len(examples)}%)")
print("=" * 70)
