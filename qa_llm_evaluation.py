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
# LLM-generated evaluation examples via QAGenerateChain
# QAEvalChain expects keys "query" and "answer"
# ---------------------------------------------------------------------------
from langchain_classic.evaluation.qa import QAGenerateChain

docs = loader.load()
generate_chain = QAGenerateChain.from_llm(llm)

# Generate one QA pair per document (use a subset to keep runtime reasonable)
sample_docs = docs[:5]
generated = generate_chain.apply([{"doc": d} for d in sample_docs])

examples = [
    {"query": g["qa_pairs"]["query"], "answer": g["qa_pairs"]["answer"]}
    for g in generated
]

# ---------------------------------------------------------------------------
# Generate predictions from the QA chain
# ---------------------------------------------------------------------------
predictions = []
for example in examples:
    result = index.query(example["query"], llm=llm, chain_type="stuff")
    predictions.append({"result": result})

# ---------------------------------------------------------------------------
# LLM-assisted evaluation with QAEvalChain
# ---------------------------------------------------------------------------
from langchain_classic.evaluation.qa import QAEvalChain

eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)

print("=" * 70)
print("LLM-ASSISTED EVALUATION")
print("=" * 70)

correct = 0
for i, (example, prediction, grade) in enumerate(
    zip(examples, predictions, graded_outputs), start=1
):
    result_text = grade.get("results", grade.get("text", "")).strip().upper()
    is_correct = "CORRECT" in result_text and "INCORRECT" not in result_text

    if is_correct:
        correct += 1

    print(f"\nExample {i}")
    print(f"  Query    : {example['query']}")
    print(f"  Answer : {example['answer']}")
    print(f"  Predicted: {prediction['result'].strip()}")
    print(f"  LLM Grade: {result_text}")

print("\n" + "=" * 70)
print(f"Score: {correct}/{len(examples)} ({100 * correct // len(examples)}%)")
print("=" * 70)
