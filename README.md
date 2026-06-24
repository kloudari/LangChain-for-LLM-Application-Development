# LangChain for LLM Application Development

Based on : https://www.deeplearning.ai/courses/langchain
A learning project exploring how to use LangChain with Large Language Models (LLMs) for application development, comparing traditional API approaches with LangChain's abstractions.

## Project Overview

This project is structured by course chapters. Each chapter covers a specific LangChain concept, with scripts adapted to use Hugging Face instead of the original OpenAI backend.

## Chapters

### Chapter 1 — Models, Prompts and Parsers

Introduces LangChain's core building blocks: prompt templates, chat models, output parsers, and chains.

| File | Description |
|------|-------------|
| `call_llm_model_classic.py` | Direct LLM calls using the Hugging Face `InferenceClient` (no LangChain) |
| `call_llm_model_langchain.py` | Same use case rewritten with `ChatPromptTemplate`, `ChatHuggingFace`, and `StrOutputParser` |
| `extract_review_info.py` | Structured extraction from a product review using `JsonOutputParser` |

### Chapter 2 — Memory

Explores different memory types to manage conversational context in multi-turn interactions, comparing memory retention strategies:

| File | Description |
|------|-------------|
| `conversation_buffer_memory.py` | **ConversationBufferMemory** — Retains the full conversation history; simple but can grow unbounded |
| `conversation_buffer_window_memory.py` | **ConversationBufferWindowMemory** — Keeps only the last N messages; prevents unbounded growth |
| `conversation_buffer_token_memory.py` | **ConversationTokenBufferMemory** — Keeps messages up to a token limit; manages memory by token count |
| `conversation_summary_memory.py` | **ConversationSummaryMemory** — Summarizes old messages; condenses history while preserving key information |

### Chapter 3 — Chains

Demonstrates how to compose multiple LLM calls into pipelines using LangChain's chain abstractions:

| File | Description |
|------|-------------|
| `llmchain.py` | **LLMChain** — Single prompt + LLM + parser wired as a LCEL chain |
| `simple_sequential_chain.py` | **SimpleSequentialChain** — Two chains in sequence; output of one feeds into the next (single input/output) |
| `sequential_chain.py` | **SequentialChain** — Multi-step pipeline with named inputs and outputs across steps |
| `router_chain.py` | **Router Chain** — Dynamically routes input to the most appropriate sub-chain based on content |

### Chapter 4 — Q&A over Documents

Demonstrates Retrieval-Augmented Generation (RAG): loading a product catalog, embedding it into a vector store, and answering natural-language questions grounded in that data.

| File | Description |
|------|-------------|
| `qa_over_documents.py` | **RAG pipeline** — Loads CSV documents, indexes them in an in-memory vector store, and answers natural-language questions using `VectorstoreIndexCreator` |
| `data/outdoor_clothing.csv` | Sample outdoor product catalog used as the document corpus |

### Chapter 5 — Evaluation

Covers how to assess the quality of RAG pipelines by comparing model answers against expected ground-truth answers.

| File | Description |
|------|-------------|
| `qa_manual_evaluation.py` | **Manual evaluation** — Runs 5 hard-coded `(query, expected answer)` pairs through the RAG pipeline, performs keyword-overlap scoring, and prints a `CORRECT / INCORRECT` verdict with a final score |
| `qa_llm_evaluation.py` | **LLM-assisted evaluation** — Same QA pairs evaluated by `QAEvalChain` (LLM-as-judge): the model compares each prediction against the expected answer and returns a `CORRECT / INCORRECT` grade |

### Chapter 6 — Agents

Demonstrates how LLM agents dynamically decide which tools to call in order to solve a task, using the ReAct (Reason + Act) pattern.

| File | Description |
|------|-------------|
| `agents.py` | **LangGraph agent** — Uses `create_agent` (LangChain 1.x / LangGraph API) with a `calculator` tool and a `WikipediaQueryRun` tool; demonstrates math computation and Wikipedia lookup |
| `agents_python.py` | **Python agent** — Equips the LLM with a `python_repl` tool to execute arbitrary Python code; used to sort a customer list by last name then first name |
| `agents_custom_tools.py` | **Custom tools agent** — Defines and registers 6 hand-crafted tools (`time`, `calculator`, `text_stats`, `celsius_to_fahrenheit`, `fahrenheit_to_celsius`, `reverse_string`) and runs a demo query for each |

## Shared Files

- `read_key.py` - Initializes the Hugging Face client with authentication
- `.env` - Environment variables

## Setup

### Prerequisites

- Python 3.8+
- Hugging Face API token

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install langchain langchain-classic langchain-community langchain-huggingface python-dotenv
   ```

   > **Note:** `langchain-classic` is required for `ConversationChain` and `ConversationBufferMemory`, which were removed from the core `langchain` package in version 1.x.

   For Chapter 4 (Q&A over Documents), also install:
   ```bash
   pip install sentence-transformers
   ```

   For Chapter 6 (`agents.py`), also install:
   ```bash
   pip install wikipedia langchain-community
   ```

3. Create a `.env` file with your Hugging Face token:
   ```
   HF_TOKEN=your_token_here
   HF_MODEL=Qwen/Qwen2.5-7B-Instruct
   ```

## Usage

### Chapter 1
```bash
python call_llm_model_classic.py   # classic API approach
python call_llm_model_langchain.py  # LangChain approach
python extract_review_info.py       # structured JSON extraction
```

### Chapter 2
```bash
python conversation_buffer_memory.py              # full history (ConversationBufferMemory)
python conversation_buffer_window_memory.py       # last N messages only
python conversation_buffer_token_memory.py        # token-limited history
python conversation_summary_memory.py             # AI-summarized history
```

### Chapter 3
```bash
python llmchain.py                # LLMChain — single prompt chain
python simple_sequential_chain.py  # SimpleSequentialChain — linear two-step pipeline
python sequential_chain.py         # SequentialChain — multi-input/output pipeline
python router_chain.py             # Router Chain — dynamic routing to sub-chains
```

### Chapter 4
```bash
python qa_over_documents.py        # RAG pipeline — Q&A over a product catalog
```

### Chapter 5
```bash
python qa_manual_evaluation.py     # Manual evaluation — hard-coded QA pairs with keyword scoring
python qa_llm_evaluation.py        # LLM-assisted evaluation — QAEvalChain grades each prediction
```

### Chapter 6
```bash
python agents.py                   # LangGraph agent — math (calculator) and Wikipedia lookup
python agents_python.py            # Python agent — sorts a list by executing generated code
python agents_custom_tools.py      # Custom tools agent — date, calculator, text stats, temperature, string reversal
```

## Key Concepts by Chapter

### Chapter 1 — Models, Prompts and Parsers
- Direct vs. LangChain-abstracted LLM calls
- `ChatPromptTemplate` and input variables
- `StrOutputParser` and `JsonOutputParser`
- Building chains with the `|` operator

### Chapter 2 — Memory
- **ConversationBufferMemory** — Full history retention (simple, unbounded growth)
- **ConversationBufferWindowMemory** — Last N messages only (fixed window)
- **ConversationTokenBufferMemory** — Token-based retention (memory limit in tokens)
- **ConversationSummaryMemory** — LLM-powered summarization (condenses conversation)
- `ConversationChain` to wire memory into a chat model
- Comparing memory strategies for long conversations

### Chapter 3 — Chains
- **LLMChain** — Basic building block: prompt → LLM → parser
- **SimpleSequentialChain** — Linear pipeline; single string passed between steps
- **SequentialChain** — Multi-variable pipeline; named inputs/outputs flow between steps
- **Router Chain** — Conditional routing; selects the best sub-chain for the given input
- LCEL `|` operator for composing chains declaratively

### Chapter 4 — Q&A over Documents
- **Document loading** — inline `CSVLoader` using Python's built-in `csv` module; produces `langchain_core` `Document` objects with no third-party loader dependency
- **Embeddings** — `HuggingFaceEmbeddings` with `sentence-transformers` for local, API-free vector representations
- **Vector store** — `InMemoryVectorStore` from `langchain_core` for lightweight, in-process similarity search
- **Index** — `VectorstoreIndexCreator` from `langchain_classic` wires loader → splitter → vector store in one call
- **Querying** — `index.query(question, llm=llm, chain_type="stuff")` for retrieval-augmented answer generation; `chain_type` controls how retrieved docs are combined (`stuff` / `map_reduce` / `refine` / `map_rerank`)

### Chapter 5 — Evaluation
- **Manual evaluation** — hard-coded `(query, expected answer)` pairs serve as a ground-truth test set
- **Keyword-overlap scoring** — checks whether key words from the expected answer appear in the model's response
- **LLM-assisted evaluation** — `QAEvalChain.from_llm(llm)` uses the model itself as a judge; it compares each prediction against the expected answer and returns `CORRECT` / `INCORRECT`
- **`QAEvalChain`** — from `langchain_classic.evaluation.qa`; accepts `examples` (with `query` / `answer` keys) and `predictions` (with `result` key)
- **Verdict & score** — each example is labelled and a final `X/N (%)` score is printed for both approaches

### Chapter 6 — Agents
- **`create_agent`** — LangGraph-based agent API (LangChain 1.x); takes an LLM and a list of tools and returns a compiled agent graph
- **`WikipediaQueryRun`** — wraps the `wikipedia` package to give the agent a live knowledge lookup tool
- **ReAct pattern** — agent reasons (Thought) then acts (Action/Observation) in a loop until the task is solved
- **`@tool` decorator** — turns any Python function into a LangChain tool; the docstring becomes the tool description the LLM reads
- **`create_tool_calling_agent`** — modern LCEL-based agent builder; pairs with `AgentExecutor` to run the loop
- **`AgentExecutor`** — orchestrates the Thought → Action → Observation loop with `verbose=True` tracing and `handle_parsing_errors=True` for robustness
- **Built-in tools** — `python_repl` executes generated code in a sandboxed stdout buffer
- **Custom tools** — `time` (today's date), `calculator` (safe `eval` with `math`), `text_stats` (word/char count), temperature converters, `reverse_string`

## Environment Variables

- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` - Your Hugging Face API token (required)
- `HF_MODEL` - Model ID to use (default: `Qwen/Qwen2.5-7B-Instruct`)
- `HF_EMBED_MODEL` - Sentence-transformers model for embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`)
