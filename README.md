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
| `conversation_token_buffer_memory.py` | **ConversationTokenBufferMemory** — Keeps messages up to a token limit; manages memory by token count |
| `conversation_summary_memory.py` | **ConversationSummaryMemory** — Summarizes old messages; condenses history while preserving key information |

### Chapter 3 — Chains

Demonstrates how to compose multiple LLM calls into pipelines using LangChain's chain abstractions:

| File | Description |
|------|-------------|
| `llmchain.py` | **LLMChain** — Single prompt + LLM + parser wired as a LCEL chain |
| `simple_sequential_chain.py` | **SimpleSequentialChain** — Two chains in sequence; output of one feeds into the next (single input/output) |
| `sequential_chain.py` | **SequentialChain** — Multi-step pipeline with named inputs and outputs across steps |
| `router_chain.py` | **Router Chain** — Dynamically routes input to the most appropriate sub-chain based on content |

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
python conversation_token_buffer_memory.py        # token-limited history
python conversation_summary_memory.py             # AI-summarized history
```

### Chapter 3
```bash
python llmchain.py                # LLMChain — single prompt chain
python simple_sequential_chain.py  # SimpleSequentialChain — linear two-step pipeline
python sequential_chain.py         # SequentialChain — multi-input/output pipeline
python router_chain.py             # Router Chain — dynamic routing to sub-chains
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

## Environment Variables

- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` - Your Hugging Face API token (required)
- `HF_MODEL` - Model ID to use (default: `Qwen/Qwen2.5-7B-Instruct`)
