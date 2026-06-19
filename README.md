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

Explores how to give LLMs short-term conversational memory using `ConversationBufferMemory` and `ConversationChain`.

| File | Description |
|------|-------------|
| `conversation_buffer_memory.py` | Multi-turn conversation using `ConversationChain` with `ConversationBufferMemory` to retain the full history |

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
python conversation_buffer_memory.py  # multi-turn conversation with memory
```

## Key Concepts by Chapter

### Chapter 1 — Models, Prompts and Parsers
- Direct vs. LangChain-abstracted LLM calls
- `ChatPromptTemplate` and input variables
- `StrOutputParser` and `JsonOutputParser`
- Building chains with the `|` operator

### Chapter 2 — Memory
- `ConversationBufferMemory` to store the full conversation history
- `ConversationChain` to wire memory into a chat model
- Inspecting accumulated history via `memory.buffer`

## Environment Variables

- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` - Your Hugging Face API token (required)
- `HF_MODEL` - Model ID to use (default: `Qwen/Qwen2.5-7B-Instruct`)
