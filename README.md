# LangChain for LLM Application Development

Based on : https://www.deeplearning.ai/courses/langchain
A learning project exploring how to use LangChain with Large Language Models (LLMs) for application development, comparing traditional API approaches with LangChain's abstractions.

## Project Overview

This project demonstrates two approaches to interacting with Hugging Face LLMs:

1. **Classic Approach** (`call_llm_model_classic.py`) - Direct use of the Hugging Face InferenceClient API
2. **LangChain Approach** (`call_llm_model_langchain.py`) - Using LangChain's abstractions and prompt templates

## Files

- `read_key.py` - Initializes the Hugging Face client with authentication
- `call_llm_model_classic.py` - Direct LLM calls using InferenceClient
- `call_llm_model_langchain.py` - LangChain chains with prompt templates
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
   pip install langchain langchain-huggingface python-dotenv
   ```

3. Create a `.env` file with your Hugging Face token:
   ```
   HF_TOKEN=your_token_here
   HF_MODEL=Qwen/Qwen2.5-7B-Instruct
   ```

## Usage

### Classic Approach
```bash
python call_llm_model_classic.py
```

### LangChain Approach
```bash
python call_llm_model_langchain.py
```

## Key Features

- Text translation and style transformation
- Prompt templating with LangChain
- Model switching and configuration
- Error handling for API calls

## Environment Variables

- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` - Your Hugging Face API token (required)
- `HF_MODEL` - Model ID to use (default: `Qwen/Qwen2.5-7B-Instruct`)
