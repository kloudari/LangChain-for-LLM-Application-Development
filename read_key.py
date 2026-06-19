import os
from huggingface_hub import InferenceClient

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
	raise RuntimeError(
		"Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN in your .env file."
	)

hf_provider = os.getenv("HF_PROVIDER", "auto")
client = InferenceClient(provider=hf_provider, token=hf_token)