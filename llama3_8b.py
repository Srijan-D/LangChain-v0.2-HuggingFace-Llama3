import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint

import textwrap


load_dotenv(find_dotenv()) # loads the environment variables defined in it into the environment.

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
response=llm.invoke("what is the capital of india")

print(response)