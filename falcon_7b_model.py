import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub 


# hugingface hub api token

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

repo_id = "tiiuae/falcon-7b-instruct" 

falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_new_tokens": 500}
)
