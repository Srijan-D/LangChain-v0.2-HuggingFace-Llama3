import os
from dotenv import load_dotenv, find_dotenv
# from langchain.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
import textwrap

# hugingface hub api token

load_dotenv(find_dotenv()) # loads the environment variables defined in it into the environment.

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
print(llm.invoke("Hugging Face is"))


# query = "What is capital of India and UAE?"

# prompt = f"""
#  <|system|>
# You are an AI assistant that follows instruction extremely quite  well.
# Please ensure honesty and provide straightforward responses.
# </s>
#  <|user|>
#  {query}
#  </s>
#  <|assistant|>
# """

# response = llm.predict(prompt)
# wrapped_text = textwrap.fill(
#     response, width=100, break_long_words=False, replace_whitespace=False
# )
# print(wrapped_text)