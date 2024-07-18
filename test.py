# This is the entire LangChain v0.2 code snippet, without any frontend or flask setup. Change the harcoded question to get response from the llm
import os
import textwrap
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, AIMessage


load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

callbacks = [StreamingStdOutCallbackHandler()]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens=320,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    callbacks=callbacks,
    streaming=True,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# dictionary to store the chat message history with its session ID
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


history = get_by_session_id("1")
history.add_message(AIMessage(content="hello"))
print(store)

prompt_template = "Start answering the query with saying `I am an AI assistant here is your answer`: {query}"
query = "What is the capital of France?"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant who's good at {ability}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()


chain_with_history = RunnableWithMessageHistory(
    chain,
    # Uses the get_by_session_id function defined in the example
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

print("*********************************")

#  Define a RunnableConfig object, with a `configurable` key.
config = {"configurable": {"session_id": "1"}}

response = chain_with_history.invoke(
    {"ability": "math", "question": "what is the cosine formula?"},
    config=config,
)

print("response", response)
