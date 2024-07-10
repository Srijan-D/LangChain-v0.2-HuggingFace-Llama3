import os
import textwrap
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv, find_dotenv

# huggingface inference endpoint
from langchain_huggingface import HuggingFaceEndpoint

#  The Runnable interface allows any two Runnables to be 'chained' together into sequences.
from langchain_core.output_parsers import StrOutputParser

# prompt template to the llm
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)

# implementing streaming stdout callback handler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Memory in LLMchain using RunnableWithMessageHistory analogous to using ``ConversationChain`` with the default ``ConversationBufferMemory``:   This class is deprecated in favor of ``RunnableWithMessageHistory` in v0.2
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, AIMessage

# Flask app simply to serve the index.html file and handle the POST request
app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static",
    template_folder="templates",
)

load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
# Get your token from here: https://huggingface.co/settings/tokens and set it in the .env file

# for streaming in response and logging it
callbacks = [StreamingStdOutCallbackHandler()]

# HuggingFace Serverless Inference API link: https://huggingface.co/docs/api-inference/en/index
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens=100,  # change this according to your requirement
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    callbacks=callbacks,
    streaming=True,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

#    # Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
# store is a dictionary that maps session IDs to their corresponding chat histories.
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


history = get_by_session_id("1")
history.add_message(AIMessage(content="hello"))
print(store)

# print first message in the history
print(history.messages[0].content)


# flask app routes
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # langChain code
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're an assistant who's good at giving breif answers to questions.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )

        print("*********************************")

        response = chain_with_history.invoke(
            {"question": query, "history": history},
            config={"configurable": {"session_id": "1"}},
        )
        print("response", response)

        wrapped_text = textwrap.fill(
            response, width=100, break_long_words=False, replace_whitespace=False
        )
        #  return render_template_string(open('index.html').read(), response=wrapped_text) # This is a bad idea as template can be huge but it is used to pass the response as a variable to the template(html)
        return wrapped_text

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
