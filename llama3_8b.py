import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import textwrap

from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

prompt_template.invoke({"topic": "cats"})


app = Flask(__name__, static_url_path='', static_folder='.')

load_dotenv(find_dotenv())  # Load environment variables

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
# Get your token from here: https://huggingface.co/settings/tokens and set it in the .env file
callbacks = [StreamingStdOutCallbackHandler()]

llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                max_new_tokens=300,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                callbacks=callbacks,
                streaming=True,
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
            )          

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        response = llm.invoke(query)
        wrapped_text = textwrap.fill(response, width=100, break_long_words=False, replace_whitespace=False)
        #  return render_template_string(open('index.html').read(), response=wrapped_text) # This is a bad idea as template can be huge but it is used to pass the response as a variable to the template(html)
        return wrapped_text
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
