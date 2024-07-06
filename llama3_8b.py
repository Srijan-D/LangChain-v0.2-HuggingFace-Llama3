import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import textwrap

app = Flask(__name__, static_url_path='', static_folder='.')

load_dotenv(find_dotenv())  # Load environment variables

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
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
        return wrapped_text
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
