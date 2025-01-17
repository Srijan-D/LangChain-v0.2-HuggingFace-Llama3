# huggingFace-langchain-Meta-Llama-3-8B

This project integrates LangChain v0.2.6, HuggingFace Serverless Inference API, and Meta-Llama-3-8B-Instruct. It provides a chat-like web interface to interact with a language model and maintain conversation history using the Runnable interface, the upgraded version of LLMChain.
LLMChain has been deprecated since 0.1.17.

## Important Features:

- This is project demontrates how to setup `LangChain v0.2.6` and integrate it with `HuggingFace Serverless Inference API` using huggingface-hub v0.23, and `Meta-Llama-3-8B` llm model
- **HuggingFace Serverless Inference API:** Use publicly accessible machine learning models or private ones via simple HTTP requests, inference hosted on Hugging Face's infrastructure.

  ```python
  from langchain_huggingface import HuggingFaceEndpoint

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

  ```

- **LLMChains using Runnable interface**: LLMChain has been deprecated in LangChainv0.2 hence I have used RunnableSequence instead. LLMChain combines a prompt template, LLM, and output parser into a class. Chain interface makes it easy to add `Statefulness` and `Memory` to any Chain to give it state, helps to pass `Callbacks` to a Chain to execute additional functionality.

  ```python
  from langchain_core.output_parsers import StrOutputParser

  chain = prompt | llm | StrOutputParser()
  ```

- The `|` symbol is similar to a unix pipe operator, which chains together the different components, feeding the output from one component as input into the next component.

  ![Diagram for v0.2](/readme-images/image.png)

- To follow the steps along:

  - We pass in user input as the query {"question": "what is the capital of India?"}, although here the query is directly coming from the front end for
    In the image we have used `topic=ice cream` for example purposes only.

  ```python
  response = chain_with_history.invoke(
           {"question": query},
           config={"configurable": {"session_id": "1"}},
       )
  ```

  - The `prompt` component takes the user input, which is then used to construct a PromptValue after using the query to construct the prompt.
  - The model component takes the generated prompt, and passes into the `Llama3 LLM model` for evaluation. The generated output from the model is a ChatMessage object.
  - Finally, the `output_parser` component takes in a ChatMessage, and transforms this into a Python string, which is returned from the invoke method
    The specific `StrOutputParser` simply converts any input into a string.

- **PromptTemplate:** Prompt template for chat models. ChatPromptTemplate implements the standard Runnable Interface. Used to create flexible templated prompts for chat models.

  - `MessagesPlaceholder`: Prompt template that assumes variable is already list of messages. Placeholder which is used to replace or pass in a list of messages from history during runtime

    ```python
    from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    )

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
    ```

- **Streaming:** Callback Handler streams to stdout on new llm token. Only works with LLMs that support streaming.

  ```python
  from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

  callbacks = [StreamingStdOutCallbackHandler()]

  llm = HuggingFaceEndpoint(
  repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
  top_k=10,
  callbacks=callbacks,
  streaming=True,
  huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
  )
  ```

### Memory in LLM chain (LangChain v0.2)

- **Memory with RunnableWithMessageHistory:** maintains Chain state, incorporating context from past runs.
- Memory in LLMchain using RunnableWithMessageHistory analogous to using `ConversationChain` with the default `ConversationBufferMemory`: This class is deprecated in favor of ` RunnableWithMessageHistory` in LangChain v0.2.
  The update includes stream, batch, and async support and flexible memory handling that extends to managing memory outside the chain.

  `RunnableWithMessageHistory` must always be called with a config that contains the appropriate parameters for the chat message history factory. By default the Runnable is expected to take a single configuration parameter called `session_id` which is a string. This parameter is used to create a new or look up an existing chat message history that matches the given session_id.

  `input_messages_key`: Key of the query `{"question":query}` that is passed during the `.invoke()` function Must be specified if the base runnable accepts a dict as input.

  `output_messages_key`: Must be specified if the base runnable returns a dict as output.

  `BaseChatMessageHistory`: Abstract base class for storing chat message history

  `InMemoryChatMessageHistory`: In memory implementation of chat message history. Stores messages in an in memory list.

  ```python
  from langchain_core.runnables.history import RunnableWithMessageHistory
  from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
  )

  store = {} // Here we use a global variable to store the chat message history.
  query="what is the capital of India?" (From the frontend form)

  def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

  history = get_by_session_id("1")

  chain_with_history = RunnableWithMessageHistory(
            chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )

  response = chain_with_history.invoke(
            {"question": query},
            config={"configurable": {"session_id": "1"}},
        )
  ```

  ### Difference b/w LangChain v0.1 & v0.2

  ![Comparing v0.1 & v0.2](/readme-images/image-1.png)

  ### Live working example of history:

  ![Live example](/readme-images/image-4.png)

## Project Structure

```plaintext
HUGGINGFACE-LANGCHAIN-LLAMA3
├── .git
├── .vercel
├── myenv
├── readme-images/
│   ├── image-1.png
│   ├── image-2.png
│   ├── image-3.png
│   ├── image-4.png
│   ├── image-5.png
│   └── image.png
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── index.js
├── templates/
│   └── index.html
├── __pycache__
├── .env
├── .env.local
├── .gitignore
├── .vercelignore
├── LICENSE
├── README.md
├── requirements.txt
├── llama3_8b.py
├── test.py
└── vercel.json
```

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- LangChain v0.2
- Hugging Face API Token <!-- - https://huggingface.co/settings/tokens -->

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Srijan-D/langchainv0.2-huggingface-llama3.git
   cd HUGGINGFACE-LANGCHAIN-LLAMA3
   ```

2. **Create a virtual environment:**:

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the environment variables:**:

   Create a .env file in the root directory and add your Hugging Face API token:

   Visit the following url to get your access token for free https://huggingface.co/settings/tokens

   Add the api key to the .env file

   ```bash
   HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
   ```

## Running the Application

1. **Start the Flask application:**

   ```bash
   python llama3_8b.py
   ```

2. **Interact with the application:**

   Open your web browser and navigate to http://127.0.0.1:5000 to interact with the application.

## Testing the Application

You can test the application using the provided test.py file which contains the code used in the POST route of the Flask app. This allows you to simulate queries directly.

simply run the following:

```bash
python test.py
```

## License

This project is licensed under the MIT License - see the [LICENSE File](https://github.com/Srijan-D/langchainv0.2-huggingface-llama3/blob/main/LICENSE)
for details.

## Contributing

Feel free to fork this repository and contribute by submitting a pull request. For major changes, please open an issue to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Acknowledgments

- [Hugging Face](https://huggingface.co/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Flask](https://flask.palletsprojects.com/)
