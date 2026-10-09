# Agriculture Chatbot

A Streamlit chatbot that answers questions about agriculture from a document knowledge base, using a retrieval-augmented LangGraph agent.

Forked from [kush2022/vibration_analysis_rag](https://github.com/kush2022/vibration_analysis_rag).

## Features

- Chat interface built with Streamlit, with streamed responses
- Answers come only from the knowledge base (a Chroma vector store)
- Follow-up questions work: the whole conversation is sent to the agent
- If the information is not in the knowledge base, the bot says so

## Setup

1. **Clone the repository** and move into the project directory.

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**. Create a `.env` file in the project root:
    ```
    OPENAI_API_KEY=your_openai_api_key
    ```
    The key is used for both the chat model (`gpt-4o-mini`) and the embeddings (`text-embedding-3-large`).

4. **Build the vector store**. Put your source documents in `app/AgricultureNB_LM/` (or pass another folder) and run:
    ```bash
    python app/rag_pipeline.py
    # or: python app/rag_pipeline.py path/to/documents
    ```
    This creates `app/agriculture_chromaV2/`. Documents are embedded in small batches, and a batch that hits a rate limit is retried.

    If your documents are `.txt` or `.docx`, `python app/convert_to_pdf.py path/to/documents` converts them to PDF first.

## Running the app

```bash
streamlit run app/main.py
```

For a terminal version of the same agent:

```bash
python app/agent.py
```

## Project structure

- `app/main.py`: Streamlit chat interface
- `app/agent.py`: agent definition and a command-line chat loop
- `app/tool.py`: retriever tool
- `app/vectorstore.py`: embedding model and vector store settings, shared by indexing and retrieval
- `app/rag_pipeline.py`: builds the vector store from a folder of documents
- `app/convert_to_pdf.py`: converts `.txt` and `.docx` files to PDF
- `requirements.txt`: Python dependencies

## Changes in this fork

- Removed an API key that was hard-coded in `rag_pipeline.py`; keys are now read from `.env` only
- Indexing and retrieval use the same embedding model (previously the index was built with one model and queried with another)
- The vector store path no longer depends on the directory the app is started from
- Importing `tool.py` no longer runs a test query on every start
- The chatbot keeps conversation history, and the agent is built once instead of on every rerun
- Rate-limited batches are retried during indexing instead of being skipped
- The Streamlit app and the command-line agent share one agent definition

## License

MIT License
