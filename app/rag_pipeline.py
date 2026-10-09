"""Build the Chroma vector store from the documents in a folder.

Usage:
    python app/rag_pipeline.py [path/to/documents]
"""
import argparse
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore import APP_DIR, COLLECTION_NAME, PERSIST_DIR, get_embeddings

DEFAULT_SOURCE_DIR = APP_DIR / "AgricultureNB_LM"


def load_chunks(source_dir):
    docs = DirectoryLoader(str(source_dir)).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_documents(docs)


def is_rate_limit_error(error):
    message = str(error).lower()
    return "quota" in message or "429" in message or "rate limit" in message


def create_vectorstore_with_batches(docs, embeddings, batch_size=10, delay=2, max_retries=5):
    """Create the vector store in batches to stay under embedding rate limits.

    A batch that hits a rate limit is retried with a growing wait, so no
    documents are silently left out of the index.
    """
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        print(f"Processing batch {start // batch_size + 1}: documents {start + 1} to {start + len(batch)}")

        for attempt in range(1, max_retries + 1):
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                if not is_rate_limit_error(e) or attempt == max_retries:
                    raise
                wait = 10 * attempt
                print(f"Rate limit hit. Waiting {wait} seconds before retrying...")
                time.sleep(wait)

        if start + batch_size < len(docs):
            time.sleep(delay)

    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Index a folder of documents into the Chroma vector store.")
    parser.add_argument("source_dir", nargs="?", default=DEFAULT_SOURCE_DIR,
                        help=f"Folder containing the documents (default: {DEFAULT_SOURCE_DIR})")
    args = parser.parse_args()

    chunks = load_chunks(args.source_dir)
    if not chunks:
        raise SystemExit(f"No documents found in {args.source_dir}")

    create_vectorstore_with_batches(chunks, get_embeddings())
    print(f"Indexed {len(chunks)} chunks into {PERSIST_DIR}")


if __name__ == "__main__":
    main()
