from langchain_classic.tools.retriever import create_retriever_tool

from vectorstore import get_vectorstore


def build_retriever_tool():
    retriever = get_vectorstore().as_retriever()
    return create_retriever_tool(
        retriever,
        name="agriculture",
        description="useful for answering questions about agriculture",
    )


if __name__ == "__main__":
    # Quick manual check that the vector store returns something
    print(build_retriever_tool().invoke({"query": "soil preparation"}))
