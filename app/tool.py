import os
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv


load_dotenv()


# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001", 
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
# Load vectorstore
vectorstore = Chroma(
    collection_name="agriculture",
    embedding_function=embeddings,
    persist_directory="./agriculture_chromaV2"
)



# Initialize retriever
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(retriever, name="agriculture", description="useful for answering questions about agriculture")

result = retriever_tool.invoke({"query": "types of reward hacking"})



print(result)
