import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tool import build_retriever_tool

load_dotenv()

NOT_FOUND_MESSAGE = "Sorry, that information is not currently available in the knowledge base."

SYSTEM_PROMPT = (
    "You are a helpful assistant for agriculture. "
    "Answer only with information returned by the agriculture retriever tool. "
    f'If the tool does not return the answer, reply exactly: "{NOT_FOUND_MESSAGE}"'
)


def build_agent():
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
    return create_react_agent(model=model, tools=[build_retriever_tool()], prompt=SYSTEM_PROMPT)


def stream_reply(agent, messages):
    """Yield the assistant's reply to a list of chat messages, piece by piece."""
    for chunk, metadata in agent.stream({"messages": messages}, stream_mode="messages"):
        if metadata.get("langgraph_node") == "agent" and isinstance(chunk.content, str):
            yield chunk.content


if __name__ == "__main__":
    agent = build_agent()
    history = []
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": user_input})
        reply = ""
        for piece in stream_reply(agent, history):
            print(piece, end="", flush=True)
            reply += piece
        print()
        history.append({"role": "assistant", "content": reply})
