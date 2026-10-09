import streamlit as st

from agent import build_agent, stream_reply

st.set_page_config(page_title="Agriculture Chatbot", page_icon=":seedling:")
st.title("🌱 Agriculture Chatbot")


@st.cache_resource
def get_agent():
    # Built once per server process rather than on every rerun
    return build_agent()


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything about agriculture..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            # The whole history is sent so the agent can handle follow-up questions
            bot_response = st.write_stream(stream_reply(get_agent(), st.session_state["messages"]))
        except Exception as e:
            bot_response = None
            st.error(f"Could not get a response: {e}")

    if bot_response:
        st.session_state["messages"].append({"role": "assistant", "content": bot_response})
