import streamlit as st
from buildup import build_up, load_data
import os

APP_TITLE = "中山大学人工智能学院知识问答助手"
DATA_FILE = "database.txt"


@st.cache_resource
def init_resources():
    client, collection = build_up()
    return client, collection


def setup_page():
    st.set_page_config(page_title="", layout="centered")
    st.write(APP_TITLE)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def build_prompt(question, docs, history):
    history_context = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in history[-5:]]
    )
    return f"""你是一个基于本地知识库的助手。请参考“资料”和“对话历史”回答用户。
    资料:
    {docs}
    对话历史:
    {history_context}
    当前问题: 
    {question}
    直接给出回答，不要提及“根据资料显示”等字眼。如果无法从资料中得出结论，请结合常识回答。
"""


def answer(question, history, client, collection):
    query_embedding = client.embeddings.create(
        model="text-embedding-v4",
        input=question
    ).data[0].embedding

    responses_db = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    docs = responses_db['documents'][0]
    prompt = build_prompt(question, docs, history)
    responses = client.chat.completions.create(
        model="qwen3.6-flash",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    return responses


def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def stream_assistant_response(prompt, client, collection):
    response_placeholder = st.empty()
    full_response = ""
    response_stream = answer(
        prompt,
        st.session_state.messages[:-1],
        client,
        collection
    )
    for chunk in response_stream:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            response_placeholder.markdown(full_response + "|")
    response_placeholder.markdown(full_response)
    return full_response


def handle_chat_input(client, collection):
    if prompt := st.chat_input("请输入"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            full_response = stream_assistant_response(prompt, client, collection)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def update_database(file_path=DATA_FILE):
    if os.path.exists(file_path):
        with st.spinner(f"正在读取 {file_path} 并生成向量数据"):
            result = load_data(file_path)
            st.success(result)
    else:
        st.error(f"未找到 {file_path} 文件，请检查路径。")


def render_sidebar():
    with st.sidebar:
        if st.button("清空对话记录"):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        if st.button("更新数据库"):
            update_database()
        st.caption("数据已在本地储存。")


def main():
    setup_page()
    client, collection = init_resources()
    init_session_state()
    render_chat_history()
    handle_chat_input(client, collection)
    render_sidebar()


if __name__ == "__main__":
    main()
