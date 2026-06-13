from __future__ import annotations
import threading
from dataclasses import dataclass
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from local_rag_engine import LocalKnowledgeBase, LocalRagConfig, RetrievedChunk
from local_model import resolve_embedding_model_path, resolve_llm_model_path
TITLE = "中山大学人工智能学院知识问答助手"
DATA_FILE = "database.txt"
MODEL_PATH = resolve_llm_model_path()
@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
class LocalCausalAnswerer:
    def __init__(self, model_name: str = MODEL_PATH, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs = {"trust_remote_code": True}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
    def stream_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]],
        config: GenerationConfig,
    ):
        prompt_text = build_prompt_text(question, chunks, history, self.tokenizer)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generate_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if config.temperature > 0:
            generate_kwargs["temperature"] = config.temperature
        worker = threading.Thread(target=self.model.generate, kwargs=generate_kwargs)
        worker.start()
        for token in streamer:
            yield token
        worker.join()
def build_prompt_text(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]],
    tokenizer,
) -> str:
    context = "\n\n".join(
        f"[资料{i + 1}|相关度 {chunk.score:.2f}]\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    history_context = "\n".join(
        f"{message['role']}: {message['content']}" for message in history[-5:]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个基于本地知识库的中文问答助手。"
                "优先依据资料回答，回答要准确、简洁"
                "不要提及根据资料。"
                "如果资料无法支持结论，请明确说明知识库中没有相关信息。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"资料：\n{context}\n\n"
                f"对话历史：\n{history_context}\n\n"
                f"当前问题：{question}"
            ),
        },
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


@st.cache_resource
def init_knowledge_base() -> LocalKnowledgeBase:
    return LocalKnowledgeBase(
        LocalRagConfig(
            data_file=DATA_FILE,
            embedding_model_name=resolve_embedding_model_path(),
        )
    )

@st.cache_resource
def init_answerer(model_name: str) -> LocalCausalAnswerer:
    return LocalCausalAnswerer(model_name=model_name)
def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_sidebar(kb: LocalKnowledgeBase) -> tuple[int, GenerationConfig, str]:
    with st.sidebar:
        st.caption(f"Embedding: `{kb.config.embedding_model_name}`")
        model_name = st.text_input("生成模型", value=MODEL_PATH)
        top_k = st.slider("检索片段数", min_value=1, max_value=8, value=kb.config.top_k)
        max_new_tokens = st.slider("最长回答token", min_value=128, max_value=1536, value=512, step=128)
        temperature = st.slider("温度", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        st.divider()
        if st.button("重建本地知识库"):
            with st.spinner("正在用本地向量模型重建知识库..."):
                count = kb.rebuild(DATA_FILE)
            st.success(f"已写入 {count} 个知识片段。")
        if st.button("清空对话记录"):
            st.session_state.messages = []
            st.rerun()
        return top_k, GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature), model_name


def answer_once(
    question: str,
    kb: LocalKnowledgeBase,
    answerer: LocalCausalAnswerer,
    top_k: int,
    config: GenerationConfig,
) -> str:
    chunks = kb.query(question, top_k=top_k)
    placeholder = st.empty()
    full_answer = ""
    for token in answerer.stream_answer(
        question,
        chunks,
        st.session_state.messages[:-1],
        config,
    ):
        full_answer += token
        placeholder.markdown(full_answer + "|")
    placeholder.markdown(full_answer)
    return full_answer


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="centered")
    st.write(TITLE)
    init_session_state()
    kb = init_knowledge_base()
    top_k, config, model_name = render_sidebar(kb)
    with st.spinner("正在检查本地知识库"):
        kb.ensure_loaded(DATA_FILE)
    render_history()
    if question := st.chat_input("请输入"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answerer = init_answerer(model_name)
            answer = answer_once(question, kb, answerer, top_k, config)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
