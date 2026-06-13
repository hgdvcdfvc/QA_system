from openai import OpenAI
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os

def build_up():
    client = OpenAI(api_key="", #输入自己的apikey。
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    db_client = chromadb.PersistentClient(path="./txt_knowledge_base")
    collection = db_client.get_or_create_collection(name="txt_docs")
    return client, collection

def load_data(file_path):
    client, collection= build_up()
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,       
        chunk_overlap=20,    
        separators=["\n\n", "\n", "。", "！", "？"] 
    )
    chunks = text_splitter.split_text(content)

    for i, chunk in enumerate(chunks):
        embedding_res = client.embeddings.create(
            model="text-embedding-v4",
            input=chunk
        )
        collection.add(
            ids=[f"txt_chunk_{i}"],
            embeddings=[embedding_res.data[0].embedding],
            documents=[chunk],
            metadatas=[{"source": file_path}] 
        )
    return f"处理完成，切分为 {len(chunks)} 个片段。"
