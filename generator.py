import os
import subprocess
from dotenv import load_dotenv
load_dotenv()

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_windows_host_ip() -> str:
    """Automatically finds the Windows host IP from inside WSL."""
    try:
        # Grabs the gateway IP (your Windows side)
        cmd = "ip route | grep default | awk '{print $3}'"
        return subprocess.check_output(cmd, shell=True, timeout=5).decode('utf-8').strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return "127.0.0.1"  # Fallback

dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
index = "vector_index"

# Ollama setup
WINDOWS_HOST_IP = get_windows_host_ip()
ollama_base_url = os.getenv("OLLAMA_BASE_URL", f"http://{WINDOWS_HOST_IP}:11434")

# FIX: If the env var contains the literal placeholder string, replace it with the actual IP
if "{WINDOWS_HOST_IP}" in ollama_base_url:
    ollama_base_url = ollama_base_url.replace("{WINDOWS_HOST_IP}", WINDOWS_HOST_IP)

# Model Configuration
LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL", 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')

# SWAP CHECK: If the user accidentally swapped them in .env, we swap them back here to prevent errors.
if "bge" in LANGUAGE_MODEL and "Llama" in EMBEDDING_MODEL:
    print("WARNING: Models appear to be swapped in configuration. Swapping them back automatically.")
    LANGUAGE_MODEL, EMBEDDING_MODEL = EMBEDDING_MODEL, LANGUAGE_MODEL


print(f"Connecting to Ollama at {ollama_base_url}")
print(f"Using Language Model: {LANGUAGE_MODEL}")
print(f"Using Embedding Model: {EMBEDDING_MODEL}")

vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    os.getenv("MONGODB_URI"),
    dbName + "." + collectionName,
    OllamaEmbeddings(base_url=ollama_base_url, model=EMBEDDING_MODEL),
    index_name=index,
)

def query_data(query):
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "pre_filter": { "hasCode": { "$eq": False } },
            "score_threshold": 0.01
        },
    )

    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Do not answer the question if there is no given context.
    Do not answer the question if it is not related to the context.
    Do not give recommendations to anything other than MongoDB.
    Context:
    {context}
    Question: {question}
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    retrieve = {
        "context": retriever | (lambda docs: print(f"DEBUG: Retrieved {len(docs)} docs") or "\n\n".join([d.page_content for d in docs])), 
        "question": RunnablePassthrough()
        }

    llm = ChatOllama(base_url=ollama_base_url, model=LANGUAGE_MODEL, temperature=0)

    response_parser = StrOutputParser()

    rag_chain = (
        retrieve
        | custom_rag_prompt
        | llm
        | response_parser
    )

    answer = rag_chain.invoke(query)
    

    return answer

print(query_data("Who wrote the Little MongoDB book?"))