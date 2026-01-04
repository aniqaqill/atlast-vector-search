import os
import subprocess
from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

def get_windows_host_ip() -> str:
    """Automatically finds the Windows host IP from inside WSL."""
    try:
        # Grabs the gateway IP (your Windows side)
        cmd = "ip route | grep default | awk '{print $3}'"
        return subprocess.check_output(cmd, shell=True, timeout=5).decode('utf-8').strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return "127.0.0.1"  # Fallback

client = MongoClient(os.getenv("MONGODB_URI"))
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]

loader = PyPDFLoader("./sample/mongodb.pdf")
pages = loader.load()
cleaned_pages = []

for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

WINDOWS_HOST_IP = get_windows_host_ip()
ollama_base_url = os.getenv("OLLAMA_BASE_URL", f"http://{WINDOWS_HOST_IP}:11434")

if "{WINDOWS_HOST_IP}" in ollama_base_url:
    ollama_base_url = ollama_base_url.replace("{WINDOWS_HOST_IP}", WINDOWS_HOST_IP)

LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL", 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')

if "bge" in LANGUAGE_MODEL and "Llama" in EMBEDDING_MODEL:
    print("WARNING: Models appear to be swapped in configuration. Swapping them back automatically.")
    LANGUAGE_MODEL, EMBEDDING_MODEL = EMBEDDING_MODEL, LANGUAGE_MODEL

print(f"Connecting to Ollama at {ollama_base_url}")
print(f"Using Language Model: {LANGUAGE_MODEL}")
print(f"Using Embedding Model: {EMBEDDING_MODEL}")

llm = ChatOllama(base_url=ollama_base_url, model=LANGUAGE_MODEL, temperature=0) 

parser = JsonOutputParser()
prompt = PromptTemplate(
    template="""You are a metadata extractor. Extract metadata from the following text into a JSON object with strictly these keys:
- title: string (a suitable title for the content)
- keywords: array of strings (list of important keywords)
- hasCode: boolean (true if the text contains code snippets, false otherwise)

Return ONLY the JSON object.

Text:
{text}

JSON:
""",
    input_variables=["text"],
)

chain = prompt | llm | parser

print("Extracting metadata and processing documents...")
docs = []
for i, page in enumerate(cleaned_pages):
    print(f"Processing page {i+1}/{len(cleaned_pages)}")
    try:
        metadata = chain.invoke({"text": page.page_content})
        
        new_metadata = page.metadata.copy()
        if isinstance(metadata, dict):
            new_metadata.update(metadata)
        else:
            print(f"Warning: Metadata is not a dict for page {i+1}")
            
        docs.append(Document(page_content=page.page_content, metadata=new_metadata))
    except Exception as e:
        print(f"Error processing page {i+1}: {e}")
        docs.append(page)

split_docs = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(base_url=ollama_base_url, model=EMBEDDING_MODEL)


print(f"Total documents to process: {len(split_docs)}")

vectorStore = MongoDBAtlasVectorSearch.from_documents(
    split_docs, embeddings, collection=collection
)

print("Vector store updated successfully.")
