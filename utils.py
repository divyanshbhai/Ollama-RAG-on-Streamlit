from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    return loader.load()

def embed_documents(category, documents):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore_dir = f"embeddings/{category}/chroma"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=vectorstore_dir)
    vectorstore.persist()
    return vectorstore

def get_vectorstore(category):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore_dir = f"embeddings/{category}/chroma"
    return Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
