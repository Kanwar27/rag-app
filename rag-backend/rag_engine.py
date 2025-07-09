import json
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI  # New package for LLM
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

CACHE_FILE = "qa_cache.json"

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(qa_cache, f)

def load_cache():
    global qa_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            qa_cache = json.load(f)

load_dotenv()
load_cache()  # Load on startup

def print_chunks(chunks):
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:\n{chunk.page_content}")

def split_documents_character_text_splitter(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print_chunks(chunks)
    return chunks

# def split_documents_sentence_text_splitter(documents):
#     splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = splitter.split_documents(documents)
#     print_chunks(chunks)
#     return chunks

# def split_documents_recursive_text_splitter(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100,
#         length_function=len
#     )
#     chunks = text_splitter.split_documents(documents)
#     print_chunks(chunks)
#     return chunks

# Build the FAISS vector store
def build_vector_store():
    # Load text files from the 'documents' directory
    loader = DirectoryLoader("documents", glob="**/*.txt")
    documents = loader.load()

    chunksCharacterTextSplitter = split_documents_character_text_splitter(documents)
    # chunksSentenceTextSplitter = split_documents_sentence_text_splitter(documents)
    # chunksRecursiveTextSplitter = split_documents_recursive_text_splitter(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    vector_store = FAISS.from_documents(chunksCharacterTextSplitter, embeddings)

    print(f"Built FAISS vector store with {vector_store} vectors.")

    return vector_store

# Function to ask question
def ask_question(question: str) -> str:
    response = qa_chain.run(question)
    return response

qa_cache = {}

def ask_question_with_cache(question: str) -> str:
    normalized_question = question.strip().lower()
    if normalized_question in qa_cache:
        print("✅ Cache hit!")
        return qa_cache[normalized_question]["answer"]

    print("🧠 Querying OpenAI...")

    with get_openai_callback() as cb:
        result = qa_chain.run(normalized_question)
        print(f"🧾 Tokens used: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
        qa_cache[normalized_question] = {
            "answer": result,
            "timestamp": time.time()
        }
        save_cache()
        return result

# Build once at startup
vector_store = build_vector_store()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vector_store.as_retriever()
)
