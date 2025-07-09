import os, certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

from fastapi import FastAPI, Query
from rag_engine import ask_question_with_cache
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/ask")
def ask(q: str = Query(...)):
    answer = ask_question_with_cache(q)
    return {"answer": answer}