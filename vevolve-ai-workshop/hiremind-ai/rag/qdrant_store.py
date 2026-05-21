from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

VECTOR_DB = []

def init_db():
    global VECTOR_DB
    VECTOR_DB = []

def add_text(text):
    vector = embeddings.embed_query(text)
    VECTOR_DB.append({"vector": vector, "text": text})

def search(query):
    query_vec = embeddings.embed_query(query)

    # simple similarity (dot product approx)
    def similarity(v1, v2):
        return sum(a*b for a, b in zip(v1, v2))

    results = sorted(
        VECTOR_DB,
        key=lambda x: similarity(query_vec, x["vector"]),
        reverse=True
    )

    return [r["text"] for r in results[:3]]