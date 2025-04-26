from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import pandas as pd
import redis
import numpy as np
import time
from prometheus_client import Counter, Gauge, start_http_server
from metrics import calculate_relevance

app = FastAPI()
r = redis.Redis()

# Load data
products = pd.read_csv("data/products.csv")
product_texts = products["text"].tolist()

# Initialize models
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss_index.bin")
bm25 = BM25Okapi([text.split() for text in product_texts])

# Metrics
REQUEST_COUNTER = Counter("search_requests", "Total requests")
LATENCY_GAUGE = Gauge("search_latency", "Response time (ms)")

@app.on_event("startup")
def startup():
    start_http_server(8001)

@app.get("/search")
def search(query: str):
    REQUEST_COUNTER.inc()
    start_time = time.time()
    try:
        # Check cache
        if cached := r.get(query):
            LATENCY_GAUGE.set((time.time() - start_time) * 1000)
            return {"results": pd.read_json(cached.decode()).to_dict()}
        # Semantic search
        query_embedding = model.encode(query)
        _, ids = index.search(np.array([query_embedding]), 50)
        # Keyword search
        bm25_scores = bm25.get_scores(query.split())
        bm25_ids = np.argsort(bm25_scores)[-50:][::-1]
        # Combine results
        combined_ids = list(set(ids[0].tolist() + bm25_ids.tolist()))
        results = products.iloc[combined_ids].head(10)
        # Cache and return
        r.setex(query, 300, results.to_json())
        LATENCY_GAUGE.set((time.time() - start_time) * 1000)
        return {"results": results.to_dict()}
    except Exception as e:
        return {"error": str(e)}
