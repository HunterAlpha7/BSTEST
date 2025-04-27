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

# Load data with cleaning
products = pd.read_csv("data/products.csv").fillna("")
product_texts = products["text"].tolist()

# Initialize models
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss_index.bin")
bm25 = BM25Okapi([str(text).lower().split() for text in product_texts])

# Metrics
REQUEST_COUNTER = Counter("search_requests", "Total requests")
LATENCY_GAUGE = Gauge("search_latency", "Response time (ms)")

def reciprocal_rank_fusion(semantic_ids, bm25_ids, k=60):
    scores = {}
    # Score semantic results
    for i, idx in enumerate(semantic_ids):
        scores[idx] = scores.get(idx, 0) + 1/(k + i + 1)
    # Score BM25 results
    for i, idx in enumerate(bm25_ids):
        scores[idx] = scores.get(idx, 0) + 1/(k + i + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

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
            return {"results": pd.read_json(cached.decode()).to_dict()}
        
        # Semantic search
        query_embedding = model.encode(query)
        _, semantic_ids = index.search(np.array([query_embedding]), 50)
        semantic_ids = semantic_ids[0].tolist()
        
        # Keyword search
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_ids = np.argsort(bm25_scores)[-50:][::-1].tolist()
        
        # Hybrid ranking
        combined_scores = reciprocal_rank_fusion(semantic_ids, bm25_ids)
        combined_ids = [idx for idx, _ in combined_scores[:100]]
        
        # Get results
        results = products.iloc[combined_ids].copy()
        results = results.replace([np.nan, np.inf, -np.inf], None)
        
        # Final sorting
        results = results.head(10)
        
        # Cache and return
        r.setex(query, 300, results.to_json(orient='records'))
        LATENCY_GAUGE.set((time.time() - start_time) * 1000)
        
        return {
            "results": results.to_dict(orient='records'),
            "latency_ms": (time.time() - start_time) * 1000
        }
        
    except Exception as e:
        return {"error": str(e)}
