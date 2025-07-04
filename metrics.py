from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np

def calculate_relevance(query: str, results: pd.DataFrame) -> float:
    """Calculate NDCG@10 score for results."""
    ideal = [3, 2, 1] + [0]*7
    actual = [
        1 if any(q in str(title).lower() for q in query.lower().split())
        else 0 
        for title in results["title"].head(10)
    ]
    return ndcg_score([ideal], [actual], k=10)
