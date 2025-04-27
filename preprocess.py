from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os

os.makedirs("data", exist_ok=True)

# Step 1: Load dataset
dataset = load_dataset(
    "wdc/products-2017",
    split="train",
    trust_remote_code=True,
    download_mode="force_redownload"
)
df = dataset.to_pandas()

# Step 2: Process products
def process_products(side: str) -> pd.DataFrame:
    products = df[[
        f"id_{side}", 
        f"title_{side}", 
        f"description_{side}", 
        f"category_{side}", 
        f"brand_{side}", 
        f"price_{side}"
    ]].copy()
    products.columns = ["id", "title", "description", "category", "brand", "price"]
    products["title"] = products["title"].fillna("").astype(str)
    products["description"] = products["description"].fillna("").astype(str)
    return products

left_products = process_products("left")
right_products = process_products("right")
all_products = pd.concat([left_products, right_products], ignore_index=True)
all_products.drop_duplicates(subset="id", inplace=True)

# Step 3: Prepare embeddings
all_products["text"] = all_products["title"] + " " + all_products["description"]
product_texts = all_products["text"].tolist()

# Step 4: Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    product_texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Step 5: Save index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "data/faiss_index.bin")
all_products.to_csv("data/products.csv", index=False)
