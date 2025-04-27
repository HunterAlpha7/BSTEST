import pandas as pd

def apply_business_rules(results: pd.DataFrame, intent: str) -> pd.DataFrame:
    # Price boosting
    if "price" in intent:
        results["relevance"] *= 1.2 - (results["price"] / results["price"].max())
    
    # Brand prioritization
    popular_brands = {"sony": 1.2, "samsung": 1.15, "apple": 1.1}
    results["brand_score"] = results["brand"].str.lower().map(popular_brands).fillna(1.0)
    results["relevance"] *= results["brand_score"]
    
    # Category filtering
    category_boost = {"electronics": 1.5, "fashion": 1.3, "home": 1.1}
    results["category_score"] = results["category"].str.lower().map(category_boost).fillna(1.0)
    results["relevance"] *= results["category_score"]
    
    return results.sort_values("relevance", ascending=False).head(10)
