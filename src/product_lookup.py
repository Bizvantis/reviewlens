import pandas as pd

_meta_df = None

def load_metadata():
    global _meta_df
    if _meta_df is None:
        _meta_df = pd.read_parquet("data/product_metadata.parquet")
        _meta_df = _meta_df.set_index("parent_asin")
    return _meta_df

def get_product_info(asin: str) -> dict:
    df = load_metadata()
    if asin in df.index:
        row = df.loc[asin]
        title = row["title"] if pd.notna(row["title"]) else "Unknown Product"
        title = title[:80] + "..." if len(str(title)) > 80 else str(title)
        return {
            "asin": asin,
            "title": title,
            "category": row["main_category"] if pd.notna(row["main_category"]) else "Musical Instruments",
            "avg_rating": round(float(row["average_rating"]), 1) if pd.notna(row["average_rating"]) else None,
            "rating_count": int(row["rating_number"]) if pd.notna(row["rating_number"]) else None,
            "price": None,
        }
    return {
        "asin": asin,
        "title": "Musical Instrument Product",
        "category": "Musical Instruments",
        "avg_rating": None,
        "rating_count": None,
        "price": None,
    }