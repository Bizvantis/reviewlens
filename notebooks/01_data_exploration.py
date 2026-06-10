from datasets import load_dataset
import pandas as pd
import os

print("Downloading Amazon Reviews 2023 - Musical_Instruments...")

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Musical_Instruments",
    split="full",
    trust_remote_code=True
)

print(f"Total reviews loaded: {len(dataset)}")

df = pd.DataFrame(dataset)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Sample Row ---")
print(df.iloc[0])

print("\n--- Basic Stats ---")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Unique products: {df['parent_asin'].nunique()}")
print(f"Rating distribution:\n{df['rating'].value_counts().sort_index()}")

os.makedirs("data", exist_ok=True)
df.to_parquet("data/reviews.parquet", index=False)
print("\nSaved to data/reviews.parquet")

# Load item metadata
from datasets import load_dataset

print("Downloading Musical Instruments metadata...")
meta = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Musical_Instruments",
    split="full",
    trust_remote_code=True
)

meta_df = pd.DataFrame(meta)
print(f"Metadata loaded: {len(meta_df)} products")
print(f"Columns: {meta_df.columns.tolist()}")
print(meta_df[["parent_asin", "title", "main_category", "average_rating", "rating_number"]].head(3))

meta_df[["parent_asin", "title", "main_category", "average_rating", "rating_number"]].to_parquet("data/product_metadata.parquet", index=False)
print("Saved to data/product_metadata.parquet")