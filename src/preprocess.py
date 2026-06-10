import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os

def load_and_filter(path="data/reviews.parquet"):
    df = pd.read_parquet(path)
    
    # Keep only needed columns
    df = df[["user_id", "parent_asin", "rating", "text", "timestamp"]].copy()
    df = df.dropna(subset=["user_id", "parent_asin", "rating"])

    print(f"Total reviews before filtering: {len(df)}")

    # --- Filter active users and products ---
    # Keep users who reviewed at least 3 products
    user_counts = df["user_id"].value_counts()
    active_users = user_counts[user_counts >= 3].index
    df = df[df["user_id"].isin(active_users)]

    # Keep products with at least 10 reviews
    product_counts = df["parent_asin"].value_counts()
    active_products = product_counts[product_counts >= 10].index
    df = df[df["parent_asin"].isin(active_products)]

    print(f"After filtering: {len(df)} reviews")
    print(f"Users: {df['user_id'].nunique()}, Products: {df['parent_asin'].nunique()}")

    return df

def encode_ids(df):
    # Sort for reproducibility
    users = sorted(df["user_id"].unique())
    products = sorted(df["parent_asin"].unique())

    # Strictly 0-based sequential mapping
    user2idx = {u: i for i, u in enumerate(users)}
    product2idx = {p: i for i, p in enumerate(products)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["product_idx"] = df["parent_asin"].map(product2idx)

    # Drop any rows where mapping failed
    df = df.dropna(subset=["user_idx", "product_idx"])
    df["user_idx"] = df["user_idx"].astype(int)
    df["product_idx"] = df["product_idx"].astype(int)

    print(f"Encoded {len(users)} users (0 to {len(users)-1})")
    print(f"Encoded {len(products)} products (0 to {len(products)-1})")

    return df, user2idx, product2idx

def build_interaction_matrix(df, n_users, n_products):
    # Use implicit feedback — treat any rating as interaction
    # We'll use rating as weight (1-5 scale kept as is)
    rows = df["user_idx"].values
    cols = df["product_idx"].values
    data = df["rating"].values.astype(np.float32)

    matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_products))
    print(f"Interaction matrix shape: {matrix.shape}")
    print(f"Sparsity: {100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])):.2f}%")

    return matrix

def save_artifacts(df, user2idx, product2idx, matrix):
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/processed_reviews.parquet", index=False)

    with open("data/user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f)

    with open("data/product2idx.pkl", "wb") as f:
        pickle.dump(product2idx, f)

    # Save reverse mappings too
    idx2user = {v: k for k, v in user2idx.items()}
    idx2product = {v: k for k, v in product2idx.items()}

    with open("data/idx2user.pkl", "wb") as f:
        pickle.dump(idx2user, f)

    with open("data/idx2product.pkl", "wb") as f:
        pickle.dump(idx2product, f)

    import scipy.sparse as sp
    sp.save_npz("data/interaction_matrix.npz", matrix)

    print("\nAll artifacts saved to data/")

if __name__ == "__main__":
    df = load_and_filter()
    df, user2idx, product2idx = encode_ids(df)

    n_users = df["user_idx"].nunique()
    n_products = df["product_idx"].nunique()

    matrix = build_interaction_matrix(df, n_users, n_products)
    save_artifacts(df, user2idx, product2idx, matrix)

    print("\n✅ Preprocessing complete!")