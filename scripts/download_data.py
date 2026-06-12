"""
ReviewLens — One-shot data download and preprocessing script.
Run this once after cloning the repo to set up everything.

Usage:
    python scripts/download_data.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("ReviewLens — Data Setup")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)
    os.makedirs("notebooks/figures", exist_ok=True)

    # Step 1: Download reviews
    print("\n[1/4] Downloading Musical Instruments reviews...")
    from datasets import load_dataset
    import pandas as pd

    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Musical_Instruments",
        split="full",
        trust_remote_code=True
    )
    df = pd.DataFrame(dataset)
    df.to_parquet("data/reviews.parquet", index=False)
    print(f"     ✅ {len(df):,} reviews saved")

    # Step 2: Download metadata
    print("\n[2/4] Downloading product metadata...")
    meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Musical_Instruments",
        split="full",
        trust_remote_code=True
    )
    meta_df = pd.DataFrame(meta)
    meta_df[["parent_asin", "title", "main_category",
             "average_rating", "rating_number"]].to_parquet(
        "data/product_metadata.parquet", index=False
    )
    print(f"     ✅ {len(meta_df):,} products saved")

    # Step 3: Preprocess
    print("\n[3/4] Preprocessing...")
    from src.preprocess import load_and_filter, encode_ids, build_interaction_matrix, save_artifacts
    df = load_and_filter()
    df, user2idx, product2idx = encode_ids(df)
    n_users = df["user_idx"].nunique()
    n_products = df["product_idx"].nunique()
    matrix = build_interaction_matrix(df, n_users, n_products)
    save_artifacts(df, user2idx, product2idx, matrix)
    print(f"     ✅ {n_users:,} users, {n_products:,} products")

    # Step 4: Train model
    print("\n[4/4] Training ALS model...")
    import scipy.sparse as sp
    import pickle
    import implicit
    import numpy as np

    matrix = sp.load_npz("data/interaction_matrix.npz")
    item_user_matrix = matrix.T.tocsr().astype(np.float32)

    model = implicit.als.AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        iterations=30,
        calculate_training_loss=True,
        random_state=42
    )
    model.fit(item_user_matrix)

    with open("data/als_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"     ✅ Model trained and saved")

    print("\n" + "=" * 60)
    print("✅ Setup complete! Run: streamlit run app/app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()