import numpy as np
import scipy.sparse as sp
import pickle
import os
import implicit

def load_artifacts():
    matrix = sp.load_npz("data/interaction_matrix.npz")

    with open("data/user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)

    with open("data/product2idx.pkl", "rb") as f:
        product2idx = pickle.load(f)

    with open("data/idx2product.pkl", "rb") as f:
        idx2product = pickle.load(f)

    return matrix, user2idx, product2idx, idx2product

def train_model(matrix):
    item_user_matrix = matrix.T.tocsr().astype(np.float32)

    model = implicit.als.AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        iterations=30,
        calculate_training_loss=True,
        random_state=42
    )

    print("\nTraining ALS model...")
    model.fit(item_user_matrix)
    return model

def get_recommendations(model, user_idx, n=10):
    # Manually compute scores using learned embeddings
    # user_factors: (n_users, factors)
    # item_factors: (n_items, factors)
    user_vec = model.item_factors[user_idx]        # shape: (64,)
    item_vecs = model.user_factors                  # shape: (180, 64)

    scores = item_vecs @ user_vec                   # shape: (180,)
    top_indices = np.argsort(scores)[::-1][:n]      # top n product indices

    return [(int(i), round(float(scores[i]), 4)) for i in top_indices]

def save_model(model):
    os.makedirs("data", exist_ok=True)
    with open("data/als_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to data/als_model.pkl")

if __name__ == "__main__":
    matrix, user2idx, product2idx, idx2product = load_artifacts()

    print(f"Matrix shape: {matrix.shape}")
    print(f"Products in idx2product: {len(idx2product)}")

    model = train_model(matrix)

    print("\n--- Sample Recommendations for User 0 ---")
    recs = get_recommendations(model, user_idx=0, n=10)
    for product_idx, score in recs:
        asin = idx2product[product_idx]
        print(f"  Product idx: {product_idx} | ASIN: {asin} | Score: {score}")

    save_model(model)
    print("\n✅ Model training complete!")