import numpy as np
import pickle
import scipy.sparse as sp

def load_model_and_artifacts():
    with open("data/als_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("data/idx2product.pkl", "rb") as f:
        idx2product = pickle.load(f)

    with open("data/product2idx.pkl", "rb") as f:
        product2idx = pickle.load(f)

    matrix = sp.load_npz("data/interaction_matrix.npz")

    return model, idx2product, product2idx, matrix

def get_top1_recommendation(user_vec, item_vecs):
    scores = item_vecs @ user_vec
    top_idx = int(np.argmax(scores))
    return top_idx, scores

def generate_counterfactual(model, user_idx, idx2product, n_steps=200, step_size=0.05):
    """
    Find the minimal perturbation to the user's embedding
    that changes the top-1 recommendation to something different.
    
    This is the core counterfactual search:
    - Start from the user's learned embedding
    - Nudge it step by step
    - Stop when the top recommendation changes
    - Report: original rec, new rec, and what changed
    """

    # implicit: item_factors = user embeddings (swapped, as we learned)
    user_vec = model.item_factors[user_idx].copy().astype(np.float64)
    item_vecs = model.user_factors.astype(np.float64)  # shape: (180, 64)

    original_top_idx, original_scores = get_top1_recommendation(user_vec, item_vecs)
    original_asin = idx2product[original_top_idx]

    print(f"\nUser {user_idx} — Original top recommendation:")
    print(f"  Product idx: {original_top_idx} | ASIN: {original_asin}")
    print(f"  Score: {original_scores[original_top_idx]:.4f}")

    # --- Counterfactual search via gradient ascent on runner-up ---
    # Find the runner-up (2nd ranked product)
    sorted_indices = np.argsort(original_scores)[::-1]
    runner_up_idx = int(sorted_indices[1])
    runner_up_asin = idx2product[runner_up_idx]

    print(f"\nRunner-up (target for counterfactual):")
    print(f"  Product idx: {runner_up_idx} | ASIN: {runner_up_asin}")
    print(f"  Score: {original_scores[runner_up_idx]:.4f}")

    # Perturb user vector toward runner-up's item vector
    target_item_vec = item_vecs[runner_up_idx]
    perturbed_vec = user_vec.copy()

    counterfactual_found = False
    steps_taken = 0

    for step in range(n_steps):
        # Gradient: move user embedding toward target item
        direction = target_item_vec - item_vecs[original_top_idx]
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        perturbed_vec = perturbed_vec + step_size * direction

        new_top_idx, new_scores = get_top1_recommendation(perturbed_vec, item_vecs)

        if new_top_idx != original_top_idx:
            counterfactual_found = True
            steps_taken = step + 1
            break

    if counterfactual_found:
        new_asin = idx2product[new_top_idx]
        
        # Measure how much the embedding changed
        perturbation_magnitude = np.linalg.norm(perturbed_vec - user_vec)

        print(f"\n✅ Counterfactual found in {steps_taken} steps!")
        print(f"  New top recommendation: {new_asin} (idx: {new_top_idx})")
        print(f"  Perturbation magnitude: {perturbation_magnitude:.4f}")
        print(f"  Interpretation: A shift of {perturbation_magnitude:.4f} units in")
        print(f"  preference space flips the recommendation from")
        print(f"  {original_asin} → {new_asin}")

        return {
            "user_idx": user_idx,
            "original_asin": original_asin,
            "original_idx": original_top_idx,
            "counterfactual_asin": new_asin,
            "counterfactual_idx": new_top_idx,
            "steps_taken": steps_taken,
            "perturbation_magnitude": round(perturbation_magnitude, 4),
            "original_score": round(original_scores[original_top_idx], 4),
            "new_score": round(new_scores[new_top_idx], 4)
        }
    else:
        print(f"\n❌ No counterfactual found in {n_steps} steps.")
        return None

if __name__ == "__main__":
    model, idx2product, product2idx, matrix = load_model_and_artifacts()

    # Test counterfactual for a few users
    for user_idx in [0, 1, 2]:
        result = generate_counterfactual(model, user_idx, idx2product, n_steps=1000, step_size=0.0001)
        print("-" * 60)