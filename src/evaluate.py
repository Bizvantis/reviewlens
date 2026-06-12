import numpy as np
import scipy.sparse as sp
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

def load_artifacts():
    with open("data/als_model.pkl", "rb") as f:
        model = pickle.load(f)
    matrix = sp.load_npz("data/interaction_matrix.npz")
    with open("data/idx2product.pkl", "rb") as f:
        idx2product = pickle.load(f)
    return model, matrix, idx2product

def get_top_k(model, user_idx, k=10):
    user_vec = model.item_factors[user_idx]
    item_vecs = model.user_factors
    scores = item_vecs @ user_vec
    return np.argsort(scores)[::-1][:k].tolist()

def precision_at_k(recommended, relevant, k=10):
    return len(set(recommended[:k]) & relevant) / k

def recall_at_k(recommended, relevant, k=10):
    if len(relevant) == 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)

def ndcg_at_k(recommended, relevant, k=10):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def popularity_baseline(matrix, user_idx, k=10):
    # Recommend most popular items user hasn't seen
    user_row = matrix[user_idx].toarray().flatten()
    seen = set(np.where(user_row > 0)[0].tolist())
    popularity = np.array(matrix.sum(axis=0)).flatten()
    popular = np.argsort(popularity)[::-1]
    recs = [i for i in popular if i not in seen][:k]
    return recs

def evaluate(n_users=300, k=10, test_fraction=0.2, random_seed=42):
    model, matrix, idx2product = load_artifacts()
    rng = np.random.RandomState(random_seed)

    n_total_users = matrix.shape[0]
    user_indices = rng.choice(n_total_users, size=n_users, replace=False)

    results = {
        "model": {"precision": [], "recall": [], "ndcg": []},
        "popularity": {"precision": [], "recall": [], "ndcg": []},
        "random": {"precision": [], "recall": [], "ndcg": []},
    }

    n_products = matrix.shape[1]
    skipped = 0

    print(f"Evaluating {n_users} users at @{k}...")

    for user_idx in user_indices:
        user_row = matrix[user_idx].toarray().flatten()
        interacted = np.where(user_row > 0)[0]

        if len(interacted) < 5:
            skipped += 1
            continue

        # Hold out test items
        n_test = max(1, int(len(interacted) * test_fraction))
        test_items = set(rng.choice(interacted, size=n_test, replace=False).tolist())

        # --- Model ---
        model_recs = get_top_k(model, user_idx, k=k)
        results["model"]["precision"].append(precision_at_k(model_recs, test_items, k))
        results["model"]["recall"].append(recall_at_k(model_recs, test_items, k))
        results["model"]["ndcg"].append(ndcg_at_k(model_recs, test_items, k))

        # --- Popularity baseline ---
        pop_recs = popularity_baseline(matrix, user_idx, k=k)
        results["popularity"]["precision"].append(precision_at_k(pop_recs, test_items, k))
        results["popularity"]["recall"].append(recall_at_k(pop_recs, test_items, k))
        results["popularity"]["ndcg"].append(ndcg_at_k(pop_recs, test_items, k))

        # --- Random baseline ---
        random_recs = rng.choice(n_products, size=k, replace=False).tolist()
        results["random"]["precision"].append(precision_at_k(random_recs, test_items, k))
        results["random"]["recall"].append(recall_at_k(random_recs, test_items, k))
        results["random"]["ndcg"].append(ndcg_at_k(random_recs, test_items, k))

    print(f"Skipped {skipped} users (too few interactions)")
    print(f"Evaluated: {len(results['model']['precision'])} users\n")

    summary = {}
    for method in ["model", "popularity", "random"]:
        summary[method] = {
            "precision": np.mean(results[method]["precision"]),
            "recall": np.mean(results[method]["recall"]),
            "ndcg": np.mean(results[method]["ndcg"]),
        }

    print(f"{'Method':<20} {'Precision@10':<16} {'Recall@10':<14} {'NDCG@10'}")
    print("-" * 65)
    for method, vals in summary.items():
        print(f"{method:<20} {vals['precision']:<16.4f} {vals['recall']:<14.4f} {vals['ndcg']:.4f}")

    return results, summary

def plot_results(results, summary):
    os.makedirs("data", exist_ok=True)

    methods = ["ALS Model", "Popularity", "Random"]
    colors = ["#ff9900", "#00d26a", "#555555"]
    keys = ["model", "popularity", "random"]
    metrics = ["precision", "recall", "ndcg"]
    metric_labels = ["Precision@10", "Recall@10", "NDCG@10"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle('ReviewLens — ALS Model Evaluation vs Baselines',
                 color='white', fontsize=14, y=1.02)

    for ax, metric, label in zip(axes, metrics, metric_labels):
        ax.set_facecolor('#1a1f2e')
        vals = [summary[k][metric] for k in keys]
        bars = ax.bar(methods, vals, color=colors, width=0.5, edgecolor='none')
        ax.set_title(label, color='white', fontsize=12, pad=12)
        ax.set_ylabel('Score', color='white')
        ax.tick_params(colors='white')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#333')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0005,
                    f'{val:.4f}', ha='center',
                    color='white', fontsize=10)
        ax.set_ylim(0, max(vals) * 1.3 + 0.001)

    plt.tight_layout()
    plt.savefig('data/evaluation_chart.png', dpi=200,
                bbox_inches='tight', facecolor='#0f1117')
    print("\nChart saved to data/evaluation_chart.png")
    plt.close()

if __name__ == "__main__":
    results, summary = evaluate(n_users=300, k=10)
    plot_results(results, summary)
    print("\n✅ Evaluation complete!")