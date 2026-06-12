"""
Run this after training to save evaluation metrics to JSON.
These are loaded by the app instead of hardcoded values.

Usage:
    python scripts/save_eval_metrics.py
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate, plot_results

def main():
    print("Running evaluation...")
    results, summary = evaluate(n_users=300, k=10)
    plot_results(results, summary)

    metrics = {
        "precision_at_10": round(summary["model"]["precision"], 4),
        "recall_at_10": round(summary["model"]["recall"], 4),
        "ndcg_at_10": round(summary["model"]["ndcg"], 4),
        "popularity_precision": round(summary["popularity"]["precision"], 4),
        "random_precision": round(summary["random"]["precision"], 4),
        "users_evaluated": len(results["model"]["precision"]),
        "n_products": 22030,
        "dataset": "Amazon Reviews 2023 - Musical Instruments"
    }

    os.makedirs("data", exist_ok=True)
    with open("data/eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Metrics saved to data/eval_results.json")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()