"""
ReviewLens FastAPI — REST API for counterfactual recommendation explanations.
Docs available at /docs after running.

Usage:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pickle
import scipy.sparse as sp
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.counterfactual import generate_counterfactual
from src.explainer import generate_explanation
from src.product_lookup import get_product_info

# --- App ---
app = FastAPI(
    title="ReviewLens API",
    description="""
## Counterfactual Explanation Engine for Amazon Product Recommendations

ReviewLens answers the question: **"Why was this recommended to you — and what would change it?"**

### Endpoints
- `GET /health` — Health check
- `GET /recommend/{user_idx}` — Get top-10 recommendations for a user
- `POST /counterfactual` — Generate counterfactual explanation
- `GET /product/{asin}` — Get product info by ASIN
- `GET /stats` — Dataset and model statistics
    """,
    version="1.0.0",
    contact={
        "name": "ReviewLens",
        "url": "https://github.com/Bizvantis/reviewlens",
    }
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load artifacts once at startup ---
@app.on_event("startup")
async def load_artifacts():
    global model, matrix, idx2product, idx2user
    print("Loading model artifacts...")

    with open("data/als_model.pkl", "rb") as f:
        model = pickle.load(f)

    matrix = sp.load_npz("data/interaction_matrix.npz")

    with open("data/idx2product.pkl", "rb") as f:
        idx2product = pickle.load(f)

    with open("data/idx2user.pkl", "rb") as f:
        idx2user = pickle.load(f)

    print(f"✅ Model loaded — {matrix.shape[0]:,} users, {matrix.shape[1]:,} products")

# --- Request/Response Models ---
class CounterfactualRequest(BaseModel):
    user_idx: int
    step_size: Optional[float] = 0.0001
    n_steps: Optional[int] = 1000
    include_explanation: Optional[bool] = True

class RecommendationResponse(BaseModel):
    user_idx: int
    user_id: str
    recommendations: list

class CounterfactualResponse(BaseModel):
    user_idx: int
    user_id: str
    original_asin: str
    original_product: dict
    counterfactual_asin: str
    counterfactual_product: dict
    perturbation_magnitude: float
    steps_taken: int
    original_score: float
    counterfactual_score: float
    stability: str
    explanation: Optional[str] = None

# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "ALS Collaborative Filtering",
        "dataset": "Amazon Reviews 2023 - Musical Instruments",
        "version": "1.0.0"
    }

@app.get("/stats")
def stats():
    import json
    with open("data/eval_results.json") as f:
        eval_metrics = json.load(f)
    return {
        "dataset": "Amazon Reviews 2023 - Musical Instruments",
        "n_users": int(matrix.shape[0]),
        "n_products": int(matrix.shape[1]),
        "n_interactions": int(matrix.nnz),
        "sparsity": round(100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])), 4),
        "model": "ALS (Alternating Least Squares)",
        "factors": 64,
        "evaluation": eval_metrics
    }

@app.get("/recommend/{user_idx}", response_model=RecommendationResponse)
def recommend(user_idx: int, n: int = 10):
    if user_idx < 0 or user_idx >= matrix.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"user_idx must be between 0 and {matrix.shape[0]-1}"
        )

    user_vec = model.item_factors[user_idx]
    item_vecs = model.user_factors
    scores = item_vecs @ user_vec
    top_indices = np.argsort(scores)[::-1][:n]

    recommendations = []
    for idx in top_indices:
        asin = idx2product[int(idx)]
        info = get_product_info(asin)
        recommendations.append({
            "rank": len(recommendations) + 1,
            "asin": asin,
            "title": info["title"],
            "avg_rating": info["avg_rating"],
            "rating_count": info["rating_count"],
            "score": round(float(scores[idx]), 4),
            "amazon_url": f"https://www.amazon.com/dp/{asin}"
        })

    return {
        "user_idx": user_idx,
        "user_id": idx2user[user_idx],
        "recommendations": recommendations
    }

@app.get("/product/{asin}")
def product(asin: str):
    info = get_product_info(asin)
    if info["title"] == "Musical Instrument Product":
        raise HTTPException(status_code=404, detail=f"Product {asin} not found")
    return {
        **info,
        "amazon_url": f"https://www.amazon.com/dp/{asin}"
    }

@app.post("/counterfactual", response_model=CounterfactualResponse)
def counterfactual(request: CounterfactualRequest):
    if request.user_idx < 0 or request.user_idx >= matrix.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"user_idx must be between 0 and {matrix.shape[0]-1}"
        )

    result = generate_counterfactual(
        model,
        request.user_idx,
        idx2product,
        n_steps=request.n_steps,
        step_size=request.step_size
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Could not find a counterfactual for this user."
        )

    # Stability label
    mag = result["perturbation_magnitude"]
    if mag >= 0.05:
        stability = "high"
    elif mag >= 0.01:
        stability = "moderate"
    else:
        stability = "low"

    # Optional LLM explanation
    explanation = None
    if request.include_explanation:
        try:
            explanation = generate_explanation(result, user_reviews=[])
        except Exception:
            explanation = "Explanation unavailable."

    orig_info = get_product_info(result["original_asin"])
    cf_info = get_product_info(result["counterfactual_asin"])

    return {
        "user_idx": request.user_idx,
        "user_id": idx2user[request.user_idx],
        "original_asin": result["original_asin"],
        "original_product": {
            **orig_info,
            "amazon_url": f"https://www.amazon.com/dp/{result['original_asin']}"
        },
        "counterfactual_asin": result["counterfactual_asin"],
        "counterfactual_product": {
            **cf_info,
            "amazon_url": f"https://www.amazon.com/dp/{result['counterfactual_asin']}"
        },
        "perturbation_magnitude": result["perturbation_magnitude"],
        "steps_taken": result["steps_taken"],
        "original_score": result["original_score"],
        "counterfactual_score": result["new_score"],
        "stability": stability,
        "explanation": explanation
    }