<div align="center">

# 🔍 ReviewLens

### Counterfactual Explanation Engine for Amazon Product Recommendations

*Why was this recommended to you — and what would change it?*

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-Amazon%20Reviews%202023-orange?style=flat-square)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
[![Model](https://img.shields.io/badge/Model-ALS%20Collaborative%20Filtering-purple?style=flat-square)](https://implicit.readthedocs.io)
[![LLM](https://img.shields.io/badge/LLM-Groq%20LLaMA3-green?style=flat-square)](https://groq.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

[**Live Demo**](#) · [**API Docs**](#) · [**Motivation**](#-motivation) · [**Results**](#-evaluation)

![Architecture](docs/architecture.png)

</div>

---

## 📌 Overview

Modern recommendation systems are highly effective at predicting user preferences, but understanding *why* a particular item was recommended — and what it would take to change that recommendation — remains difficult.

> *"Why this product, and not that one?"*

**ReviewLens** is a counterfactual explanation engine that:

1. **Trains** an ALS collaborative filtering model on ~3M Amazon Musical Instruments reviews
2. **Finds** the minimal perturbation to a user's preference embedding that flips the top recommendation
3. **Explains** the result in plain English via Groq LLaMA 3
4. **Exposes** everything through a REST API for integration into other platforms

## 💡 Motivation

This addresses real, practical needs in recommendation systems:

| Use Case | Description |
|-----------|-------------|
| 🔧 Model Debugging | Understand which latent preference signals drive a recommendation |
| ⚖️ Explainability | Provide human-readable explanations for automated decisions |
| 📈 Seller Insights | Explore how product positioning affects recommendation outcomes |
| 🧪 Model Analysis | Compare recommendation behavior across model versions |

---

## 🏗 Architecture

```
Amazon Reviews 2023 (3M reviews)
        ↓
Preprocessing → Sparse Interaction Matrix (210K users × 22K products)
        ↓
ALS Training (64 factors, 30 epochs, implicit library)
        ↓
User Embedding → Gradient Perturbation → Flip Detection
        ↓
Counterfactual Result (original ASIN, CF ASIN, magnitude, steps)
        ↓
Groq LLaMA3 → Plain English Explanation
        ↓
Streamlit Demo + FastAPI REST Endpoints
```

---

## 📊 Evaluation

Evaluated on 78 users across 22,030 products:

| Method | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| ALS Model | 0.0192 | 0.1432 | 0.0959 |

> Popularity and random baselines score near zero on this 22K-product catalog with 99.98% sparsity, confirming the ALS model learns genuine personalization signals rather than popularity bias or noise.

**Counterfactual engine results (sample):**

| User | Original ASIN | CF ASIN | Steps | Magnitude | Stability |
|---|---|---|---|---|---|
| User 41 | B018FCZKR2 | B0CBK1WSMR | 253 | 0.1265 | High ✅ |
| User 176 | B006X9KG3HW | B07CYRYQ8G | 11 | 0.0055 | Low ❌ |
| User 231 | B0BHG58G2F | B07D5W5X3Z | 1 | 0.0001 | Unstable ⚠️ |

> Variation in stability across users is itself a finding — some users sit at razor-thin decision boundaries while others have strong, stable preferences.

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Bizvantis/reviewlens.git
cd reviewlens
conda create -n reviewlens python=3.10 -y
conda activate reviewlens
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
# Get a free key at https://console.groq.com
```

### 3. Download data & train model

```bash
python scripts/download_data.py
```

This single script downloads the dataset, preprocesses it, trains the ALS model, and saves all artifacts. Takes ~15 minutes on first run.

### 4. Generate evaluation metrics

```bash
python scripts/save_eval_metrics.py
```

### 5. Launch the demo

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 6. Launch the API

```bash
uvicorn api.main:app --reload
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 API Reference

### `GET /recommend/{user_idx}`

Get top-10 personalized recommendations for a user.

```bash
curl http://localhost:8000/recommend/41
```

```json
{
  "user_idx": 41,
  "user_id": "AGKASBHYZPGTEPO6LWZPVJWB2BVA",
  "recommendations": [
    {
      "rank": 1,
      "asin": "B018FCZKR2",
      "title": "Hosa Technology Guitar Cable",
      "avg_rating": 4.7,
      "rating_count": 2341,
      "score": 0.7982,
      "amazon_url": "https://www.amazon.com/dp/B018FCZKR2"
    }
  ]
}
```

### `POST /counterfactual`

Generate a counterfactual explanation for a user's top recommendation.

```bash
curl -X POST http://localhost:8000/counterfactual \
  -H "Content-Type: application/json" \
  -d '{"user_idx": 41, "step_size": 0.0001, "include_explanation": true}'
```

```json
{
  "user_idx": 41,
  "original_asin": "B018FCZKR2",
  "original_product": {
    "title": "Hosa Technology Guitar Cable",
    "avg_rating": 4.7
  },
  "counterfactual_asin": "B0CBK1WSMR",
  "counterfactual_product": {
    "title": "Ernie Ball Guitar Strings",
    "avg_rating": 4.8
  },
  "perturbation_magnitude": 0.1265,
  "steps_taken": 253,
  "stability": "high",
  "explanation": "You're being recommended the Hosa cable because your history shows a preference for performance accessories..."
}
```

### Other endpoints

| Endpoint | Description |
|---|---|
| `GET /stats` | Dataset and model statistics |
| `GET /product/{asin}` | Product metadata lookup by ASIN |
| `GET /health` | Health check |

---

## 📁 Project Structure

```
reviewlens/
├── api/
│   └── main.py               # FastAPI REST endpoints
├── app/
│   └── app.py                # Streamlit demo
├── data/                      # Generated artifacts (gitignored)
│   ├── als_model.pkl
│   ├── interaction_matrix.npz
│   ├── eval_results.json
│   └── evaluation_chart.png
├── docs/
│   └── architecture.png      # System architecture diagram
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_eda.py
│   └── figures/               # EDA visualizations
├── scripts/
│   ├── download_data.py      # One-shot setup script
│   ├── save_eval_metrics.py  # Evaluation runner
│   └── generate_architecture.py
├── src/
│   ├── preprocess.py          # Data preprocessing pipeline
│   ├── train_model.py         # ALS model training
│   ├── counterfactual.py       # Counterfactual search engine
│   ├── explainer.py            # Groq LLM explanation layer
│   ├── evaluate.py             # Evaluation metrics
│   └── product_lookup.py       # Product metadata lookup
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔬 How the Counterfactual Engine Works

The core idea is to treat recommendation explanations as an **optimization problem**.

Given a user embedding **u** and item embedding matrix **I**, the top recommendation is:

```
rec* = argmax_i (I · u)
```

To find the counterfactual, we search for the minimal perturbation **δ** such that:

```
argmax_i (I · (u + δ)) ≠ rec*
```

This is solved via **gradient ascent** toward the runner-up item's embedding direction:

```python
direction = item_vec[runner_up] - item_vec[original]
direction = direction / ||direction||
u_perturbed = u + step_size * direction
```

The **perturbation magnitude ||δ||** at the flip point is the stability metric — a smaller value means the recommendation is less robust to preference changes.

---

## 🌍 Real-World Applications

| Use Case | Description |
|---|---|
| 🔧 **ML Debugging** | Engineers can probe any user's recommendation and understand which preference signals drive it |
| ⚖️ **EU AI Act Compliance** | Users can request explanations for automated recommendation decisions |
| 📈 **Seller Analytics** | Sellers learn which preference dimensions their product is missing to reach more users |
| 🧪 **A/B Testing** | Users with low stability scores are good candidates for recommendation diversity experiments |

---

## 📦 Dataset

- **Source:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Category:** Musical Instruments
- **Reviews:** 3,017,439 total → 919,823 after filtering
- **Users:** 210,057 (active reviewers)
- **Products:** 22,030
- **Sparsity:** 99.98%
- **Citation:** Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Recommendation Model | `implicit` ALS Collaborative Filtering |
| Counterfactual Search | Gradient perturbation (NumPy) |
| LLM Explanations | Groq API · LLaMA 3.3 70B Versatile |
| Product Metadata | McAuley Amazon Reviews 2023 |
| Demo App | Streamlit |
| REST API | FastAPI + Uvicorn |
| Data Processing | Pandas · SciPy Sparse |
| Evaluation | Precision@K · Recall@K · NDCG@K |
| Language | Python 3.10 |

---

## 🔮 Roadmap

- [ ] Multi-category support (Electronics, Books, Fashion)
- [ ] Hybrid model with product features (content-aware ALS)
- [ ] Real-time retraining pipeline with new review ingestion
- [ ] Seller-facing dashboard with counterfactual insights
- [ ] Batch stability analysis across the entire user base
- [ ] Docker containerization for one-command deployment

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built on [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

*If you find this useful, please ⭐ the repo*

</div>