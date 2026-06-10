import streamlit as st
import scipy.sparse as sp
import pickle
import numpy as np
import sys
import os
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.counterfactual import generate_counterfactual, load_model_and_artifacts
from src.explainer import generate_explanation
from src.product_lookup import get_product_info

# --- Page Config ---
st.set_page_config(
    page_title="ReviewLens | Amazon Recommendation Explainer",
    page_icon="🔍",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .product-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .asin-badge {
        background: #232f3e;
        color: #ff9900;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: monospace;
        font-size: 13px;
        font-weight: bold;
    }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .stability-high { color: #00d26a; }
    .stability-low { color: #ff4444; }
    .amazon-orange { color: #ff9900; }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        margin: 20px 0 10px 0;
        color: #ffffff;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a2a1a, #1a3a1a);
        border-left: 4px solid #00d26a;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        font-size: 16px;
        line-height: 1.6;
        color: #e0e0e0;
    }
    .review-box {
        background: #1a1f2e;
        border-left: 3px solid #ff9900;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14px;
        color: #c0c0c0;
        font-style: italic;
    }
    .tag {
        display: inline-block;
        background: #2d3561;
        color: #a0aec0;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin: 3px;
    }
    .flip-arrow {
        font-size: 32px;
        text-align: center;
        color: #ff9900;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load artifacts ---
@st.cache_resource
def load_all():
    model, idx2product, product2idx, matrix = load_model_and_artifacts()
    df = pd.read_parquet("data/processed_reviews.parquet")
    with open("data/idx2user.pkl", "rb") as f:
        idx2user = pickle.load(f)
    with open("data/user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)
    return model, idx2product, product2idx, matrix, df, idx2user, user2idx

@st.cache_data(ttl=3600)
def fetch_product_info(asin: str) -> dict:
    """Fetch real product title from Amazon via Open Library / fallback."""
    try:
        url = f"https://www.amazon.in/dp/{asin}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "html.parser")
            title_tag = soup.find(id="productTitle")
            if title_tag:
                return {"title": title_tag.get_text().strip(), "asin": asin}
    except:
        pass
    return {"title": f"Musical Instrument Product", "asin": asin}

def stability_label(magnitude: float) -> tuple:
    if magnitude >= 0.05:
        return "High Stability", "stability-high", "✅"
    elif magnitude >= 0.01:
        return "Moderate Stability", "amazon-orange", "⚠️"
    else:
        return "Low Stability", "stability-low", "❌"

# --- Header ---
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size:52px;margin-top:10px'>🔍</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin-bottom:0;color:#ffffff'>ReviewLens</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;font-size:16px;margin-top:4px'>Counterfactual Explanation Engine for Amazon Product Recommendations</p>", unsafe_allow_html=True)

st.markdown("---")

# --- Load data ---
model, idx2product, product2idx, matrix, df, idx2user, user2idx = load_all()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    user_idx = st.slider(
        "Select User Index",
        min_value=0,
        max_value=min(500, matrix.shape[0] - 1),
        value=41,
        help="Each index represents a real Amazon reviewer"
    )

    step_size = st.select_slider(
        "Counterfactual Precision",
        options=[0.0001, 0.0005, 0.001, 0.005],
        value=0.0001,
        help="Smaller = more precise minimal perturbation"
    )

    st.markdown("---")
    st.markdown("### 📖 How It Works")
    st.markdown("""
    1. **Train** an ALS collaborative filtering model on Amazon reviews
    2. **Perturb** the user's preference embedding minimally
    3. **Detect** the flip point where recommendation changes
    4. **Explain** the insight in plain English via LLM
    """)
    st.markdown("---")
    st.caption("Built on Amazon Reviews 2023 · Musical Instruments · ALS + Groq LLaMA3")

# --- User Profile ---
user_id = idx2user[user_idx]
user_reviews_df = df[df["user_id"] == user_id]["text"].dropna().tolist()
user_ratings_df = df[df["user_id"] == user_id]["rating"].tolist()

st.markdown(f"### 👤 User Profile")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("User Index", user_idx)
with col2:
    st.metric("Total Reviews", len(user_reviews_df))
with col3:
    avg_rating = round(sum(user_ratings_df) / len(user_ratings_df), 2) if user_ratings_df else "N/A"
    st.metric("Avg Rating Given", avg_rating)

st.markdown(f"<div style='color:#888;font-size:13px;margin-bottom:10px'>User ID: <code>{user_id}</code></div>", unsafe_allow_html=True)

if user_reviews_df:
    with st.expander(f"📝 View this user's past reviews ({len(user_reviews_df)} total)"):
        for i, review in enumerate(user_reviews_df[:5]):
            st.markdown(f"<div class='review-box'>{review[:250]}{'...' if len(review) > 250 else ''}</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Main Button ---
if st.button("🔍 Generate Counterfactual Explanation", type="primary", use_container_width=True):

    with st.spinner("Running counterfactual search across preference space..."):
        result = generate_counterfactual(
            model, user_idx, idx2product,
            n_steps=1000,
            step_size=step_size
        )

    if result:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # --- Product Comparison ---
        col_orig, col_arrow, col_cf = st.columns([5, 1, 5])

        with col_orig:
            orig_info = get_product_info(result["original_asin"])
            st.markdown("<div class='section-header'>🎯 Current Recommendation</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='product-card'>
                <div style='font-size:16px;font-weight:600;color:#ffffff;margin-bottom:10px'>{orig_info['title']}</div>
                <div style='margin-bottom:8px'><span class='asin-badge'>ASIN: {result['original_asin']}</span></div>
                {'<div style="color:#ff9900;font-size:13px">⭐ ' + str(orig_info['avg_rating']) + ' (' + str(orig_info['rating_count']) + ' reviews)</div>' if orig_info['avg_rating'] else ''}
                <div style='font-size:28px;font-weight:700;color:#ff9900;margin:10px 0'>{result['original_score']}</div>
                <div style='color:#888;font-size:13px'>Confidence Score</div>
                <div style='margin-top:12px'>
                    <a href='https://www.amazon.com/dp/{result["original_asin"]}' target='_blank'
                       style='color:#ff9900;text-decoration:none;font-size:13px'>
                       🔗 View on Amazon →
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_arrow:
            st.markdown("<div style='text-align:center;margin-top:80px;font-size:36px'>→</div>", unsafe_allow_html=True)

        with col_cf:
            cf_info = get_product_info(result["counterfactual_asin"])
            st.markdown("<div class='section-header'>🔄 Counterfactual Recommendation</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='product-card' style='border-color:#00d26a33'>
                <div style='font-size:16px;font-weight:600;color:#ffffff;margin-bottom:10px'>{cf_info['title']}</div>
                <div style='margin-bottom:8px'><span class='asin-badge'>ASIN: {result['counterfactual_asin']}</span></div>
                {'<div style="color:#00d26a;font-size:13px">⭐ ' + str(cf_info['avg_rating']) + ' (' + str(cf_info['rating_count']) + ' reviews)</div>' if cf_info['avg_rating'] else ''}
                <div style='font-size:28px;font-weight:700;color:#00d26a;margin:10px 0'>{result['new_score']}</div>
                <div style='color:#888;font-size:13px'>Score After Preference Shift</div>
                <div style='margin-top:12px'>
                    <a href='https://www.amazon.com/dp/{result["counterfactual_asin"]}' target='_blank'
                       style='color:#00d26a;text-decoration:none;font-size:13px'>
                       🔗 View on Amazon →
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- Stability Metrics ---
        st.markdown("### 📈 Recommendation Stability Analysis")
        label, css_class, icon = stability_label(result["perturbation_magnitude"])

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Perturbation Magnitude", result["perturbation_magnitude"],
                      help="Minimum preference shift needed to flip the recommendation")
        with col_m2:
            st.metric("Steps to Flip", result["steps_taken"],
                      help="Number of gradient steps before recommendation changed")
        with col_m3:
            st.metric("Original Score", result["original_score"])
        with col_m4:
            st.metric("Counterfactual Score", result["new_score"])

        score_gap = round(result["original_score"] - result["new_score"], 4)
        st.markdown(f"""
        <div style='background:#1a1f2e;border-radius:8px;padding:15px;margin:10px 0'>
            <span style='font-size:15px'>{icon} Stability: <strong class='{css_class}'>{label}</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp; Score gap between products: <strong>{score_gap}</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp; Flip found in <strong>{result['steps_taken']} steps</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Stability bar
        bar_val = min(result["perturbation_magnitude"] / 0.1, 1.0)
        st.progress(bar_val, text=f"Preference stability: {int(bar_val * 100)}%")

        st.markdown("---")

        # --- LLM Explanation ---
        st.markdown("### 💬 What This Means For You")
        with st.spinner("Generating personalized explanation..."):
            explanation = generate_explanation(result, user_reviews_df)

        st.markdown(f"<div class='insight-box'>{explanation}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # --- Research Insight ---
        st.markdown("### 🔬 Research Insight")
        if result["perturbation_magnitude"] < 0.005:
            insight = "This user's recommendation is **highly unstable** — sitting at a decision boundary. In production systems, such users would benefit from diversity injection rather than a single confident recommendation."
        elif result["perturbation_magnitude"] < 0.02:
            insight = "This user shows **moderate preference clarity**. The recommendation is reasonably stable but could shift with a few new interactions — ideal candidate for A/B testing alternative recommendations."
        else:
            insight = "This user has **strong, stable preferences**. The recommendation system is highly confident. Counterfactual distance suggests this user has consistent historical behavior that anchors the model firmly."

        st.info(insight)

        st.markdown("---")
        st.caption("🔬 ReviewLens · ALS Collaborative Filtering · Counterfactual Explanation Engine · Groq LLaMA3 · Amazon Reviews 2023")

    else:
        st.warning("No counterfactual found for this user. Try adjusting precision or selecting a different user.")