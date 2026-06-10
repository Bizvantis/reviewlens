import streamlit as st
import scipy.sparse as sp
import pickle
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.counterfactual import generate_counterfactual, load_model_and_artifacts
from src.explainer import generate_explanation
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="ReviewLens",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 ReviewLens")
st.markdown("#### *Why was this recommended to you — and what would change it?*")
st.markdown("---")

# --- Load artifacts once ---
@st.cache_resource
def load_all():
    model, idx2product, product2idx, matrix = load_model_and_artifacts()
    df = pd.read_parquet("data/processed_reviews.parquet")
    with open("data/idx2user.pkl", "rb") as f:
        idx2user = pickle.load(f)
    with open("data/user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)
    return model, idx2product, product2idx, matrix, df, idx2user, user2idx

model, idx2product, product2idx, matrix, df, idx2user, user2idx = load_all()

# --- User Input ---
st.markdown("### Select a User")
max_user = matrix.shape[0] - 1
user_idx = st.slider("User Index", min_value=0, max_value=min(200, max_user), value=0)

user_id = idx2user[user_idx]
user_reviews_df = df[df["user_id"] == user_id]["text"].dropna().tolist()

st.markdown(f"**User ID:** `{user_id}`")
st.markdown(f"**Past reviews:** {len(user_reviews_df)}")

if user_reviews_df:
    with st.expander("See this user's past reviews"):
        for r in user_reviews_df[:5]:
            st.markdown(f"> {r[:200]}...")

st.markdown("---")

# --- Generate Counterfactual ---
if st.button("🔍 Explain My Recommendation", type="primary"):
    with st.spinner("Analyzing your preferences..."):
        result = generate_counterfactual(
            model, user_idx, idx2product,
            n_steps=500, step_size=0.005
        )

    if result:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Current Recommendation")
            st.code(result["original_asin"])
            st.metric("Confidence Score", result["original_score"])

        with col2:
            st.markdown("### 🔄 Counterfactual Recommendation")
            st.code(result["counterfactual_asin"])
            st.metric("New Score", result["new_score"])

        st.markdown("---")
        st.markdown("### 📊 Preference Stability")
        st.metric(
            "Perturbation Needed to Flip Recommendation",
            f"{result['perturbation_magnitude']}",
            help="Smaller = less stable recommendation"
        )
        st.progress(min(result["perturbation_magnitude"] * 10, 1.0))

        st.markdown("---")
        st.markdown("### 💬 What This Means For You")

        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(result, user_reviews_df)

        st.info(explanation)

        st.markdown("---")
        st.caption("Built with ALS Matrix Factorization + Counterfactual Search + Gemini Flash")

    else:
        st.warning("Could not find a counterfactual for this user. Try a different user index.")