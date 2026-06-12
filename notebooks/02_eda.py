"""
ReviewLens — Exploratory Data Analysis
Generates all figures saved to notebooks/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scipy.sparse as sp
import pickle
import os

os.makedirs("notebooks/figures", exist_ok=True)
plt.rcParams['figure.facecolor'] = '#0f1117'
plt.rcParams['axes.facecolor'] = '#1a1f2e'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

print("Loading data...")
df = pd.read_parquet("data/processed_reviews.parquet")
matrix = sp.load_npz("data/interaction_matrix.npz")

with open("data/idx2product.pkl", "rb") as f:
    idx2product = pickle.load(f)

print(f"Reviews: {len(df):,}")
print(f"Users: {df['user_id'].nunique():,}")
print(f"Products: {df['parent_asin'].nunique():,}")
print(f"Matrix shape: {matrix.shape}")
print(f"Sparsity: {100*(1 - matrix.nnz/(matrix.shape[0]*matrix.shape[1])):.2f}%")

# ── Figure 1: Rating Distribution ──────────────────────────────
print("\n[1/5] Rating distribution...")
fig, ax = plt.subplots(figsize=(10, 5))
rating_counts = df['rating'].value_counts().sort_index()
colors = ['#ff4444', '#ff8800', '#ffcc00', '#88cc00', '#ff9900']
bars = ax.bar(rating_counts.index.astype(str), rating_counts.values,
              color=colors, edgecolor='none', width=0.6)
ax.set_title('Rating Distribution — Musical Instruments Dataset',
             fontsize=14, pad=15, color='white')
ax.set_xlabel('Rating', fontsize=12)
ax.set_ylabel('Number of Reviews', fontsize=12)
for bar, val in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{val:,}', ha='center', fontsize=10, color='white')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#333')
plt.tight_layout()
plt.savefig('notebooks/figures/01_rating_distribution.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("     ✅ Saved")

# ── Figure 2: User Activity Distribution (Power Law) ───────────
print("[2/5] User activity distribution...")
user_counts = df.groupby('user_id').size().sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw distribution
ax = axes[0]
ax.hist(user_counts.values, bins=50, color='#ff9900',
        edgecolor='none', alpha=0.85)
ax.set_title('User Review Count Distribution', fontsize=13, pad=12)
ax.set_xlabel('Number of Reviews per User')
ax.set_ylabel('Number of Users')
ax.axvline(user_counts.median(), color='#00d26a', linestyle='--',
           linewidth=1.5, label=f'Median: {user_counts.median():.0f}')
ax.legend(facecolor='#1a1f2e', labelcolor='white')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#333')

# Log-log (power law)
ax = axes[1]
rank = np.arange(1, len(user_counts) + 1)
ax.loglog(rank, user_counts.values, color='#ff9900',
          linewidth=1.5, alpha=0.8)
ax.set_title('User Activity — Log-Log Scale (Power Law)', fontsize=13, pad=12)
ax.set_xlabel('User Rank')
ax.set_ylabel('Review Count')
ax.grid(True, alpha=0.2, color='#444')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#333')

plt.tight_layout()
plt.savefig('notebooks/figures/02_user_activity.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("     ✅ Saved")

# ── Figure 3: Product Popularity Curve ─────────────────────────
print("[3/5] Product popularity curve...")
product_counts = df.groupby('parent_asin').size().sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(product_counts.values, bins=50, color='#00d26a',
        edgecolor='none', alpha=0.85)
ax.set_title('Product Review Count Distribution', fontsize=13, pad=12)
ax.set_xlabel('Number of Reviews per Product')
ax.set_ylabel('Number of Products')
ax.axvline(product_counts.median(), color='#ff9900', linestyle='--',
           linewidth=1.5, label=f'Median: {product_counts.median():.0f}')
ax.legend(facecolor='#1a1f2e', labelcolor='white')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#333')

ax = axes[1]
top_n = 30
top_products = product_counts.head(top_n)
ax.barh(range(top_n), top_products.values,
        color='#00d26a', edgecolor='none', alpha=0.85)
ax.set_title(f'Top {top_n} Most Reviewed Products', fontsize=13, pad=12)
ax.set_xlabel('Number of Reviews')
ax.set_ylabel('Product Rank')
ax.invert_yaxis()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#333')

plt.tight_layout()
plt.savefig('notebooks/figures/03_product_popularity.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("     ✅ Saved")

# ── Figure 4: Interaction Matrix Sparsity ──────────────────────
print("[4/5] Sparsity visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sample of matrix
ax = axes[0]
sample_size = min(200, matrix.shape[0])
sample_matrix = matrix[:sample_size, :200].toarray()
im = ax.imshow(sample_matrix, aspect='auto',
               cmap='YlOrRd', interpolation='none')
ax.set_title(f'Interaction Matrix Sample\n(200 users × 200 products)',
             fontsize=13, pad=12)
ax.set_xlabel('Product Index')
ax.set_ylabel('User Index')
plt.colorbar(im, ax=ax, label='Rating')

# Sparsity stats
ax = axes[1]
ax.axis('off')
sparsity = 100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
stats = [
    ("Total Users", f"{matrix.shape[0]:,}"),
    ("Total Products", f"{matrix.shape[1]:,}"),
    ("Total Interactions", f"{matrix.nnz:,}"),
    ("Matrix Sparsity", f"{sparsity:.2f}%"),
    ("Avg Reviews/User", f"{matrix.nnz/matrix.shape[0]:.1f}"),
    ("Avg Reviews/Product", f"{matrix.nnz/matrix.shape[1]:.1f}"),
]
y = 0.85
for label, value in stats:
    ax.text(0.1, y, label, transform=ax.transAxes,
            fontsize=12, color='#888')
    ax.text(0.65, y, value, transform=ax.transAxes,
            fontsize=12, color='#ff9900', fontweight='bold')
    y -= 0.13
ax.set_title('Dataset Statistics', fontsize=13, pad=12)

plt.tight_layout()
plt.savefig('notebooks/figures/04_sparsity.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("     ✅ Saved")

# ── Figure 5: Rating over Time ──────────────────────────────────
print("[5/5] Rating trends over time...")
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df['year_month'] = df['date'].dt.to_period('M')
monthly = df.groupby('year_month').agg(
    avg_rating=('rating', 'mean'),
    review_count=('rating', 'count')
).reset_index()
monthly['year_month_str'] = monthly['year_month'].astype(str)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax = axes[0]
ax.plot(range(len(monthly)), monthly['avg_rating'],
        color='#ff9900', linewidth=1.5)
ax.fill_between(range(len(monthly)), monthly['avg_rating'],
                alpha=0.2, color='#ff9900')
ax.set_title('Average Rating Over Time', fontsize=13, pad=12)
ax.set_ylabel('Average Rating')
ax.set_ylim(1, 5)
ax.axhline(monthly['avg_rating'].mean(), color='#00d26a',
           linestyle='--', linewidth=1, alpha=0.7,
           label=f"Overall avg: {monthly['avg_rating'].mean():.2f}")
ax.legend(facecolor='#1a1f2e', labelcolor='white')
tick_step = max(1, len(monthly) // 10)
ax.set_xticks(range(0, len(monthly), tick_step))
ax.set_xticklabels(monthly['year_month_str'].iloc[::tick_step],
                   rotation=45, ha='right', fontsize=8)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#333')

ax = axes[1]
ax.bar(range(len(monthly)), monthly['review_count'],
       color='#2d3561', edgecolor='none', alpha=0.85)
ax.set_title('Review Volume Over Time', fontsize=13, pad=12)
ax.set_ylabel('Number of Reviews')
ax.set_xlabel('Time')
ax.set_xticks(range(0, len(monthly), tick_step))
ax.set_xticklabels(monthly['year_month_str'].iloc[::tick_step],
                   rotation=45, ha='right', fontsize=8)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#333')

plt.tight_layout()
plt.savefig('notebooks/figures/05_rating_trends.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("     ✅ Saved")

print("\n" + "=" * 50)
print("✅ EDA complete! All figures saved to notebooks/figures/")
print("=" * 50)