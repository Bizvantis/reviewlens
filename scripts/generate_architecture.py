"""
Generates the ReviewLens architecture diagram.
Saved to docs/architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs("docs", exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(18, 10))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')

def box(ax, x, y, w, h, label, sublabel, color, text_color='white'):
    fancy = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.1",
                           facecolor=color,
                           edgecolor='#ffffff22',
                           linewidth=1.5)
    ax.add_patch(fancy)
    ax.text(x + w/2, y + h*0.62, label,
            ha='center', va='center',
            color=text_color, fontsize=11,
            fontweight='bold')
    ax.text(x + w/2, y + h*0.28, sublabel,
            ha='center', va='center',
            color='#aaaaaa', fontsize=8.5)

def arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#ff9900',
                                lw=2.0, connectionstyle='arc3,rad=0.0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.18, label, ha='center',
                color='#ff9900', fontsize=8)

# --- Title ---
ax.text(9, 9.4, 'ReviewLens — System Architecture',
        ha='center', va='center',
        color='white', fontsize=16, fontweight='bold')
ax.text(9, 9.0, 'Counterfactual Explanation Engine for Amazon Product Recommendations',
        ha='center', va='center',
        color='#888', fontsize=10)

# --- Row 1: Data Layer ---
ax.text(0.3, 8.3, 'DATA LAYER', color='#555', fontsize=8, fontweight='bold')
box(ax, 0.3, 7.2, 2.8, 0.9, 'Amazon Reviews 2023', 'Musical Instruments · 3M reviews', '#1a2a4a')
box(ax, 3.5, 7.2, 2.8, 0.9, 'Product Metadata', '213K products · titles · ratings', '#1a2a4a')
box(ax, 6.7, 7.2, 2.8, 0.9, 'Preprocessing', 'Filter · Encode · Sparse Matrix', '#1a2a4a')

# --- Row 2: Model Layer ---
ax.text(0.3, 6.5, 'MODEL LAYER', color='#555', fontsize=8, fontweight='bold')
box(ax, 0.3, 5.4, 2.8, 0.9, 'ALS Model', 'Implicit · 64 factors · 30 epochs', '#2a1a4a')
box(ax, 3.5, 5.4, 2.8, 0.9, 'User Embeddings', '210K users × 64 dims', '#2a1a4a')
box(ax, 6.7, 5.4, 2.8, 0.9, 'Item Embeddings', '22K products × 64 dims', '#2a1a4a')

# --- Row 3: Engine Layer ---
ax.text(0.3, 4.7, 'COUNTERFACTUAL ENGINE', color='#555', fontsize=8, fontweight='bold')
box(ax, 0.3, 3.6, 2.8, 0.9, 'Preference Vector', 'User embedding lookup', '#4a2a1a')
box(ax, 3.5, 3.6, 2.8, 0.9, 'Gradient Perturbation', 'Minimal preference shift search', '#4a2a1a')
box(ax, 6.7, 3.6, 2.8, 0.9, 'Flip Detection', 'Top-1 recommendation change', '#4a2a1a')

# --- Row 4: Output Layer ---
ax.text(0.3, 2.9, 'OUTPUT LAYER', color='#555', fontsize=8, fontweight='bold')
box(ax, 0.3, 1.8, 2.8, 0.9, 'LLM Explanation', 'Groq LLaMA3 · plain English', '#1a4a2a')
box(ax, 3.5, 1.8, 2.8, 0.9, 'Stability Score', 'Perturbation magnitude metric', '#1a4a2a')
box(ax, 6.7, 1.8, 2.8, 0.9, 'Product Lookup', 'Real titles · ratings · links', '#1a4a2a')

# --- Row 5: Interface Layer ---
ax.text(0.3, 1.1, 'INTERFACE LAYER', color='#555', fontsize=8, fontweight='bold')
box(ax, 0.3, 0.1, 2.8, 0.8, 'Streamlit App', 'Interactive demo · localhost:8501', '#2a3a1a')
box(ax, 3.5, 0.1, 2.8, 0.8, 'FastAPI', 'REST API · /docs · localhost:8000', '#2a3a1a')
box(ax, 6.7, 0.1, 2.8, 0.8, 'Evaluation', 'Precision · Recall · NDCG@10', '#2a3a1a')

# --- Right side: Evaluation box ---
box(ax, 10.5, 5.4, 3.5, 3.5, '', '', '#111827')
ax.text(12.25, 8.6, '📊 Model Evaluation', ha='center',
        color='white', fontsize=11, fontweight='bold')
metrics = [
    ('Precision@10', '0.0192'),
    ('Recall@10', '0.1432'),
    ('NDCG@10', '0.0959'),
    ('vs Random', '0.0000'),
    ('vs Popularity', '0.0000'),
    ('Sparsity', '99.98%'),
]
for i, (k, v) in enumerate(metrics):
    y_pos = 8.1 - i * 0.48
    ax.text(10.8, y_pos, k, color='#888', fontsize=9)
    ax.text(13.7, y_pos, v, color='#ff9900', fontsize=9,
            fontweight='bold', ha='right')

# --- Right side: Stack box ---
box(ax, 10.5, 1.8, 3.5, 3.3, '', '', '#111827')
ax.text(12.25, 4.85, '🛠 Tech Stack', ha='center',
        color='white', fontsize=11, fontweight='bold')
stack = [
    ('Model', 'implicit ALS'),
    ('Dataset', 'McAuley Amazon 2023'),
    ('LLM', 'Groq LLaMA3'),
    ('App', 'Streamlit'),
    ('API', 'FastAPI + Uvicorn'),
    ('Language', 'Python 3.10'),
]
for i, (k, v) in enumerate(stack):
    y_pos = 4.4 - i * 0.45
    ax.text(10.8, y_pos, k, color='#888', fontsize=9)
    ax.text(13.7, y_pos, v, color='#00d26a', fontsize=9,
            fontweight='bold', ha='right')

# --- Arrows ---
arrow(ax, 1.7, 7.2, 1.7, 6.3)
arrow(ax, 4.9, 7.2, 4.9, 6.3)
arrow(ax, 8.1, 7.2, 8.1, 6.3)
arrow(ax, 1.7, 5.4, 1.7, 4.5)
arrow(ax, 4.9, 5.4, 4.9, 4.5)
arrow(ax, 8.1, 5.4, 8.1, 4.5)
arrow(ax, 1.7, 3.6, 1.7, 2.7)
arrow(ax, 4.9, 3.6, 4.9, 2.7)
arrow(ax, 8.1, 3.6, 8.1, 2.7)
arrow(ax, 1.7, 1.8, 1.7, 0.9)
arrow(ax, 4.9, 1.8, 4.9, 0.9)
arrow(ax, 8.1, 1.8, 8.1, 0.9)

plt.tight_layout()
plt.savefig('docs/architecture.png', dpi=180,
            bbox_inches='tight', facecolor='#0f1117')
print("✅ Architecture diagram saved to docs/architecture.png")
plt.close()