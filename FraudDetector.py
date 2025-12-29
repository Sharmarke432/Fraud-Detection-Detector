import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Credit Card Fraud Detection – Unsupervised Anomaly Detection")
st.markdown(
    "Unsupervised fraud detection on the Kaggle credit card dataset using "
    "**Isolation Forest** and **Local Outlier Factor (LOF)**."
)

# -------------------------------------------------------------------
# 1. Sidebar – controls and data loading
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

contamination_pct = st.sidebar.slider(
    "Assumed fraud rate in data (%)",
    min_value=0.05,
    max_value=0.5,
    value=0.17,
    step=0.01,
)
contamination = contamination_pct / 100.0

n_trees = st.sidebar.slider(
    "Isolation Forest: number of trees",
    min_value=50,
    max_value=300,
    value=100,
    step=25,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Dataset")

st.sidebar.markdown(
    "- Download `creditcard.csv` from Kaggle\n"
    "- Place it in the **repo root** or upload below"
)

uploaded = st.sidebar.file_uploader("Or upload creditcard.csv", type=["csv"])


@st.cache_data(show_spinner=True)
def load_data(file_obj):
    if file_obj is not None:
        df_ = pd.read_csv(file_obj)
    else:
        df_ = pd.read_csv("creditcard.csv")
    return df_


# Try to load data
try:
    df = load_data(uploaded)
except Exception as e:
    st.error(
        "Could not load `creditcard.csv`.\n\n"
        "- Make sure it is in the same folder as `app.py`, **or**\n"
        "- Upload it using the sidebar.\n\n"
        f"Error: {e}"
    )
    st.stop()

# -------------------------------------------------------------------
# 2. Basic dataset info
# -------------------------------------------------------------------
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total transactions", f"{len(df):,}")
with col2:
    n_fraud = int(df["Class"].sum())
    st.metric("Fraud cases", f"{n_fraud:,}")
with col3:
    st.metric("Fraud rate", f"{df['Class'].mean():.3%}")
with col4:
    st.metric("Features", f"{df.shape[1] - 1} + target")

with st.expander("Preview data"):
    st.dataframe(df.head())

# Class distribution chart
class_counts = df["Class"].value_counts().rename(index={0: "Normal", 1: "Fraud"})
fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x=class_counts.index, y=class_counts.values, palette=["#4c72b0", "#dd8452"], ax=ax)
ax.set_title("Class distribution")
ax.set_ylabel("Count")
st.pyplot(fig)

st.markdown("---")

# -------------------------------------------------------------------
# 3. Train models (reuse your notebook logic in compact form)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def train_models(df, contamination, n_trees):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Isolation Forest
    iso = IsolationForest(
        n_estimators=n_trees,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)
    iso_scores = -iso.decision_function(X_test)
    iso_threshold = np.percentile(iso_scores, 100 * (1 - contamination))
    iso_pred = (iso_scores > iso_threshold).astype(int)

    iso_auprc = average_precision_score(y_test, iso_scores)
    iso_auc = roc_auc_score(y_test, iso_scores)
    iso_precision = precision_score(y_test, iso_pred, zero_division=0)
    iso_recall = recall_score(y_test, iso_pred, zero_division=0)
    iso_f1 = f1_score(y_test, iso_pred, zero_division=0)

    # LOF
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=True,
    )
    lof.fit(X_train)
    lof_scores = -lof.decision_function(X_test)
    lof_threshold = np.percentile(lof_scores, 100 * (1 - contamination))
    lof_pred = (lof_scores > lof_threshold).astype(int)

    lof_auprc = average_precision_score(y_test, lof_scores)
    lof_auc = roc_auc_score(y_test, lof_scores)
    lof_precision = precision_score(y_test, lof_pred, zero_division=0)
    lof_recall = recall_score(y_test, lof_pred, zero_division=0)
    lof_f1 = f1_score(y_test, lof_pred, zero_division=0)

    baseline_auprc = y_test.mean()

    results = {
        "X_test": X_test,
        "y_test": y_test,
        "iso_scores": iso_scores,
        "iso_pred": iso_pred,
        "iso_threshold": iso_threshold,
        "iso_auprc": iso_auprc,
        "iso_auc": iso_auc,
        "iso_precision": iso_precision,
        "iso_recall": iso_recall,
        "iso_f1": iso_f1,
        "lof_scores": lof_scores,
        "lof_pred": lof_pred,
        "lof_threshold": lof_threshold,
        "lof_auprc": lof_auprc,
        "lof_auc": lof_auc,
        "lof_precision": lof_precision,
        "lof_recall": lof_recall,
        "lof_f1": lof_f1,
        "baseline_auprc": baseline_auprc,
    }
    return results


with st.spinner("Training Isolation Forest and LOF..."):
    res = train_models(df, contamination, n_trees)

st.success("Models trained successfully.")

# -------------------------------------------------------------------
# 4. Metrics summary
# -------------------------------------------------------------------
st.subheader("📌 Model Performance")

metrics_df = pd.DataFrame(
    {
        "Metric": ["AUPRC", "AUC-ROC", "Precision", "Recall", "F1"],
        "Isolation Forest": [
            res["iso_auprc"],
            res["iso_auc"],
            res["iso_precision"],
            res["iso_recall"],
            res["iso_f1"],
        ],
        "LOF": [
            res["lof_auprc"],
            res["lof_auc"],
            res["lof_precision"],
            res["lof_recall"],
            res["lof_f1"],
        ],
    }
)

st.dataframe(
    metrics_df.style.format(
        {
            "Isolation Forest": "{:.4f}",
            "LOF": "{:.4f}",
        }
    ),
    use_container_width=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Baseline AUPRC (fraud rate)", f"{res['baseline_auprc']:.4f}")
with col2:
    st.metric("IF vs baseline (x)", f"{res['iso_auprc']/res['baseline_auprc']:.1f}x")
with col3:
    st.metric("LOF vs baseline (x)", f"{res['lof_auprc']/res['baseline_auprc']:.1f}x")

st.markdown("---")

# -------------------------------------------------------------------
# 5. Visualizations: PR, ROC, Score distributions
# -------------------------------------------------------------------
st.subheader("📉 Precision–Recall & ROC Curves")

y_test = res["y_test"]
iso_scores = res["iso_scores"]
lof_scores = res["lof_scores"]

# PR curves
iso_p, iso_r, _ = precision_recall_curve(y_test, iso_scores)
lof_p, lof_r, _ = precision_recall_curve(y_test, lof_scores)

fig_pr, ax = plt.subplots(figsize=(6, 4))
ax.plot(iso_r, iso_p, label=f"Isolation Forest (AUPRC={res['iso_auprc']:.3f})")
ax.plot(lof_r, lof_p, label=f"LOF (AUPRC={res['lof_auprc']:.3f})")
ax.hlines(
    y=res["baseline_auprc"],
    xmin=0,
    xmax=1,
    colors="red",
    linestyles="--",
    label=f"Baseline ({res['baseline_auprc']:.4f})",
)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curves")
ax.legend()
st.pyplot(fig_pr)

# ROC curves
iso_fpr, iso_tpr, _ = roc_curve(y_test, iso_scores)
lof_fpr, lof_tpr, _ = roc_curve(y_test, lof_scores)

fig_roc, ax = plt.subplots(figsize=(6, 4))
ax.plot(iso_fpr, iso_tpr, label=f"Isolation Forest (AUC={res['iso_auc']:.3f})")
ax.plot(lof_fpr, lof_tpr, label=f"LOF (AUC={res['lof_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend()
st.pyplot(fig_roc)

st.markdown("---")

st.subheader("📦 Anomaly Score Distributions")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

axes2[0].hist(iso_scores[y_test == 0], bins=50, alpha=0.7, label="Normal", color="#4c72b0")
axes2[0].hist(iso_scores[y_test == 1], bins=50, alpha=0.7, label="Fraud", color="#dd8452")
axes2[0].axvline(res["iso_threshold"], color="black", linestyle="--", label="Threshold")
axes2[0].set_title("Isolation Forest scores")
axes2[0].set_xlabel("Anomaly score")
axes2[0].legend()

axes2[1].hist(lof_scores[y_test == 0], bins=50, alpha=0.7, label="Normal", color="#55a868")
axes2[1].hist(lof_scores[y_test == 1], bins=50, alpha=0.7, label="Fraud", color="#c44e52")
axes2[1].axvline(res["lof_threshold"], color="black", linestyle="--", label="Threshold")
axes2[1].set_title("LOF scores")
axes2[1].set_xlabel("Anomaly score")
axes2[1].legend()

st.pyplot(fig2)

st.markdown("---")

# -------------------------------------------------------------------
# 6. PCA 2D view
# -------------------------------------------------------------------
st.subheader("🔍 PCA 2D – Detected anomalies")

# Sample for speed if huge
X_test_sample = res["X_test"]
y_sample = y_test
iso_pred = res["iso_pred"]
lof_pred = res["lof_pred"]

max_points = 10000
if len(X_test_sample) > max_points:
    idx = np.random.RandomState(42).choice(len(X_test_sample), size=max_points, replace=False)
    X_test_sample = X_test_sample.iloc[idx]
    y_sample = y_sample.iloc[idx]
    iso_pred = iso_pred[idx]
    lof_pred = lof_pred[idx]

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_test_sample)

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))

# Isolation Forest
axes3[0].scatter(
    X_pca[y_sample == 0, 0],
    X_pca[y_sample == 0, 1],
    c="#d0e1f9",
    s=5,
    label="Normal",
)
axes3[0].scatter(
    X_pca[iso_pred == 1, 0],
    X_pca[iso_pred == 1, 1],
    c="#f18f01",
    s=10,
    label="IF anomaly",
)
axes3[0].set_title("Isolation Forest – PCA 2D")
axes3[0].set_xlabel("PC1")
axes3[0].set_ylabel("PC2")
axes3[0].legend()

# LOF
axes3[1].scatter(
    X_pca[y_sample == 0, 0],
    X_pca[y_sample == 0, 1],
    c="#d9f2e6",
    s=5,
    label="Normal",
)
axes3[1].scatter(
    X_pca[lof_pred == 1, 0],
    X_pca[lof_pred == 1, 1],
    c="#9b59b6",
    s=10,
    label="LOF anomaly",
)
axes3[1].set_title("LOF – PCA 2D")
axes3[1].set_xlabel("PC1")
axes3[1].set_ylabel("PC2")
axes3[1].legend()

st.pyplot(fig3)

st.markdown("---")

# -------------------------------------------------------------------
# 7. Business impact section
# -------------------------------------------------------------------
st.subheader("💰 Business impact (Isolation Forest)")

iso_tp = int(((iso_pred == 1) & (y_sample == 1)).sum())
iso_fp = int(((iso_pred == 1) & (y_sample == 0)).sum())

assumed_loss_per_fraud = st.number_input(
    "Loss per undetected fraud (£)",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
)
investigation_cost = st.number_input(
    "Cost per investigated case (£)",
    min_value=10,
    max_value=500,
    value=125,
    step=5,
)

savings = iso_tp * assumed_loss_per_fraud
cost = iso_fp * investigation_cost
net = savings - cost

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Fraud caught (TP)", iso_tp)
with c2:
    st.metric("False alarms (FP)", iso_fp)
with c3:
    st.metric("Net savings", f"£{net:,.0f}")

st.markdown(
    "This is a **rough estimate** assuming every detected fraud would have been fully lost "
    f"and every false alarm costs £{investigation_cost} of analyst time."
)
