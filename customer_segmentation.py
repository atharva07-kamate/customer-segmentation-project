"""
Customer Segmentation System
----------------------------
Dataset : Mall Customer Segmentation (Kaggle, vjchoudhary7)
Usage   : python customer_segmentation.py
          (reads Mall_Customers.csv next to this script, writes to results/)
          optional: --data <csv path> --out <folder> --k <clusters>

Steps
  1. Load, clean and explore the data
  2. Scale features
  3. Find the optimal number of clusters (elbow, silhouette, Davies-Bouldin)
  4. Fit K-Means and compare with Hierarchical and Gaussian Mixture
  5. Visualise (2-D scatter and PCA projection)
  6. Profile and name each segment, with business actions
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # works without a display; remove if you want pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
INCOME, SPEND, AGE = "Annual Income (k$)", "Spending Score (1-100)", "Age"
REQUIRED_COLUMNS = [AGE, INCOME, SPEND, "Gender"]


class DataError(Exception):
    """Raised for problems with the input data (shown as a simple message)."""


# ----------------------------------------------------------------------------
# 1. Load + preprocess
# ----------------------------------------------------------------------------
def load_data(path):
    if not os.path.isfile(path):
        raise DataError(
            f"Dataset file not found: {path}\n"
            "Put 'Mall_Customers.csv' in the same folder as this script, "
            "or pass its location with --data <path>.")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataError(f"Could not read the CSV file ({e}).")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Genre": "Gender"})  # some Kaggle copies use "Genre"

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataError(
            f"The CSV is missing required column(s): {missing_cols}\n"
            f"Columns found: {list(df.columns)}\n"
            f"Columns needed: {REQUIRED_COLUMNS}")

    for col in (AGE, INCOME, SPEND):  # make sure numeric columns are numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("Duplicate rows:", df.duplicated().sum())
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[AGE, INCOME, SPEND]).copy()  # numeric features are essential
    df["Gender"] = df["Gender"].fillna("Unknown")
    print(f"Rows removed during cleaning: {n0 - len(df)}")
    if len(df) < 10:
        raise DataError("Too few valid rows (need at least 10) after cleaning the data.")
    df = df.reset_index(drop=True)
    print("\nSummary statistics:\n", df.describe().round(2))
    return df


def explore(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, [AGE, INCOME, SPEND]):
        ax.hist(df[col], bins=20, color="#4C78A8", edgecolor="white")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "01_distributions.png"), dpi=130)
    plt.close()


def preprocess(df, features):
    """Scale features so Age (18-70) doesn't dominate Income (15-137) etc."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    return X, scaler


# ----------------------------------------------------------------------------
# 2. Choose k
# ----------------------------------------------------------------------------
def evaluate_k(X, out, k_range=range(2, 11)):
    rows = []
    k_range = [k for k in k_range if k < len(X)]
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
        })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "k_evaluation.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(res.k, res.inertia, "o-")
    axes[0].set(title="Elbow method (inertia, lower is better)", xlabel="k")
    axes[1].plot(res.k, res.silhouette, "o-", color="green")
    axes[1].set(title="Silhouette (higher is better)", xlabel="k")
    axes[2].plot(res.k, res.davies_bouldin, "o-", color="crimson")
    axes[2].set(title="Davies-Bouldin (lower is better)", xlabel="k")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "02_optimal_k.png"), dpi=130)
    plt.close()

    print("\nCluster-count evaluation:\n", res.round(3).to_string(index=False))
    return res


def pick_k(res):
    """Best silhouette wins; the elbow plot is a visual cross-check."""
    return int(res.loc[res.silhouette.idxmax(), "k"])


# ----------------------------------------------------------------------------
# 3. Cluster
# ----------------------------------------------------------------------------
def compare_algorithms(X, k):
    models = {
        "K-Means": KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE),
        "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "Gaussian Mixture": GaussianMixture(n_components=k, random_state=RANDOM_STATE),
    }
    rows, labels = [], {}
    for name, m in models.items():
        lab = m.fit_predict(X)
        labels[name] = lab
        rows.append({"algorithm": name,
                     "silhouette": silhouette_score(X, lab),
                     "davies_bouldin": davies_bouldin_score(X, lab)})
    comp = pd.DataFrame(rows)
    print("\nAlgorithm comparison:\n", comp.round(3).to_string(index=False))
    return labels, comp


# ----------------------------------------------------------------------------
# 4. Visualise
# ----------------------------------------------------------------------------
def plot_segments(df, X, labels, out):
    # Cluster centers in original units (mean Income / Spending of each cluster)
    centers = df.assign(_c=labels).groupby("_c")[[INCOME, SPEND]].mean()
    palette = plt.cm.tab10(np.arange(labels.max() + 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(labels.max() + 1):
        m = labels == c
        ax.scatter(df.loc[m, INCOME], df.loc[m, SPEND], s=55, color=palette[c],
                   label=f"Cluster {c}", edgecolor="white", linewidth=0.5)
    ax.scatter(centers[INCOME], centers[SPEND], marker="X", s=260, color="black",
               edgecolor="yellow", linewidth=1.5, label="Cluster centers", zorder=5)
    for c, r in centers.iterrows():
        ax.annotate(str(c), (r[INCOME], r[SPEND]), textcoords="offset points",
                    xytext=(10, 10), fontsize=11, fontweight="bold")
    ax.set(xlabel=INCOME, ylabel=SPEND, title="Customer Segments: Income vs Spending Score")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "03_income_vs_spending.png"), dpi=130)
    plt.close()

    # PCA gives a 2-D view of all scaled features at once
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    P = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(labels.max() + 1):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=55, color=palette[c], label=f"Cluster {c}")
    ax.set(title=f"PCA projection ({pca.explained_variance_ratio_.sum():.0%} variance)",
           xlabel="PC1", ylabel="PC2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "04_pca_projection.png"), dpi=130)
    plt.close()


# ----------------------------------------------------------------------------
# 5. Interpret
# ----------------------------------------------------------------------------
def name_segment(row, med):
    """Rule-based label from income/spending relative to the dataset median."""
    hi_inc, hi_spd = row[INCOME] >= med[INCOME], row[SPEND] >= med[SPEND]
    young = row[AGE] < med[AGE]
    if abs(row[INCOME] - med[INCOME]) <= 15 and abs(row[SPEND] - med[SPEND]) <= 15:
        return "Mainstream Shoppers"
    if hi_inc and hi_spd:
        return "Premium Spenders (VIPs)"
    if hi_inc and not hi_spd:
        return "Cautious Affluent"
    if not hi_inc and hi_spd:
        return "Young Trend Seekers" if young else "Enthusiastic Budget Spenders"
    return "Budget-Conscious"


ACTIONS = {
    "Mainstream Shoppers": "Largest 'average' group: seasonal promos, upsell to premium, nudge toward loyalty.",
    "Premium Spenders (VIPs)": "Loyalty tiers, early access, personal shopping, premium cross-sell.",
    "Cautious Affluent": "Win them over with quality/value messaging, exclusive previews, targeted offers.",
    "Young Trend Seekers": "Social-media campaigns, new arrivals, influencer tie-ins, installment/BNPL deals.",
    "Enthusiastic Budget Spenders": "Discount events, bundles, points programmes to keep spend sustainable.",
    "Budget-Conscious": "Low-cost promotions and essentials; avoid heavy marketing spend.",
}


INTERPRETATION = {
    "Mainstream Shoppers": "Mid income and mid spending: the stable core of the customer base.",
    "Premium Spenders (VIPs)": "High income and high spending: the most valuable customers.",
    "Cautious Affluent": "High income but low spending: untapped potential, not yet convinced.",
    "Young Trend Seekers": "Lower income but high spending, younger: impulsive, trend-driven buyers.",
    "Enthusiastic Budget Spenders": "Lower income but high spending: loyal, but spending may be unsustainable.",
    "Budget-Conscious": "Low income and low spending: price-sensitive, low revenue per customer.",
}


def profile_segments(df, labels, out):
    df = df.copy()
    df["Cluster"] = labels
    med = df[[AGE, INCOME, SPEND]].median()

    prof = df.groupby("Cluster").agg(
        customers=("Cluster", "size"),
        avg_age=(AGE, "mean"),
        avg_income=(INCOME, "mean"),
        avg_spending=(SPEND, "mean"),
        pct_female=("Gender", lambda s: (s == "Female").mean() * 100),
    ).round(1)
    prof = prof.rename(columns={"customers": "customer_count"})
    prof["share_%"] = (prof.customer_count / len(df) * 100).round(1)

    names = {c: name_segment(df[df.Cluster == c][[AGE, INCOME, SPEND]].mean(), med)
             for c in prof.index}
    # If two clusters get the same name, suffix them so they stay distinguishable
    seen = {}
    for c, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} #{seen[n]}"
    prof.insert(0, "segment", prof.index.map(names))
    base = [n.split(" #")[0] for n in prof.segment]
    prof["business_interpretation"] = [INTERPRETATION.get(n, "") for n in base]
    prof["recommended_action"] = [ACTIONS.get(n, "") for n in base]

    print("\n" + "=" * 70 + "\nSEGMENT PROFILES\n" + "=" * 70)
    for c, r in prof.iterrows():
        print(f"\nCluster {c}: {r.segment}  ({r.customer_count} customers, {r['share_%']}%)")
        print(f"  Avg age {r.avg_age} | income ${r.avg_income}k | spending score {r.avg_spending}"
              f" | {r.pct_female}% female")
        print(f"  Meaning: {r.business_interpretation}")
        print(f"  Action: {r.recommended_action}")

    prof.index.name = "Cluster"
    prof.to_csv(os.path.join(out, "segment_profiles.csv"))
    df["Segment"] = df.Cluster.map(names)
    df.to_csv(os.path.join(out, "customers_with_segments.csv"), index=False)

    # Bar chart comparing segment averages
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, t in zip(axes, ["avg_age", "avg_income", "avg_spending"],
                          ["Avg age", "Avg annual income (k$)", "Avg spending score"]):
        ax.bar(prof.index.astype(str), prof[col], color=plt.cm.tab10(np.arange(len(prof))))
        ax.set(title=t, xlabel="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "05_segment_profiles.png"), dpi=130)
    plt.close()
    return df


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Customer Segmentation System (K-Means)")
    ap.add_argument("--data", default=os.path.join(BASE_DIR, "Mall_Customers.csv"),
                    help="path to the CSV (default: Mall_Customers.csv in the project folder)")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "results"),
                    help="output folder (default: results/ in the project folder)")
    ap.add_argument("--k", type=int, default=None, help="force number of clusters")
    ap.add_argument("--features", nargs="+", default=[INCOME, SPEND],
                    help="e.g. --features 'Annual Income (k$)' 'Spending Score (1-100)' Age")
    args = ap.parse_args()

    try:
        df = load_data(args.data)
        bad = [f for f in args.features if f not in df.columns]
        if bad:
            raise DataError(f"Unknown feature column(s): {bad}. Available: {list(df.columns)}")
        os.makedirs(args.out, exist_ok=True)
        explore(df, args.out)

        X, _ = preprocess(df, args.features)
        res = evaluate_k(X, args.out)
        k = args.k or pick_k(res)
        print(f"\nChosen k = {k}")

        labels_by_algo, comp = compare_algorithms(X, k)
        comp.to_csv(os.path.join(args.out, "algorithm_comparison.csv"), index=False)
        labels = labels_by_algo["K-Means"]  # final model

        plot_segments(df, X, labels, args.out)
        profile_segments(df, labels, args.out)
        print(f"\nAll outputs saved to: {args.out}")
    except DataError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
