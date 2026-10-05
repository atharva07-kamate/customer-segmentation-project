import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: #172554;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Genre": "Gender"})

    required = ["Annual Income (k$)", "Spending Score (1-100)"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV mein required columns nahi mile: {missing}")

    df = df.dropna(subset=required).copy()
    return df


try:
    original_df = load_data()
except Exception as error:
    st.error(f"Dataset load nahi hua: {error}")
    st.stop()


INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"
features = [INCOME, SPENDING]

st.sidebar.title("🛍️ Customer Insights")
st.sidebar.caption("Mall Customer Segmentation")
page = st.sidebar.radio(
    "Menu",
    ["Dashboard", "Customer Segments", "Explore Data", "Model & Method"]
)
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
st.sidebar.info("Clustering features: annual income and spending score")


df = original_df.copy()
X = StandardScaler().fit_transform(df[features])
model = KMeans(n_clusters=k, n_init=10, random_state=42)
df["Cluster"] = model.fit_predict(X)
silhouette = silhouette_score(X, df["Cluster"])

centers = df.groupby("Cluster").agg(
    Customers=("Cluster", "size"),
    Avg_Income=(INCOME, "mean"),
    Avg_Spending=(SPENDING, "mean"),
)

median_income = df[INCOME].median()
median_spending = df[SPENDING].median()


def segment_name(row):
    high_income = row["Avg_Income"] >= median_income
    high_spending = row["Avg_Spending"] >= median_spending

    if high_income and high_spending:
        return "Premium Spenders"
    if high_income and not high_spending:
        return "Cautious Affluent"
    if not high_income and high_spending:
        return "Enthusiastic Budget Spenders"
    return "Budget-Conscious"


centers["Segment"] = centers.apply(segment_name, axis=1)
segment_names = centers["Segment"].to_dict()
df["Segment"] = df["Cluster"].map(segment_names)


st.title("🛍️ Mall Customer Segmentation")
st.caption(
    "K-Means clustering se customers ke income aur spending patterns samjho."
)


if page == "Dashboard":
    st.subheader("Project Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total customers", f"{len(df)}")
    col2.metric("Customer groups", f"{k}")
    col3.metric("Silhouette score", f"{silhouette:.3f}")

    st.markdown("### Income vs spending clusters")
    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster in sorted(df["Cluster"].unique()):
        group = df[df["Cluster"] == cluster]
        ax.scatter(
            group[INCOME],
            group[SPENDING],
            label=f"Cluster {cluster}: {segment_names[cluster]}",
            alpha=0.8,
            s=55
        )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1–100)")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Segment summary")
    st.dataframe(
        centers.reset_index().rename(columns={"index": "Cluster"}),
        use_container_width=True,
        hide_index=True
    )


elif page == "Customer Segments":
    st.subheader("Customer Segments")
    st.write(
        "Cluster numbers machine ke labels hain. Segment names income aur "
        "spending averages ko dataset medians se compare karke diye gaye hain."
    )

    for cluster, row in centers.iterrows():
        with st.container(border=True):
            st.markdown(f"### Cluster {cluster}: {row['Segment']}")
            a, b, c = st.columns(3)
            a.metric("Customers", int(row["Customers"]))
            b.metric("Average income", f"${row['Avg_Income']:.1f}k")
            c.metric("Average spending score", f"{row['Avg_Spending']:.1f}")

    st.markdown("### Business interpretation")
    st.write("- **Premium Spenders:** loyalty rewards aur premium offers.")
    st.write("- **Cautious Affluent:** quality-focused offers aur product previews.")
    st.write("- **Enthusiastic Budget Spenders:** bundles aur value promotions.")
    st.write("- **Budget-Conscious:** low-cost offers aur essentials.")


elif page == "Explore Data":
    st.subheader("Explore the customer data")

    chart_cols = st.columns(3)
    for col, feature, title in zip(
        chart_cols,
        ["Age", INCOME, SPENDING],
        ["Age distribution", "Annual income distribution", "Spending score distribution"]
    ):
        if feature in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(df[feature].dropna(), bins=15, color="#4C78A8", edgecolor="white")
            ax.set_title(title)
            ax.set_ylabel("Customers")
            col.pyplot(fig)
            plt.close(fig)

    st.markdown("### Customer records with assigned segments")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download segmented customers CSV",
        data=csv,
        file_name="customers_with_segments.csv",
        mime="text/csv"
    )


elif page == "Model & Method":
    st.subheader("Model evaluation")
    st.write(
        "Har k ke liye K-Means dobara fit hota hai. Silhouette zyada hona "
        "aur Davies–Bouldin kam hona aam taur par behtar separated clusters "
        "ka sanket hai. In scores ko business context ke saath interpret karo."
    )

    evaluation_rows = []
    for test_k in range(2, min(10, len(df) - 1) + 1):
        test_model = KMeans(n_clusters=test_k, n_init=10, random_state=42)
        test_labels = test_model.fit_predict(X)
        evaluation_rows.append({
            "k": test_k,
            "Inertia": test_model.inertia_,
            "Silhouette": silhouette_score(X, test_labels),
            "Davies-Bouldin": davies_bouldin_score(X, test_labels),
        })

    evaluation = pd.DataFrame(evaluation_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(evaluation["k"], evaluation["Inertia"], "o-")
    axes[0].set_title("Elbow: Inertia (lower)")
    axes[1].plot(evaluation["k"], evaluation["Silhouette"], "o-", color="green")
    axes[1].set_title("Silhouette (higher)")
    axes[2].plot(evaluation["k"], evaluation["Davies-Bouldin"], "o-", color="crimson")
    axes[2].set_title("Davies–Bouldin (lower)")
    for ax in axes:
        ax.set_xlabel("Number of clusters (k)")
        ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### PCA projection")
    projection = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=df["Cluster"],
        cmap="tab10",
        s=45,
        alpha=0.85
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Customer clusters projected into 2D")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Evaluation values")
    st.dataframe(evaluation.round(3), use_container_width=True, hide_index=True)

    st.markdown("### Method used")
    st.write("1. CSV load karke required columns check kiye.")
    st.write("2. Income aur spending score ko StandardScaler se scale kiya.")
    st.write("3. K-Means se selected k customer groups banaye.")
    st.write("4. Silhouette score aur other metrics se clustering evaluate ki.")
