import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.title("Mall Customer Segmentation")
st.write("K-Means clustering se customers ko spending aur income ke basis par groups mein baanto.")

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Genre": "Gender"})
    required = ["Annual Income (k$)", "Spending Score (1-100)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV mein required columns nahi mile: {missing}")
    df = df.dropna(subset=required).copy()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"CSV load nahi hui: {e}")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)
st.write(f"Total customers: **{len(df)}**")

k = st.slider("Kitne customer groups (clusters) chahiye?", 2, 10, 5)

features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = StandardScaler().fit_transform(df[features])
model = KMeans(n_clusters=k, n_init=10, random_state=42)
df["Cluster"] = model.fit_predict(X)

score = silhouette_score(X, df["Cluster"])
st.metric("Silhouette score", f"{score:.3f}")

st.subheader("Customer clusters")
fig, ax = plt.subplots(figsize=(9, 5))
for cluster in sorted(df["Cluster"].unique()):
    part = df[df["Cluster"] == cluster]
    ax.scatter(
        part["Annual Income (k$)"],
        part["Spending Score (1-100)"],
        label=f"Cluster {cluster}",
        alpha=0.8,
    )
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.legend()
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.subheader("Segment profiles")
profile = df.groupby("Cluster").agg(
    Customers=("Cluster", "size"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
).round(1)
st.dataframe(profile, use_container_width=True)

st.subheader("Customers with assigned clusters")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download customer segments CSV",
    data=csv,
    file_name="customers_with_segments.csv",
    mime="text/csv",
)
