import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background-color: #111827;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #F3F4F6;
        }

        [data-testid="stMetric"] {
            background: #F8FAFC;
            color: #0F172A !important;
            border: 1px solid #E2E8F0;
            padding: 18px;
            border-radius: 12px;
        }

        [data-testid="stMetric"] *,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #0F172A !important;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"
FEATURES = [INCOME, SPENDING]

SEGMENT_ACTIONS = {
    "Premium Spenders": (
        "Consider premium product recommendations, exclusive previews, "
        "and loyalty benefits."
    ),
    "Cautious Affluent": (
        "Test quality-focused messaging, product education, and targeted offers."
    ),
    "Enthusiastic Budget Spenders": (
        "Consider value bundles, rewards, and promotions that fit their budgets."
    ),
    "Budget-Conscious": (
        "Focus on essential products, clear value, and cost-conscious offers."
    ),
}


@st.cache_data
def load_customer_data():
    data = pd.read_csv("Mall_Customers.csv")
    data.columns = data.columns.str.strip()
    data = data.rename(columns={"Genre": "Gender"})

    missing_columns = [column for column in FEATURES if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=FEATURES)
    data = data.drop_duplicates().reset_index(drop=True)

    if len(data) < 10:
        raise ValueError("At least 10 valid customer records are required.")

    return data


def assign_segment_names(profile, median_income, median_spending):
    """Create readable, unique names from each cluster's average profile."""
    names = {}
    used_names = {}

    for cluster, row in profile.iterrows():
        high_income = row["Average income (k$)"] >= median_income
        high_spending = row["Average spending score"] >= median_spending

        if high_income and high_spending:
            base_name = "Premium Spenders"
        elif high_income and not high_spending:
            base_name = "Cautious Affluent"
        elif not high_income and high_spending:
            base_name = "Enthusiastic Budget Spenders"
        else:
            base_name = "Budget-Conscious"

        used_names[base_name] = used_names.get(base_name, 0) + 1
        suffix = f" #{used_names[base_name]}" if used_names[base_name] > 1 else ""
        names[cluster] = base_name + suffix

    return names


def make_cluster_chart(data, segment_names):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for cluster in sorted(data["Cluster"].unique()):
        group = data[data["Cluster"] == cluster]
        ax.scatter(
            group[INCOME],
            group[SPENDING],
            s=55,
            alpha=0.82,
            label=f"Cluster {cluster}: {segment_names[cluster]}",
        )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1–100)")
    ax.set_title("Customer Groups by Income and Spending")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


try:
    original_data = load_customer_data()
except Exception as error:
    st.error(f"Unable to load the customer dataset: {error}")
    st.stop()


st.sidebar.markdown("# Customer Insights")
st.sidebar.caption("Mall Customer Segmentation")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Segment Profiles", "Data Explorer", "Model Evaluation"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
cluster_count = st.sidebar.slider(
    "Number of clusters (k)",
    min_value=2,
    max_value=min(10, len(original_data) - 1),
    value=5,
    help="Choose how many customer groups K-Means should create.",
)
st.sidebar.info("Clustering features: annual income and spending score.")


# Scale the selected features and fit K-Means.
data = original_data.copy()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[FEATURES])

model = KMeans(
    n_clusters=cluster_count,
    n_init=10,
    random_state=42,
)
data["Cluster"] = model.fit_predict(scaled_features)

cluster_profile = data.groupby("Cluster").agg(
    Customers=("Cluster", "size"),
    **{
        "Average income (k$)": (INCOME, "mean"),
        "Average spending score": (SPENDING, "mean"),
    },
).round(1)

median_income = data[INCOME].median()
median_spending = data[SPENDING].median()
segment_names = assign_segment_names(
    cluster_profile,
    median_income,
    median_spending,
)
data["Segment"] = data["Cluster"].map(segment_names)

current_silhouette = silhouette_score(scaled_features, data["Cluster"])


st.title("Customer Segmentation Analytics")
st.caption(
    "Analyze customer groups by annual income and spending behavior "
    "using K-Means clustering."
)


if page == "Overview":
    st.subheader("Business Overview")

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Customer records", f"{len(data):,}")
    metric_2.metric("Customer segments", cluster_count)
    metric_3.metric("Silhouette score", f"{current_silhouette:.3f}")

    st.markdown("### Customer segment map")
    chart = make_cluster_chart(data, segment_names)
    st.pyplot(chart, use_container_width=True)
    plt.close(chart)

    st.markdown("### Segment snapshot")
    overview_table = cluster_profile.copy()
    overview_table.insert(
        0,
        "Segment",
        [segment_names[cluster] for cluster in overview_table.index],
    )
    st.dataframe(overview_table, use_container_width=True)

    st.caption(
        "Segment names are rule-based summaries of cluster averages. "
        "Review the profiles before using them for business decisions."
    )


elif page == "Segment Profiles":
    st.subheader("Customer Segment Profiles")
    st.write(
        "Each segment summarizes customers grouped by similar income and "
        "spending patterns. Recommendations are example business actions."
    )

    for cluster, row in cluster_profile.iterrows():
        segment = segment_names[cluster]
        action_key = segment.split(" #")[0]
        action = SEGMENT_ACTIONS.get(
            action_key,
            "Review this group for suitable offers.",
        )

        with st.container(border=True):
            st.markdown(f"### {segment}")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Customers", int(row["Customers"]))
            col_b.metric("Average income", f"${row['Average income (k$)']:.1f}k")
            col_c.metric(
                "Average spending score",
                f"{row['Average spending score']:.1f}",
            )
            st.write(f"**Suggested business action:** {action}")

    st.markdown("### Profile summary")
    profile_table = cluster_profile.copy()
    profile_table.insert(
        0,
        "Segment",
        [segment_names[cluster] for cluster in profile_table.index],
    )
    st.dataframe(profile_table, use_container_width=True)


elif page == "Data Explorer":
    st.subheader("Explore Customer Data")
    st.write(
        "Review the source data, inspect feature distributions, and download "
        "customer records with their assigned clusters."
    )

    chart_columns = st.columns(3)
    distribution_features = [
        ("Age", "Age distribution"),
        (INCOME, "Annual income distribution"),
        (SPENDING, "Spending score distribution"),
    ]

    for column, (feature, title) in zip(chart_columns, distribution_features):
        if feature in data.columns:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            values = pd.to_numeric(data[feature], errors="coerce").dropna()
            ax.hist(
                values,
                bins=15,
                color="#2563EB",
                edgecolor="white",
            )
            ax.set_title(title)
            ax.set_ylabel("Customer count")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            column.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("### Customer records and assigned segments")
    st.dataframe(data, use_container_width=True, hide_index=True)

    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download segmented customer data",
        data=csv_data,
        file_name="customers_with_segments.csv",
        mime="text/csv",
    )


elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    st.write(
        "K-Means is fitted separately for each value of k. A higher Silhouette "
        "score and a lower Davies–Bouldin score generally indicate better-separated "
        "clusters. Interpret these metrics alongside business context."
    )

    max_test_k = min(10, len(data) - 1)
    evaluation_rows = []

    for test_k in range(2, max_test_k + 1):
        test_model = KMeans(
            n_clusters=test_k,
            n_init=10,
            random_state=42,
        )
        test_labels = test_model.fit_predict(scaled_features)

        evaluation_rows.append(
            {
                "k": test_k,
                "Inertia": test_model.inertia_,
                "Silhouette score": silhouette_score(
                    scaled_features,
                    test_labels,
                ),
                "Davies-Bouldin score": davies_bouldin_score(
                    scaled_features,
                    test_labels,
                ),
            }
        )

    evaluation = pd.DataFrame(evaluation_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(evaluation["k"], evaluation["Inertia"], marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_ylabel("Inertia (lower is better)")

    axes[1].plot(
        evaluation["k"],
        evaluation["Silhouette score"],
        marker="o",
        color="green",
    )
    axes[1].set_title("Silhouette Score")
    axes[1].set_ylabel("Higher is generally better")

    axes[2].plot(
        evaluation["k"],
        evaluation["Davies-Bouldin score"],
        marker="o",
        color="crimson",
    )
    axes[2].set_title("Davies–Bouldin Score")
    axes[2].set_ylabel("Lower is generally better")

    for ax in axes:
        ax.set_xlabel("Number of clusters (k)")
        ax.grid(alpha=0.22)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### PCA cluster projection")
    pca_projection = PCA(n_components=2).fit_transform(scaled_features)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        pca_projection[:, 0],
        pca_projection[:, 1],
        c=data["Cluster"],
        cmap="tab10",
        s=48,
        alpha=0.82,
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Customer Groups in Two Dimensions")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Evaluation results")
    st.dataframe(evaluation.round(3), use_container_width=True, hide_index=True)

    with st.expander("Methodology"):
        st.write("1. Load the Mall Customers CSV and check required fields.")
        st.write("2. Remove duplicate rows and records missing clustering features.")
        st.write("3. Scale annual income and spending score with StandardScaler.")
        st.write("4. Fit K-Means using the selected number of clusters.")
        st.write("5. Compare candidate cluster counts using evaluation metrics.")
        st.write(
            "6. Summarize clusters using their average income and spending patterns."
        )
