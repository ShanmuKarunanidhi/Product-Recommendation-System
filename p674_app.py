import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_files():

    cluster_model = pickle.load(
        open("cluster_model.pkl", "rb")
    )

    cluster_df = pickle.load(
        open("cluster_df.pkl", "rb")
    )

    item_similarity_df = pickle.load(
        open("item_similarity.pkl", "rb")
    )

    user_item = pickle.load(
        open("user_item.pkl", "rb")
    )

    return (
        cluster_model,
        cluster_df,
        item_similarity_df,
        user_item
    )

(
    cluster_model,
    cluster_df,
    item_similarity_df,
    user_item
) = load_files()

# ==========================================
# FUNCTIONS
# ==========================================

def recommend_products(product_id, top_n=5):

    recommendations = item_similarity_df[
        product_id
    ].sort_values(
        ascending=False
    )

    return recommendations.iloc[
        1:top_n+1
    ]


def get_cluster(user_id):

    cluster = cluster_df[
        cluster_df["userId"] == user_id
    ]["cluster"].values[0]

    return cluster


def get_similar_users(user_id, top_n=5):

    cluster_no = get_cluster(user_id)

    cluster_users = cluster_df[
        cluster_df["cluster"] == cluster_no
    ]["userId"].tolist()

    cluster_users.remove(user_id)

    return cluster_users[:top_n]


# ==========================================
# TITLE
# ==========================================

st.title("🛒 Product Recommendation System")

st.markdown("""
This recommendation engine uses:

- Item-Item Collaborative Filtering
- Product Recommendation to given user
- Similar User Identification
""")

st.divider()

# ==========================================
# USER SELECTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    user_id = st.selectbox(
        "Select User",
        cluster_df["userId"].unique()
    )

with col2:

    product_id = st.selectbox(
        "Select Product",
        item_similarity_df.index
    )

# ==========================================
# BUTTON
# ==========================================

if st.button("Generate Recommendations"):

    cluster_no = get_cluster(user_id)

    similar_users = get_similar_users(
        user_id,
        top_n=5
    )

    recommended_products = recommend_products(
        product_id,
        top_n=5
    )

    st.success("Recommendations Generated Successfully")

    st.divider()

    # ======================================
    # INFORMATION
    # ======================================

    colA, colB = st.columns(2)

    with colA:

        st.subheader("Recommendation Details")

        st.write(
            "**Recommendation Method:** Item-Item Collaborative Filtering"
        )

        st.write(
            "**Clustering Method Used:** MiniBatch KMeans"
        )

        st.write(
            f"**Assigned Cluster:** {cluster_no}"
        )

    with colB:

        st.subheader("Selected Inputs")

        st.write(
            f"**User ID:** {user_id}"
        )

        st.write(
            f"**Product ID:** {product_id}"
        )

    st.divider()

    # ======================================
    # SIMILAR USERS
    # ======================================

    st.subheader("👥 Similar Users")

    similar_user_df = pd.DataFrame({
        "User ID": similar_users
    })

    st.dataframe(
        similar_user_df,
        use_container_width=True
    )

    st.divider()

    # ======================================
    # RECOMMENDED PRODUCTS
    # ======================================

    st.subheader(" Recommended Products")

    rec_df = pd.DataFrame({

        "Product ID":
        recommended_products.index,

        "Similarity Score":
        recommended_products.values

    })

    st.dataframe(
        rec_df,
        use_container_width=True
    )

    st.divider()

    st.balloons()