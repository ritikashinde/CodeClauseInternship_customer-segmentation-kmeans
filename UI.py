import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("Customer Segmentation with K-Means")
st.markdown("This app uses the default Mall Customers dataset to segment customers using K-Means clustering.")

# Load default dataset
df = pd.read_csv("Mall_Customers.csv")
st.write("### Preview of Customer Data", df.head())

# Feature selection
features = ['Annual Income (k$)', 'Spending Score (1-100)']
if all(col in df.columns for col in features):
    X = df[features]

    # Select number of clusters
    k = st.slider("Select number of clusters (k)", 2, 10, 5)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Plotting
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=features[0], y=features[1], hue='Cluster', palette='Set1', ax=ax)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='yellow', s=200, marker='X', label='Centroids')
    ax.set_title("Customer Segments")
    ax.legend()
    st.pyplot(fig)

    # Download segmented data
    st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="segmented_customers.csv", mime="text/csv")

else:
    st.error("Dataset must contain: 'Annual Income (k$)', 'Spending Score (1-100)'")
