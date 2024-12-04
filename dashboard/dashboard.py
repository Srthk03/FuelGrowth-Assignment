import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# Title and Introduction
st.title("Video Face Clustering Dashboard")
st.markdown("""
This dashboard allows you to process videos for unique face clustering, view cluster distributions, and download results.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file (e.g., video_fingerprints_avg_performance.csv)", type=["csv"])

if uploaded_file:
    # Load the uploaded CSV
    st.write("Preview of uploaded data:")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # Process Data Pipeline Button
    if st.button("Run Clustering Pipeline"):
        with st.spinner("Processing videos... this may take some time."):
            # Replace with your data pipeline function
            os.system("python clustering_pipeline.py")  # Adjust to your pipeline script

        st.success("Clustering completed!")
        st.write("Download the processed clusters below:")

        # Load and display results
        processed_file = "final_video_clusters_merged.csv"  # Replace with your output file
        if os.path.exists(processed_file):
            processed_data = pd.read_csv(processed_file)
            st.write("Clustered Data Preview:")
            st.dataframe(processed_data.head())

            # Download button
            st.download_button(
                label="Download Merged Clusters CSV",
                data=processed_data.to_csv(index=False).encode("utf-8"),
                file_name="final_video_clusters_merged.csv",
                mime="text/csv"
            )

            # Visualization
            st.subheader("Cluster Visualizations")
            
            # Cluster Distribution
            cluster_counts = processed_data['cluster_id'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
            ax.set_title("Cluster Distribution")
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("Number of Videos")
            st.pyplot(fig)

            # Performance Distribution
            st.subheader("Performance Distribution")
            fig, ax = plt.subplots()
            sns.histplot(processed_data['average_performance'], kde=True, ax=ax)
            ax.set_title("Performance Score Distribution")
            st.pyplot(fig)

            # PCA for Embeddings
            st.subheader("Cluster Embeddings (2D Projection)")
            embeddings = pd.DataFrame([eval(row) for row in processed_data['video_urls']])  # Replace with actual embeddings
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)

            fig, ax = plt.subplots()
            scatter = ax.scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=processed_data['cluster_id'], cmap='viridis'
            )
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            ax.set_title("PCA Projection of Clusters")
            st.pyplot(fig)
        else:
            st.error("Processed file not found. Ensure the pipeline script saves results.")

# Footer
st.markdown("Powered by Streamlit")
