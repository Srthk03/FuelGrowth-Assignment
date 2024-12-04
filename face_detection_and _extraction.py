# This module performs the face detection, extraction,creates face embeddings and clustering on the whole unique fingerprints with average data we previously saved in a csv file.


import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import mediapipe as mp
import os
from collections import defaultdict

# Step 1: Preprocess and Average Performance Scores
def preprocess_and_average(input_filename="video_fingerprints_avg_performance.csv"):
    df_avg = pd.read_csv(input_filename)
    return df_avg

# Step 2: FaceMesh for Face Landmark Detection and Embedding Extraction
def extract_face_embeddings_and_save_images(video_url, face_mesh_model, output_folder, frame_skip=20):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_url}")
        return []

    embeddings = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames that are not multiples of `frame_skip`
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Convert to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with FaceMesh
        results = face_mesh_model.process(frame_rgb)

        if results.multi_face_landmarks:
            for i, landmarks in enumerate(results.multi_face_landmarks):
                # Flatten the landmarks into a list (each face's landmarks)
                face_embedding = []
                for landmark in landmarks.landmark:
                    face_embedding.append(landmark.x)
                    face_embedding.append(landmark.y)
                    face_embedding.append(landmark.z)
                embeddings.append(face_embedding)

                # Save the face image to the output folder
                face_folder = os.path.join(output_folder, "temp_faces")
                os.makedirs(face_folder, exist_ok=True)
                face_img_path = os.path.join(face_folder, f"{video_url.split('/')[-1]}_frame{frame_count}_face{i}.jpg")
                cv2.imwrite(face_img_path, frame)

        frame_count += 1

    cap.release()
    return embeddings

# Step 3: Perform K-Means Clustering for Face Matching Across All Videos
def cluster_faces_kmeans(face_embeddings, n_clusters=25):
    if len(face_embeddings) < 2:
        # If there are fewer than 2 face embeddings, we can't perform clustering
        return [0] * len(face_embeddings)  # Assign all faces to the same cluster (cluster 0)

    # Dynamically adjust the number of clusters based on the number of face embeddings
    n_clusters = min(n_clusters, len(face_embeddings))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(face_embeddings)

    return kmeans.labels_

# Step 4: Process Videos and Cluster Faces
def process_videos_and_cluster(input_csv, output_csv, face_mesh_model=None, n_clusters=15):
    df = preprocess_and_average(input_csv)
    results = []
    video_urls = df['video_url'].tolist()  # Get the video URLs from the DataFrame
    performance_scores = df['performance'].tolist()  # Get performance scores from the DataFrame

    # Output folder for storing images
    output_folder = "cluster_images"
    os.makedirs(output_folder, exist_ok=True)

    # Collect all face embeddings for each video
    all_face_embeddings = []  # To hold all face embeddings
    video_indices = []  # To keep track of the video index for each embedding
    no_face_videos = []  # To hold video URLs with no detected faces

    # Extract embeddings and save images for all videos
    for idx, video_url in enumerate(video_urls):
        print(f"Processing video: {video_url}")

        # Extract face embeddings and save images for the video
        embeddings = extract_face_embeddings_and_save_images(video_url, face_mesh_model, output_folder)

        if embeddings:
            for embedding in embeddings:
                all_face_embeddings.append(embedding)
                video_indices.append(idx)
        else:
            # Add videos with no faces to the "no face" list
            no_face_videos.append((video_url, performance_scores[idx]))

    # Cluster based on face embeddings across all videos
    labels = cluster_faces_kmeans(all_face_embeddings, n_clusters)

    # Map the assigned cluster labels back to the corresponding video URLs
    video_cluster_mapping = defaultdict(list)
    for idx, label in zip(video_indices, labels):
        video_cluster_mapping[label].append(video_urls[idx])

    # Assign a unique cluster number to each video based on face matching
    cluster_folders = {}
    for cluster_id, video_urls_in_cluster in video_cluster_mapping.items():
        # Create a folder for each cluster
        cluster_folder = os.path.join(output_folder, f"cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)
        cluster_folders[cluster_id] = cluster_folder

        for video_url in video_urls_in_cluster:
            performance = performance_scores[video_urls.index(video_url)]  # Get the performance score for the video
            results.append({
                'cluster_id': cluster_id,
                'video_urls': list(set(video_urls_in_cluster)),
                'average_performance': np.mean([performance_scores[video_urls.index(url)] for url in video_urls_in_cluster])
            })

    # Handle "no face" videos as Cluster 0
    no_face_cluster_folder = os.path.join(output_folder, "cluster_0")
    os.makedirs(no_face_cluster_folder, exist_ok=True)
    for video_url, performance in no_face_videos:
        results.append({
            'cluster_id': 0,
            'video_urls': [video_url],
            'average_performance': performance
        })

    # Sort results by cluster_id
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='cluster_id', ascending=True)

    # Save results to a CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Main Function
def main():
    # Initialize MediaPipe FaceMesh model
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # Call the function without passing 'n_clusters' argument
    process_videos_and_cluster(
        input_csv="video_fingerprints_avg_performance.csv",
        output_csv="final_video_clusters_v3.csv",
        face_mesh_model=mp_face_mesh
    )
 #The above program will generate a csv with cluster values assigned to all the videos, according to the face embedding similarities.





def merge_clusters(input_csv="final_video_clusters_v3.csv", output_csv="merged_clusters.csv"):
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Group by cluster_id and aggregate
    merged_df = df.groupby('cluster_id').agg({
        'video_urls': lambda x: list(set([item for sublist in x.apply(eval) for item in sublist])),
        'average_performance': 'mean'
    }).reset_index()

    # Save the merged DataFrame to a new CSV
    merged_df.to_csv(output_csv, index=False)

    print(f"Merged clusters saved to {output_csv}")
    print(merged_df.head())
          
#The above program will combine the videos based on their clusters, and generate a new csv with cluster_id as the primary key.


if __name__ == "__main__":
    main()
    merge_clusters()
 
