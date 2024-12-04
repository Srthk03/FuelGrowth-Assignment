import cv2
import pandas as pd
import os
import hashlib
import traceback

#  generate fingerprint for a video URL
def generate_fingerprint(video_url):
    try:
        # Open the video stream using OpenCV
        cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_url}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break  

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (64, 64))  # Reduce size for faster processing
            frames.append(resized_frame)

        cap.release()

        # Generate a "fingerprint" for the video (e.g., a hash of the frames)
        frame_hashes = [hashlib.md5(f.tobytes()).hexdigest() for f in frames]  # Hash each frame
        fingerprint = hashlib.md5("".join(frame_hashes).encode()).hexdigest()  # Combine and hash frame hashes

        return fingerprint

    except Exception as e:
        # Log the error and return None in case of failure
        print(f"Error processing {video_url}: {str(e)}")
        traceback.print_exc()
        return None

def process_videos(input_filename="social_data.csv"):
    df = pd.read_csv(input_filename)

    result_data = []

    for index, row in df.iterrows():
        video_url = row['Video URL']
        performance_score = row['Performance']

        try:
            print(f"Processing video: {video_url}")
            fingerprint = generate_fingerprint(video_url)

            if fingerprint:
                result_data.append({
                    "video_url": video_url,
                    "fingerprint": fingerprint,
                    "performance": performance_score,
                    "status": "Processed"
                })
            else:
                result_data.append({
                    "video_url": video_url,
                    "fingerprint": "N/A",
                    "performance": performance_score,
                    "status": "Skipped (Error)"
                })

        except Exception as e:
            print(f"Unexpected error with video {video_url}: {str(e)}")
            traceback.print_exc()
            result_data.append({
                "video_url": video_url,
                "fingerprint": "N/A",
                "performance": performance_score,
                "status": "Error"
            })

    return result_data

def save_to_csv(data, output_filename="video_fingerprints_with_performance.csv"):
    df = pd.DataFrame(data)

    df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


# Main execution: Process videos and save results to CSV
result_data = process_videos(input_filename="social_data.csv")
save_to_csv(result_data) #obtain csv for unique fingerprints for every video


# PROCESS TO FIND UNIQUE VIDEOS USING FINGERPRINT TO MANAGE DUPLICATE VIDEOS


# Step 1: Combine rows with the same fingerprint and average their performance scores
def preprocess_and_average(input_filename="video_fingerprints_with_performance.csv"):
    # Read the input CSV file
    df = pd.read_csv(input_filename)

    # Group by fingerprint and calculate the average of performance scores
    df_avg = df.groupby('fingerprint').agg({
        'video_url': 'first',  # Keep the first video URL (or you can use other strategies to choose one URL)
        'performance': 'mean'  # Average the performance score
    }).reset_index()

    return df_avg
df_avg = preprocess_and_average(input_filename="video_fingerprints_with_performance.csv")

df_avg.to_csv("video_fingerprints_avg_performance.csv", index=False)
print("Processed data saved to 'video_fingerprints_avg_performance.csv'")