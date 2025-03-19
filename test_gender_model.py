import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def load_data(data_folder, label_file):
    # Load the CSV file containing filenames and genders
    df = pd.read_csv(label_file)

    X, y = [], []

    for _, row in df.iterrows():
        audio_file = row["filename"]
        gender = row["gender"]

        # Adjust the audio_file path to handle the folder structure
        audio_file_path = os.path.join(data_folder, audio_file)
        audio_file_path = os.path.normpath(audio_file_path)  # Normalize the path for the current platform

        if not os.path.exists(audio_file_path):
            print(f"File not found: {audio_file_path}")
            continue

        # Load the audio data from the '.npy' file
        audio_data = np.load(audio_file_path)

        # Use the mean of the audio features as a fixed-length representation
        audio_data = np.mean(audio_data, axis=0)

        X.append(audio_data)
        y.append(gender)

    X = np.array(X)
    y = np.array(y)

    # Convert gender labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y

# Load the saved model
saved_model_path = "results"  # Replace with the path to the saved model
model = tf.keras.models.load_model(saved_model_path)

# Example usage:
if __name__ == "__main__":
    data_folder = "reshaped_data_i"  # Assuming the '.npy' audio files are in the 'data_i' folder in the same directory as this script
    label_file = "balanced-all.csv"  # Replace with the path to the CSV file with filenames and genders

    # Load test data
    X_test, y_test = load_data(data_folder, label_file)
    df = pd.read_csv(label_file)  # Define the DataFrame after loading the data

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Make predictions on new audio samples
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # Convert integer labels back to gender names
    label_encoder = LabelEncoder()
    label_encoder.fit(["male", "female"])
    predicted_genders = label_encoder.inverse_transform(predicted_labels)

    # Print the predictions for each sample
    print("\nPredictions:")
    for i, filename in enumerate(df["filename"]):
        print(f"{filename}: {predicted_genders[i]}")
