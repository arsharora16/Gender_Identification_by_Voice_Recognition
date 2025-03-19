import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
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

def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_gender_model(data_folder, label_file, save_model_path):
    X, y = load_data(data_folder, label_file)

    # Check if we have enough data for training
    if len(X) == 0:
        print("No data found. Please check if the dataset is correctly loaded.")
        return

    input_shape = (X.shape[1],)  # Input shape is (num_features,)
    num_classes = len(np.unique(y))

    model = build_model(input_shape, num_classes)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save(save_model_path)

# Example usage:
if __name__ == "__main__":
    data_folder = "reshaped_data_i"  # Assuming the '.npy' audio files are in the 'data_i' folder in the same directory as this script
    label_file = "balanced-all.csv"  # Replace with the path to the CSV file with filenames and genders
    save_model_path = "results"  # Replace with the desired save path for the trained model
    train_gender_model(data_folder, label_file, save_model_path)
