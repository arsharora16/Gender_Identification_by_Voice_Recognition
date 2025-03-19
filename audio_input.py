import librosa
import numpy as np
import tensorflow as tf
import pyaudio
from sklearn.preprocessing import LabelEncoder

def load_model(saved_model_path):
    return tf.keras.models.load_model(saved_model_path)

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32  # Change to floating-point format
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording audio...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
    return audio_data, RATE

def extract_audio_features(audio_data, sample_rate):
    # Normalize the audio data to the range [-1, 1]
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Print the normalized audio data
    print("[==========Normalized audio data:==========]")
    print(audio_data)
    print("[========================================]")

    # Extract Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

    # Transpose the MFCCs to have time along the first axis
    mfccs = mfccs.T

    # Print the MFCCs
    #print("[==========MFCCs:==========]")
    #print(mfccs)
    #print("[========================================]")

    # Take the mean of MFCCs across time to get a fixed-size representation
    mfccs_mean = np.mean(mfccs, axis=0)

    # Pad or truncate the MFCCs to have a fixed size (e.g., 128)
    fixed_size = 128
    if len(mfccs_mean) < fixed_size:
        pad_size = fixed_size - len(mfccs_mean)
        mfccs_mean = np.pad(mfccs_mean, (0, pad_size))
    else:
        mfccs_mean = mfccs_mean[:fixed_size]

    # Reshape to have a batch dimension of 1
    audio_features = np.expand_dims(mfccs_mean, axis=0)

    return audio_features

if __name__ == "__main__":
    # Path to the trained model
    saved_model_path = "results"  # Replace with the path to the saved model

    # Load the trained model
    model = load_model(saved_model_path)

    # Record audio from the microphone
    audio_data, sample_rate = record_audio()

    # Ensure audio_data is not empty
    if len(audio_data) == 0:
        raise ValueError("No audio data recorded.")

    # Extract audio features (MFCCs)
    audio_features = extract_audio_features(audio_data, sample_rate)

    # Predict the gender
    predicted_gender_probs = model.predict(audio_features)
    predicted_label = np.argmax(predicted_gender_probs, axis=1)[0]

    label_encoder = LabelEncoder()
    label_encoder.fit(["male", "female"])
    predicted_gender = label_encoder.inverse_transform([predicted_label])[0]

    prob_male_percentage = predicted_gender_probs[0][1] * 100
    prob_female_percentage = predicted_gender_probs[0][0] * 100

    # Print the predicted gender and corresponding probabilities
    print(f"Predicted gender: {predicted_gender}")
    print(f"Probability of male: {prob_male_percentage:.2f}%", end='')
    print(f"Probability of female: {prob_female_percentage:.2f}%")
