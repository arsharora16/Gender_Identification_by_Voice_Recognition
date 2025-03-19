import os
import numpy as np

# Assuming each audio sample contains 128 audio features
num_features = 128

# Create a new folder to store the reshaped data
output_folder = "reshaped_data_i"
os.makedirs(output_folder, exist_ok=True)

# List all .npy files in the 'data_i' folder
data_i_folder = "data_i"
npy_files = [file for file in os.listdir(data_i_folder) if file.endswith(".npy")]

for npy_file in npy_files:
    # Load data from the .npy file
    data = np.load(os.path.join(data_i_folder, npy_file))

    # Reshape the data into a 2D array (treat each sample as a single frame)
    reshaped_data = data.reshape(-1, num_features)

    # Save the reshaped data to a new .npy file in 'reshaped_data_i' folder
    output_file = os.path.join(output_folder, npy_file)
    np.save(output_file, reshaped_data)

print("Reshaping and saving complete!")
