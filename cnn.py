import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split

# Define dataset path
data_path = "dataset"
labels = sorted(os.listdir(data_path))  # Dynamically fetch all labels

# Load dataset
X, y = [], []
label_map = {label: idx for idx, label in enumerate(labels)}  # Label encoding

for label in labels:
    label_path = os.path.join(data_path, label)
    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(label_path, file))
            X.append(data)
            y.append(label_map[label])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize data
X = X / np.max(X)  # Normalize between 0 and 1

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN input (samples, timesteps, features)
X_train = X_train.reshape(-1, 21, 3)  # 21 hand landmarks with (x, y, z) coordinates
X_test = X_test.reshape(-1, 21, 3)

# Define CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(21, 3)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
epochs = 50
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

# Save model and label map
model.save("asl_cnn_model.h5")
np.save("label_map.npy", labels)

print("Model training complete and saved as 'asl_cnn_model.h5'")