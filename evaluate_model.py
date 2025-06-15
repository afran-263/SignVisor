import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

# ----------------------------
# CONFIGURATION
# ----------------------------
DATASET_PATH = "dataset"
MODEL_PATH = "asl_cnn_model.h5"
LABEL_MAP_PATH = "label_map.npy"
TEST_SIZE = 0.2  # 80/20 split

# ----------------------------
# LOAD DATASET
# ----------------------------
labels = sorted(os.listdir(DATASET_PATH))
label_map = {label: idx for idx, label in enumerate(labels)}
reverse_label_map = {idx: label for label, idx in label_map.items()}

X, y = [], []

for label in labels:
    label_folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(label_folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(label_folder, file))
            X.append(data)
            y.append(label_map[label])

X = np.array(X)
y = np.array(y)

# Normalize
X = X / np.max(X)

# Reshape for CNN input
X = X.reshape(-1, 21, 3)

# ----------------------------
# SPLIT DATA
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = load_model(MODEL_PATH)

# ----------------------------
# PREDICT ON VALIDATION SET
# ----------------------------
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# ----------------------------
# ACCURACY & REPORT
# ----------------------------
accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

report = classification_report(y_val, y_pred, target_names=[reverse_label_map[i] for i in range(len(labels))])
print("\nClassification Report:\n")
print(report)

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[reverse_label_map[i] for i in range(len(labels))],
            yticklabels=[reverse_label_map[i] for i in range(len(labels))])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ----------------------------
# DISPLAY 2 CORRECTLY CLASSIFIED SAMPLES (Improved Scatter with Axes)
# ----------------------------
correct_indices = np.where(y_val == y_pred)[0]

if len(correct_indices) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, ax in zip(correct_indices[:2], axes):
        landmark = X_val[i].reshape(21, 3)
        
        # X: Horizontal position (0 = left, 1 = right)
        # Y: Vertical position (0 = top, 1 = bottom)
        # Note: Y-axis is flipped below to match image coordinate space
        
        ax.scatter(landmark[:, 0], landmark[:, 1], c='royalblue', edgecolors='black', s=60)
        ax.set_title(f"True: {reverse_label_map[y_val[i]]}\nPredicted: {reverse_label_map[y_pred[i]]}", fontsize=12)
        ax.set_xlabel("X Position (left → right)", fontsize=10)
        ax.set_ylabel("Y Position (top → bottom)", fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_facecolor('#f9f9f9')
        ax.set_aspect('equal')  # Preserve proportions to reflect real hand shape

        # Set fixed axis ranges for consistent comparison
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Flip Y-axis for intuitive hand orientation (top of screen = top of plot)

        for spine in ax.spines.values():
            spine.set_edgecolor('#bbbbbb')
            spine.set_linewidth(1.2)

    plt.suptitle("Correctly Classified Hand Landmark Scatter Plots", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough correctly classified samples to display.")
