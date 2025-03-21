import cv2
import os
import numpy as np
import mediapipe as mp
import sys

# Check if a label argument is passed
if len(sys.argv) > 1:
    label = sys.argv[1]  # Get label from command-line argument
else:
    print("Error: No label provided.")
    exit()

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Define dataset path
data_path = "dataset"
os.makedirs(data_path, exist_ok=True)

# Create directory for the given label
label_path = os.path.join(data_path, label)
os.makedirs(label_path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Collecting data for '{label}'... Press 's' to save, 'q' to quit.")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Display label and count
            cv2.putText(frame, f"Sign: {label} ({count})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 's' to save hand landmarks
    if key == ord('s') and results.multi_hand_landmarks:
        np.save(os.path.join(label_path, f"{label}_{count}.npy"), landmarks)
        print(f"Saved: {label}_{count}.npy")
        count += 1

    # Press 'q' to quit
    elif key == ord('q'):
        print("Exiting data collection.")
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
