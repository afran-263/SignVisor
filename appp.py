from flask import Flask, request, jsonify, render_template
import subprocess
import threading
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS
import subprocess

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable CORS for frontend access

# Load trained model and labels
try:
    model = load_model("asl_cnn_model.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    labels = np.load("label_map.npy")
    print("Model and labels loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # Initialize with placeholder model and labels in case they don't exist yet
    model = None
    labels = np.array([])

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the integrated page

# Keep these routes for backward compatibility
@app.route('/data')
def data_page():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

@app.route('/text')
def text_page():
    return render_template('index.html')

@app.route('/start_data_collection', methods=['POST'])
def start_data_collection():
    data = request.json
    label = data.get("label")
    if not label:
        return jsonify({"error": "Label is required"}), 400
    
    # Start data.py and pass the label as input
    subprocess.Popen(["python", "data.py", label])
    return jsonify({"message": f"Data collection started for label: {label}"})

@app.route('/save_frame', methods=['POST'])
def save_frame():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400  # Handle missing image
    
    file = request.files['image']
    label = request.args.get("label", "unknown")
    label_path = os.path.join("dataset", label)
    os.makedirs(label_path, exist_ok=True)
    file_path = os.path.join(label_path, f"{label}_{len(os.listdir(label_path))}.jpg")
    file.save(file_path)
    return jsonify({"message": "Frame saved successfully!"})

@app.route('/train', methods=['POST'])
def train_model():
    def train():
        global model, labels
        try:
            subprocess.run(["python", "cnn.py"])  # Run training script
            
            # Reload the trained model
            model = load_model("asl_cnn_model.h5")
            labels = np.load("label_map.npy")
            print("Model training completed and reloaded.")
            
            # Notify frontend about completion
            with open("static/train_status.txt", "w") as f:
                f.write("Training completed!")
        except Exception as e:
            print(f"Error during training: {e}")
            with open("static/train_status.txt", "w") as f:
                f.write("Training failed!")

    # Write initial message to the status file
    with open("static/train_status.txt", "w") as f:
        f.write("Model training started! This may take several minutes.")

    # Start training in a separate thread
    thread = threading.Thread(target=train)
    thread.daemon = True
    thread.start()

    return jsonify({"message": "Model training started! This may take several minutes."})

@app.route('/train_status', methods=['GET'])
def get_train_status():
    """Fetches the latest training status"""
    try:
        with open("static/train_status.txt", "r") as f:
            status = f.read()
        return jsonify({"status": status})
    except FileNotFoundError:
        return jsonify({"status": "No training process found"})


@app.route('/collect', methods=['POST'])
def collect_data():
    data = request.get_json()
    label = data.get("label")

    if not label:
        return jsonify({"message": "Label is required"}), 400

    try:
        # Create dataset directory if it doesn't exist
        os.makedirs("dataset", exist_ok=True)
        os.makedirs(f"dataset/{label}", exist_ok=True)
        
        # Run data.py with the label as an argument
        subprocess.Popen(["python", "data.py", label])
        return jsonify({"message": f"Data collection started for label: {label}"})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/predict_sign', methods=['POST'])
def predict_sign():
    global model, labels
    
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 400
        
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    try:
        # Read and process image
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, 21, 3)
                landmarks = landmarks / np.max(landmarks)
                
                prediction = model.predict(landmarks)
                if len(labels) > 0:
                    predicted_label = labels[np.argmax(prediction)]
                    confidence = float(np.max(prediction))
                    return jsonify({
                        "prediction": predicted_label,
                        "confidence": f"{confidence:.2f}"
                    })
                else:
                    return jsonify({"error": "No labels found. Please train the model first."})
        
        return jsonify({"prediction": "No hand detected"})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)