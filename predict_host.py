from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import time
import socket
import struct
import os
import subprocess
from flask_cors import CORS

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
UDP_IP = "0.0.0.0"       # Listen on all interfaces (for UDP reception)
UDP_PORT = 5001          # UDP port for incoming data
MODEL_PATH = "asl_cnn_model.h5"
FRAME_TIMEOUT = 2.0      # Seconds to wait before discarding an incomplete frame
CONFIDENCE_THRESHOLD = 0.80  # Only update text if confidence >= this value

# Desired view window size (in pixels)
VIEW_WIDTH = 820
VIEW_HEIGHT = 640

# ------------------------------------------------------------
# Load ASL Model and Class Definitions
# ------------------------------------------------------------
# Ensure your model includes a "next" symbol for word breaks.
asl_classes = ["hello", "peace", "how are you", "give me some water"]
try:
    model = load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    labels = np.load("label_map.npy")
    print("Model and labels loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    labels = np.array([])

# ------------------------------------------------------------
# Initialize Mediapipe Hands
# ------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5)

# ------------------------------------------------------------
# Global Variables and Locks
# ------------------------------------------------------------
latest_frame = None
frame_lock = threading.Lock()

text_buffer = ""
text_lock = threading.Lock()
last_prediction = None  # To avoid repeating the same word consecutively
current_prediction = ""  # Holds the current active predicted sign

# Buffer for reassembling UDP fragments:
frames_buffer = {}
HEADER_FORMAT = "!HBB"  # 2 bytes frameId, 1 byte total_fragments, 1 byte fragment_index
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# ------------------------------------------------------------
# Flask App Initialization
# ------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ------------------------------------------------------------
# UDP Receiver Function
# ------------------------------------------------------------
def udp_receiver():
    global latest_frame, frames_buffer, text_buffer, last_prediction, current_prediction
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"UDP server listening on port {UDP_PORT}")
    
    while True:
        try:
            data, addr = sock.recvfrom(65535)
            if len(data) < HEADER_SIZE:
                continue
            header = data[:HEADER_SIZE]
            frame_id, total_fragments, fragment_index = struct.unpack(HEADER_FORMAT, header)
            payload = data[HEADER_SIZE:]
            
            if frame_id not in frames_buffer:
                frames_buffer[frame_id] = {
                    "fragments": [None] * total_fragments,
                    "total": total_fragments,
                    "timestamp": time.time()
                }
            frames_buffer[frame_id]["fragments"][fragment_index] = payload
            
            if all(frag is not None for frag in frames_buffer[frame_id]["fragments"]):
                full_frame = b"".join(frames_buffer[frame_id]["fragments"])
                del frames_buffer[frame_id]
                frame = cv2.imdecode(np.frombuffer(full_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print("Error converting frame:", e)
                    continue
                results = hands.process(rgb_frame)
                prediction_label = None
                confidence = 0.0
                if results.multi_hand_landmarks and model is not None:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    landmarks = landmarks.flatten().reshape(1, 21, 3)
                    max_val = np.max(landmarks)
                    if max_val > 0:
                        landmarks = landmarks / max_val
                    prediction = model.predict(landmarks)
                    if len(labels) > 0:
                        prediction_label = labels[np.argmax(prediction)]
                    else:
                        prediction_label = asl_classes[np.argmax(prediction)]
                    confidence = float(np.max(prediction))
                    cv2.putText(frame, f"Prediction: {prediction_label} ({confidence:.2f})",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    with frame_lock:
                        latest_frame = jpeg.tobytes()
                    # Update current active prediction and text buffer if confidence is sufficient
                    if prediction_label is not None and confidence >= CONFIDENCE_THRESHOLD:
                        current_prediction = prediction_label
                        with text_lock:
                            if prediction_label.lower() == "next":
                                if len(text_buffer) == 0 or text_buffer[-1] != " ":
                                    text_buffer += " "
                                    last_prediction = None
                            else:
                                if prediction_label != last_prediction:
                                    text_buffer += prediction_label
                                    last_prediction = prediction_label
                            if len(text_buffer) > 100:
                                text_buffer = text_buffer[-100:]
                            print("Text Buffer:", repr(text_buffer))
            
            current_time = time.time()
            to_delete = [fid for fid, info in frames_buffer.items() if current_time - info["timestamp"] > FRAME_TIMEOUT]
            for fid in to_delete:
                print(f"Discarding incomplete frame ID {fid} due to timeout")
                del frames_buffer[fid]
                
        except Exception as e:
            print("UDP receiver error:", e)
            time.sleep(0.1)

# ------------------------------------------------------------
# New Endpoint: Return the current active predicted sign
# ------------------------------------------------------------
@app.route("/current_prediction")
def get_current_prediction():
    global current_prediction
    return current_prediction

# ------------------------------------------------------------
# Frame Generator Function: Serve processed frames as MJPEG
# ------------------------------------------------------------
def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                placeholder = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for video...", (50, VIEW_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', placeholder)
                frame_bytes = jpeg.tobytes() if ret else None
            else:
                frame = cv2.imdecode(np.frombuffer(latest_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    frame_bytes = latest_frame
                else:
                    try:
                        resized = cv2.resize(frame, (VIEW_WIDTH, VIEW_HEIGHT))
                        ret, jpeg = cv2.imencode('.jpg', resized)
                        frame_bytes = jpeg.tobytes() if ret else latest_frame
                    except Exception as e:
                        print("Error in generate_frames during resize/encode:", e)
                        frame_bytes = latest_frame
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

# ------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", width=VIEW_WIDTH, height=VIEW_HEIGHT)

@app.route("/view")
def view():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/text")
def get_text():
    with text_lock:
        return f"<pre>{text_buffer}</pre>"

# ------------------------------------------------------------
# Data Collection Endpoint
# ------------------------------------------------------------
@app.route('/start_data_collection', methods=['POST'])
def start_data_collection():
    data = request.json
    label = data.get("label")
    if not label:
        return jsonify({"error": "Label is required"}), 400
    try:
        subprocess.Popen(["python", "data.py", label])
        return jsonify({"message": f"Data collection started for label: {label}"})
    except Exception as e:
        return jsonify({"error": f"Failed to start data collection: {str(e)}"}), 500

# ------------------------------------------------------------
# Training Endpoint
# ------------------------------------------------------------
@app.route('/train', methods=['POST'])
def train_model():
    def train():
        global model, labels
        try:
            subprocess.run(["python", "cnn.py"])
            model = load_model("asl_cnn_model.h5")
            labels = np.load("label_map.npy")
            print("Model training completed and reloaded.")
            with open("static/train_status.txt", "w") as f:
                f.write("Training completed!")
        except Exception as e:
            print(f"Error during training: {e}")
            with open("static/train_status.txt", "w") as f:
                f.write("Training failed!")
    with open("static/train_status.txt", "w") as f:
        f.write("Model training started! This may take several minutes.")
    thread = threading.Thread(target=train)
    thread.daemon = True
    thread.start()
    return jsonify({"message": "Model training started! This may take several minutes."})

@app.route('/train_status', methods=['GET'])
def get_train_status():
    try:
        with open("static/train_status.txt", "r") as f:
            status = f.read()
        return jsonify({"status": status})
    except FileNotFoundError:
        return jsonify({"status": "No training process found"})

# ------------------------------------------------------------
# Sign Prediction Endpoint (using file upload)
# ------------------------------------------------------------
@app.route('/predict_sign', methods=['POST'])
def predict_sign():
    global model, labels
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 400
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    try:
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
    
# ------------------------------------------------------------
# Main: Start UDP receiver thread and run the Flask app
# ------------------------------------------------------------
if __name__ == "__main__":
    udp_thread = threading.Thread(target=udp_receiver, daemon=True)
    udp_thread.start()
    app.run(host="0.0.0.0", port=5000)
