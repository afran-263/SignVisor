from flask import Flask, Response, render_template
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import time
import socket
import struct

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
UDP_IP = "0.0.0.0"       # Listen on all interfaces
UDP_PORT = 5001          # UDP port to receive data
MODEL_PATH = "asl_cnn_model.h5"
FRAME_TIMEOUT = 2.0      # Seconds to wait before discarding an incomplete frame
CONFIDENCE_THRESHOLD = 0.80  # Only update text buffer if confidence >= this threshold

# Desired view window size (in pixels)
VIEW_WIDTH = 820
VIEW_HEIGHT = 640

# ------------------------------------------------------------
# Load ASL Model and Class Definitions
# ------------------------------------------------------------
# Ensure your model includes a "next" symbol to denote word breaks.
asl_classes = ["hello", "peace", "how are you", "give me some water"]
model = load_model(MODEL_PATH)

# ------------------------------------------------------------
# Initialize Mediapipe Hands
# ------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5)

# ------------------------------------------------------------
# Global variables for latest processed frame, text buffer, and locks
# ------------------------------------------------------------
latest_frame = None
frame_lock = threading.Lock()
text_buffer = ""
text_lock = threading.Lock()
last_prediction = None  # To avoid repeating the same word consecutively

# Dictionary to hold fragments for each frame ID.
# Format: { frame_id: { "fragments": [frag0, frag1, ...], "total": total_fragments, "timestamp": time_received } }
frames_buffer = {}

# Header format: 2 bytes frameId, 1 byte total_fragments, 1 byte fragment_index
HEADER_FORMAT = "!HBB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

app = Flask(__name__)

def udp_receiver():
    global latest_frame, frames_buffer, text_buffer, last_prediction
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"UDP server listening on port {UDP_PORT}")
    
    while True:
        try:
            data, addr = sock.recvfrom(65535)
            if len(data) < HEADER_SIZE:
                continue
            # Parse header
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
            
            # Reassemble if complete
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
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    landmarks = landmarks.flatten().reshape(1, 21, 3)
                    max_val = np.max(landmarks)
                    if max_val > 0:
                        landmarks = landmarks / max_val
                    prediction = model.predict(landmarks)
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
                    # Only update text buffer if confidence is above threshold
                    if prediction_label is not None and confidence >= CONFIDENCE_THRESHOLD:
                        with text_lock:
                            if prediction_label.lower() == "next":
                                if len(text_buffer) == 0 or text_buffer[-1] != " ":
                                    text_buffer += " "
                                    last_prediction = None  # Reset after word break
                            else:
                                if prediction_label != last_prediction:
                                    text_buffer += prediction_label
                                    last_prediction = prediction_label
                            if len(text_buffer) > 100:
                                text_buffer = text_buffer[-100:]
                            print("Text Buffer:", repr(text_buffer))
            
            # Cleanup incomplete frames
            current_time = time.time()
            to_delete = []
            for fid, info in frames_buffer.items():
                if current_time - info["timestamp"] > FRAME_TIMEOUT:
                    to_delete.append(fid)
            for fid in to_delete:
                print(f"Discarding incomplete frame ID {fid} due to timeout")
                del frames_buffer[fid]
                
        except Exception as e:
            print("UDP receiver error:", e)
            time.sleep(0.1)

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                placeholder = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for video...", (50, VIEW_HEIGHT//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', placeholder)
                frame_bytes = jpeg.tobytes() if ret else None
            else:
                frame = cv2.imdecode(np.frombuffer(latest_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    frame_bytes = latest_frame
                else:
                    resized = cv2.resize(frame, (VIEW_WIDTH, VIEW_HEIGHT))
                    ret, jpeg = cv2.imencode('.jpg', resized)
                    frame_bytes = jpeg.tobytes() if ret else None
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

@app.route("/view")
def view():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/text")
def get_text():
    with text_lock:
        return f"<pre>{text_buffer}</pre>"

@app.route("/")
def index():
    return render_template("index.html", width=VIEW_WIDTH, height=VIEW_HEIGHT)

if __name__ == "__main__":
    t = threading.Thread(target=udp_receiver, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000)
