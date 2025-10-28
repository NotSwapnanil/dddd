from flask import Flask, request, jsonify, Response, render_template_string
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64

app = Flask(__name__)

# =========================
# Load Trained Model
# =========================
model = joblib.load("asl_model.pkl")

# =========================
# Initialize MediaPipe
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


# =========================
# ESP32-CAM API Endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        img = None

        # --- CASE 1: ESP32 sends raw JPEG bytes ---
        if request.data and len(request.data) > 0:
            npimg = np.frombuffer(request.data, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # --- CASE 2: Form file upload (multipart/form-data) ---
        elif 'image' in request.files:
            file = request.files['image']
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # --- CASE 3: Base64 image string ---
        elif 'image' in request.form:
            img_data = base64.b64decode(request.form['image'])
            npimg = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # --- If no image found ---
        else:
            return jsonify({"error": "No image found"}), 400

        # --- Run MediaPipe + Model Prediction ---
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({"gesture": "No hand detected"})

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]

        prediction = model.predict([landmarks])[0]
        return jsonify({"gesture": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Web Interface (Laptop Webcam)
# =========================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>ASL Gesture Recognition</title>
  <style>
    body { text-align: center; background-color: #111; color: white; font-family: sans-serif; }
    img { width: 80%; border: 4px solid #00FF88; border-radius: 20px; margin-top: 20px; }
    h1 { color: #00FF88; }
  </style>
</head>
<body>
  <h1>🖐 ASL Gesture Recognition Webcam</h1>
  <img src="{{ url_for('webcam_feed') }}" />
  <p>Flask Server Running — ESP32 Endpoint at <b>/predict</b></p>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/webcam_feed")
def webcam_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
                prediction = model.predict([landmarks])[0]
                cv2.putText(frame, f'{prediction}', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# =========================
# Run Flask Server
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
