from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import time
import threading

app = Flask(__name__)

# -----------------------------
# Configuration & Globals
# -----------------------------
MODEL_NAME = "best.pt" # change to "yolov8n.pt" for default
CAMERA_INDEX = 0       # 0 is usually the default webcam, you had 2

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open webcam at index {CAMERA_INDEX}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Shared state variables
lock = threading.Lock()
current_frame_bytes = None
latest_data = {
    "fps": 0.0,
    "objects": [],
    "top_label": "None",
    "top_conf": 0.0,
    "object_count": 0
}

# -----------------------------
# Background Processing Thread
# -----------------------------
def process_camera():
    global current_frame_bytes, latest_data
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1) # Wait a bit if camera drops
            continue

        # Run YOLO
        results = model(frame, imgsz=416, conf=0.25, verbose=False)
        r = results[0]
        annotated_frame = r.plot()

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time

        detected_objects = []
        top_label = "None"
        top_conf = 0.0

        if r.boxes is not None and len(r.boxes) > 0:
            class_ids = r.boxes.cls.tolist()
            confidences = r.boxes.conf.tolist()
            names = r.names

            for cls_id, conf in zip(class_ids, confidences):
                label = names[int(cls_id)]
                detected_objects.append({
                    "label": label,
                    "confidence": round(float(conf), 3)
                })

                if conf > top_conf:
                    top_conf = float(conf)
                    top_label = label

        # Draw overlays
        cv2.putText(annotated_frame, f"Top: {top_label} ({top_conf:.2f})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Encode frame ONCE per cycle, not per user
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if ret:
            with lock:
                current_frame_bytes = buffer.tobytes()
                latest_data = {
                    "fps": round(fps, 2),
                    "objects": detected_objects,
                    "top_label": top_label,
                    "top_conf": round(top_conf, 3),
                    "object_count": len(detected_objects)
                }

# Start the background thread
thread = threading.Thread(target=process_camera, daemon=True)
thread.start()

# -----------------------------
# Web Routes
# -----------------------------
def generate_frames():
    """Yields the latest pre-processed frame to connected web clients."""
    while True:
        with lock:
            frame_to_send = current_frame_bytes
        
        if frame_to_send is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_to_send + b"\r\n")
        
        # Sleep slightly to prevent maxing out web server CPU
        time.sleep(0.03) 

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def detections():
    with lock:
        return jsonify(latest_data)

if __name__ == "__main__":
    try:
        # threaded=True allows Flask to handle multiple web clients at once
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True) 
    finally:
        cap.release()