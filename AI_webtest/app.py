from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import random
import time
import threading

# -----------------------------
# Flask + Socket.IO setup
# -----------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "waste-sorter-prototype"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -----------------------------
# Config
# -----------------------------
PORT = 50000
SIMULATION_MODE = False          # True = fake detections, False = real webcam + YOLO Tracker
DETECTION_INTERVAL_SEC = 0.05    # Run tracker fast to maintain ID continuity
CONFIDENCE_THRESHOLD = 0.45
MODEL_PATH = "yolo26n.pt"        # Make sure you have your specific model here
IMGSZ = 416

# -----------------------------
# COCO object -> waste bin
# -----------------------------
YOLO_MAPPING = {
    # recycle
    "bottle": "recycle",
    "wine glass": "recycle",
    "book": "recycle",

    # wet
    "banana": "wet",
    "apple": "wet",
    "orange": "wet",
    "broccoli": "wet",
    "carrot": "wet",
    "sandwich": "wet",
    "pizza": "wet",
    "donut": "wet",
    "cake": "wet",

    # hazardous
    "cell phone": "hazardous",
    "laptop": "hazardous",
    "mouse": "hazardous",
    "keyboard": "hazardous",
    "remote": "hazardous",
    "tv": "hazardous",

    # general
    "cup": "general",
    "bowl": "general",
    "fork": "general",
    "knife": "general",
    "spoon": "general",
    "scissors": "general",
    "toothbrush": "general",
    "teddy bear": "general",
}

# -----------------------------
# Globals
# -----------------------------
model = None
cap = None

latest_jpeg = None
frame_lock = threading.Lock()

# TRACKING STATE
# Keep a set of all unique object IDs the tracker has seen and processed
processed_track_ids = set()

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return {
        "status": "ok",
        "mode": "simulation" if SIMULATION_MODE else "real_tracking",
        "model": MODEL_PATH
    }

@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# -----------------------------
# Socket events
# -----------------------------
@socketio.on("connect")
def on_connect():
    print("[SocketIO] Client connected", flush=True)
    socketio.emit("system_status", {
        "ok": True,
        "message": "Prototype Mode" if SIMULATION_MODE else "YOLO Tracker Live"
    })

@socketio.on("disconnect")
def on_disconnect():
    print("[SocketIO] Client disconnected", flush=True)

# -----------------------------
# Helpers
# -----------------------------
def get_simulated_detection():
    detected_class = random.choice(list(YOLO_MAPPING.keys()))
    confidence = round(random.uniform(0.70, 0.98), 2)
    return detected_class, confidence

def open_camera():
    candidates = [
        (0, None),
        (0, cv2.CAP_MSMF),
        (0, cv2.CAP_DSHOW),
        (1, None),
        (1, cv2.CAP_MSMF),
        (1, cv2.CAP_DSHOW),
    ]

    for idx, backend in candidates:
        try:
            if backend is None:
                test_cap = cv2.VideoCapture(idx)
            else:
                test_cap = cv2.VideoCapture(idx, backend)

            if not test_cap.isOpened():
                test_cap.release()
                continue

            ok, frame = test_cap.read()
            if ok and frame is not None and frame.size > 0 and frame.mean() > 1:
                print(f"[CAMERA] Opened camera index={idx}, backend={backend}", flush=True)
                return test_cap

            test_cap.release()
        except Exception:
            pass

    return None

def update_latest_frame(frame):
    global latest_jpeg

    ok, buffer = cv2.imencode(".jpg", frame)
    if ok:
        with frame_lock:
            latest_jpeg = buffer.tobytes()

def mjpeg_generator():
    global latest_jpeg

    while True:
        socketio.sleep(0.05)

        with frame_lock:
            frame = latest_jpeg

        if frame is None:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

def process_tracking_from_frame(frame):
    """
    Uses YOLO built-in tracking (ByteTrack) to identify objects and their unique IDs.
    Returns the annotated frame and a list of new detections to broadcast.
    """
    global processed_track_ids
    
    # We use .track() instead of model()
    # persist=True tells the model to keep tracking IDs across consecutive frames
    # tracker="bytetrack.yaml" is great for mechanical/conveyor movement
    results = model.track(frame, imgsz=IMGSZ, conf=CONFIDENCE_THRESHOLD, persist=True, tracker="bytetrack.yaml", verbose=False)
    r = results[0]

    annotated = r.plot()
    new_detections = []

    # If nothing is detected or tracking IDs haven't initialized yet
    if r.boxes is None or r.boxes.id is None:
        return annotated, new_detections

    class_ids = r.boxes.cls.tolist()
    confidences = r.boxes.conf.tolist()
    track_ids = r.boxes.id.tolist()
    names = r.names

    for i in range(len(track_ids)):
        t_id = int(track_ids[i])
        
        # If this is a brand new object the tracker hasn't processed yet
        if t_id not in processed_track_ids:
            processed_track_ids.add(t_id)
            
            # To prevent memory from growing infinitely over weeks of uptime
            if len(processed_track_ids) > 10000:
                # Remove random elements or just clear it (clearing is fine for prototypes)
                processed_track_ids.clear()
                processed_track_ids.add(t_id)

            label = names[int(class_ids[i])]
            conf = float(confidences[i])
            bin_type = YOLO_MAPPING.get(label, "general")

            new_detections.append({
                "object_name": label,
                "bin_type": bin_type,
                "confidence": conf,
                "track_id": t_id
            })

    return annotated, new_detections

# -----------------------------
# Background loop
# -----------------------------
def yolo_detection_loop():
    global model, cap

    mode_name = "SIMULATION" if SIMULATION_MODE else "REAL YOLO TRACKER"
    print(f"--- Detection Loop Started ({mode_name}) ---", flush=True)

    if SIMULATION_MODE:
        blank = 255 * (cv2.UMat(180, 240, cv2.CV_8UC3).get())
        cv2.putText(blank, "Prototype Mode", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        update_latest_frame(blank)
    else:
        print("[YOLO] Loading model...", flush=True)
        model = YOLO(MODEL_PATH)
        print(f"[YOLO] Model loaded: {MODEL_PATH}", flush=True)

        cap = open_camera()
        if cap is None:
            print("[ERROR] Cannot open webcam.", flush=True)
            socketio.emit("system_status", {
                "ok": False,
                "message": "Camera Error"
            })
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        socketio.emit("system_status", {
            "ok": True,
            "message": "YOLO Tracker Live"
        })

    last_detection_run = 0.0

    while True:
        # Reduced sleep to give the tracker more frames per second for smooth ID mapping
        socketio.sleep(0.01) 

        if SIMULATION_MODE:
            now = time.time()
            # Simulation can still use the 1-second interval
            if now - last_detection_run >= 1.0:
                last_detection_run = now
                detected_class, confidence = get_simulated_detection()
                bin_type = YOLO_MAPPING.get(detected_class, "general")

                payload = {
                    "object_name": detected_class,
                    "bin_type": bin_type,
                    "confidence": confidence,
                }

                print(f"[SIM DETECT] {detected_class} ({confidence:.2f}) -> {bin_type}", flush=True)
                socketio.emit("trash_detected", payload)
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        if frame.mean() < 2:
            continue

        now = time.time()
        run_detection = (now - last_detection_run >= DETECTION_INTERVAL_SEC)

        if run_detection:
            last_detection_run = now
            annotated, new_items = process_tracking_from_frame(frame)
            update_latest_frame(annotated)

            for item in new_items:
                print(f"[TRACK ID: {item['track_id']}] {item['object_name']} ({item['confidence']:.2f}) -> {item['bin_type']}", flush=True)
                socketio.emit("trash_detected", {
                    "object_name": item["object_name"],
                    "bin_type": item["bin_type"],
                    "confidence": item["confidence"]
                })


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("APP FILE IS RUNNING", flush=True)
    print(f"Starting server on http://localhost:{PORT}", flush=True)

    socketio.start_background_task(yolo_detection_loop)

    socketio.run(
        app,
        host="0.0.0.0",
        port=PORT,
        allow_unsafe_werkzeug=True,
        use_reloader=False
    )