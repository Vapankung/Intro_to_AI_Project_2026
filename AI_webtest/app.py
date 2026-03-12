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
SIMULATION_MODE = False
DETECTION_INTERVAL_SEC = 0.05
CONFIDENCE_THRESHOLD = 0.45
MODEL_PATH = "best_test.pt"
IMGSZ = 416

# Camera config
CAMERA_CANDIDATES = [
    (0, cv2.CAP_DSHOW),
    (0, cv2.CAP_MSMF),
    (0, None),
    (1, cv2.CAP_DSHOW),
    (1, cv2.CAP_MSMF),
    (1, None),
    (2, cv2.CAP_DSHOW),
    (2, cv2.CAP_MSMF),
    (2, None),
]
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_WARMUP_SEC = 1.0
CAMERA_READ_RETRIES = 15
CAMERA_RETRY_DELAY_SEC = 0.1
MAX_CONSECUTIVE_READ_FAILURES = 30

# -----------------------------
# YOLO object -> waste bin
# -----------------------------
YOLO_MAPPING = {

    # recycle
    "Glass Bottle": "recycle",
    "Plastic Bottle": "recycle",
    "Can": "recycle",
    "Paper": "recycle",

    # hazardous
    "Electronic": "hazardous",
    "Syringe": "hazardous",
    "Light Bulb": "hazardous",
    "Battery": "hazardous",

    # wet / organic
    "Food left over": "wet",
    "Food on a plate": "wet",
    "Plant": "wet",
    "Feces": "wet",

    # general waste
    "Cloth": "general",
    "Clothe": "general",
    "Foam box": "general",
    "Plastic bag": "general",
    "Snack bag": "general"
}

# -----------------------------
# Globals
# -----------------------------
model = None
cap = None

latest_jpeg = None
frame_lock = threading.Lock()

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


def backend_name(backend):
    if backend is None:
        return "AUTO"
    if backend == cv2.CAP_DSHOW:
        return "CAP_DSHOW"
    if backend == cv2.CAP_MSMF:
        return "CAP_MSMF"
    return str(backend)


def release_camera(camera):
    try:
        if camera is not None:
            camera.release()
    except Exception as e:
        print(f"[CAMERA] Release error: {e}", flush=True)


def try_read_valid_frame(camera, retries=CAMERA_READ_RETRIES, delay=CAMERA_RETRY_DELAY_SEC):
    """
    Try reading several times because many webcams return bad/empty frames at startup.
    """
    for attempt in range(1, retries + 1):
        ok, frame = camera.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
        print(f"[CAMERA] Read attempt {attempt}/{retries} failed", flush=True)
        time.sleep(delay)
    return False, None


def open_camera():
    """
    Open a webcam using several index/backend combinations.
    """
    for idx, backend in CAMERA_CANDIDATES:
        try:
            print(f"[CAMERA] Trying index={idx}, backend={backend_name(backend)}", flush=True)

            if backend is None:
                test_cap = cv2.VideoCapture(idx)
            else:
                test_cap = cv2.VideoCapture(idx, backend)

            if not test_cap.isOpened():
                print(f"[CAMERA] isOpened() failed for index={idx}, backend={backend_name(backend)}", flush=True)
                release_camera(test_cap)
                continue

            # Set resolution early
            test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            # Warm up camera
            time.sleep(CAMERA_WARMUP_SEC)

            ok, frame = try_read_valid_frame(test_cap)
            if ok:
                h, w = frame.shape[:2]
                print(
                    f"[CAMERA] Opened successfully index={idx}, backend={backend_name(backend)}, frame={w}x{h}",
                    flush=True
                )
                return test_cap

            print(f"[CAMERA] Opened but no valid frame for index={idx}, backend={backend_name(backend)}", flush=True)
            release_camera(test_cap)

        except Exception as e:
            print(f"[CAMERA] Exception for index={idx}, backend={backend_name(backend)}: {e}", flush=True)

    return None


def reopen_camera():
    """
    Release and reopen camera if it dies during runtime.
    """
    global cap
    print("[CAMERA] Attempting to reopen camera...", flush=True)
    release_camera(cap)
    cap = open_camera()

    if cap is None:
        print("[CAMERA] Reopen failed", flush=True)
        socketio.emit("system_status", {
            "ok": False,
            "message": "Camera Error"
        })
        return False

    print("[CAMERA] Reopen successful", flush=True)
    socketio.emit("system_status", {
        "ok": True,
        "message": "YOLO Tracker Live"
    })
    return True


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
    Use YOLO built-in tracking to identify objects and their unique IDs.
    Returns the annotated frame and a list of new detections to broadcast.
    """
    global processed_track_ids, model

    results = model.track(
        frame,
        imgsz=IMGSZ,
        conf=CONFIDENCE_THRESHOLD,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    r = results[0]

    annotated = r.plot()
    new_detections = []

    if r.boxes is None or r.boxes.id is None:
        return annotated, new_detections

    class_ids = r.boxes.cls.tolist()
    confidences = r.boxes.conf.tolist()
    track_ids = r.boxes.id.tolist()
    names = r.names

    for i in range(len(track_ids)):
        t_id = int(track_ids[i])

        if t_id not in processed_track_ids:
            processed_track_ids.add(t_id)

            if len(processed_track_ids) > 10000:
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
        cv2.putText(
            blank,
            "Prototype Mode",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )
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

        socketio.emit("system_status", {
            "ok": True,
            "message": "YOLO Tracker Live"
        })

    last_detection_run = 0.0
    consecutive_read_failures = 0

    while True:
        socketio.sleep(0.01)

        if SIMULATION_MODE:
            now = time.time()
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
        if not ok or frame is None or frame.size == 0:
            consecutive_read_failures += 1
            if consecutive_read_failures % 5 == 0:
                print(f"[CAMERA] Runtime read failure count = {consecutive_read_failures}", flush=True)

            if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                print("[CAMERA] Too many read failures, reopening camera...", flush=True)
                success = reopen_camera()
                consecutive_read_failures = 0
                if not success:
                    socketio.sleep(1.0)
            continue

        consecutive_read_failures = 0

        now = time.time()
        run_detection = (now - last_detection_run >= DETECTION_INTERVAL_SEC)

        if run_detection:
            last_detection_run = now
            try:
                annotated, new_items = process_tracking_from_frame(frame)
                update_latest_frame(annotated)

                for item in new_items:
                    print(
                        f"[TRACK ID: {item['track_id']}] "
                        f"{item['object_name']} ({item['confidence']:.2f}) -> {item['bin_type']}",
                        flush=True
                    )
                    socketio.emit("trash_detected", {
                        "object_name": item["object_name"],
                        "bin_type": item["bin_type"],
                        "confidence": item["confidence"]
                    })

            except Exception as e:
                print(f"[YOLO] Tracking error: {e}", flush=True)


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
    