from flask import Flask, render_template
from flask_socketio import SocketIO
import random

# -----------------------------
# Flask + Socket.IO setup
# -----------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "waste-sorter-prototype"

# async_mode='threading' keeps setup simple for prototypes
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -----------------------------
# Config
# -----------------------------
PORT = 50000
SIMULATION_MODE = True
DETECTION_INTERVAL_SEC = 2.5
CONFIDENCE_THRESHOLD = 0.5

# Map YOLO classes -> bin types used by HTML
YOLO_MAPPING = {
    "glass_bottle": "recycle",
    "can": "recycle",
    "paper": "recycle",
    "plastic_bottle": "recycle",
    "apple_core": "wet",
    "banana_peel": "wet",
    "battery": "hazardous",
    "electronic": "hazardous",
    "light_bulb": "hazardous",
    "syring": "hazardous",
    "tissue": "general",
}

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return {"status": "ok", "mode": "simulation" if SIMULATION_MODE else "real"}


# -----------------------------
# Socket events
# -----------------------------
@socketio.on("connect")
def on_connect():
    print("[SocketIO] Client connected", flush=True)

@socketio.on("disconnect")
def on_disconnect():
    print("[SocketIO] Client disconnected", flush=True)


# -----------------------------
# Detection loop (simulated)
# -----------------------------
def get_simulated_detection():
    detected_class = random.choice(list(YOLO_MAPPING.keys()))
    confidence = round(random.uniform(0.70, 0.98), 2)
    return detected_class, confidence


def yolo_detection_loop():
    """
    Prototype mode:
    - Simulates YOLO detections every few seconds
    Later:
    - Replace the simulation block with actual camera + YOLO inference
    """
    mode_name = "SIMULATION" if SIMULATION_MODE else "REAL YOLO"
    print(f"--- Detection Loop Started ({mode_name}) ---", flush=True)

    while True:
        # IMPORTANT: use socketio.sleep() in SocketIO background tasks
        socketio.sleep(DETECTION_INTERVAL_SEC)

        if SIMULATION_MODE:
            detected_class, confidence = get_simulated_detection()
        else:
            # TODO: replace with real YOLO inference
            # ret, frame = cap.read()
            # results = model(frame)
            # parse results -> detected_class, confidence
            continue

        if confidence <= CONFIDENCE_THRESHOLD:
            continue

        bin_type = YOLO_MAPPING.get(detected_class, "general")

        payload = {
            "object_name": detected_class,
            "bin_type": bin_type,
            "confidence": confidence,  # 0.0 - 1.0
        }

        print(f"[DETECT] {detected_class} ({confidence}) -> {bin_type}", flush=True)
        socketio.emit("trash_detected", payload)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("APP FILE IS RUNNING", flush=True)
    print(f"Starting server on http://localhost:{PORT}", flush=True)

    # Start background simulated YOLO loop
    socketio.start_background_task(yolo_detection_loop)

    # use_reloader=False prevents duplicate background thread in some setups
    socketio.run(
        app,
        host="0.0.0.0",
        port=PORT,
        allow_unsafe_werkzeug=True,
        use_reloader=False
    )