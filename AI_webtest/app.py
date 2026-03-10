from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import time

app = Flask(__name__)

# Use COCO-pretrained model for demo
model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# COCO label -> waste category
WASTE_MAPPING = {
    # Recyclable
    "bottle": "Recyclable",
    "wine glass": "Recyclable",
    "book": "Recyclable",

    # Wet / food waste
    "banana": "Wet",
    "apple": "Wet",
    "orange": "Wet",
    "broccoli": "Wet",
    "carrot": "Wet",
    "sandwich": "Wet",
    "pizza": "Wet",
    "donut": "Wet",
    "cake": "Wet",

    # Hazardous / e-waste demo
    "cell phone": "Hazardous",
    "laptop": "Hazardous",
    "mouse": "Hazardous",
    "keyboard": "Hazardous",
    "remote": "Hazardous",
    "hair drier": "Hazardous",

    # General
    "cup": "General",
    "bowl": "General",
    "fork": "General",
    "knife": "General",
    "spoon": "General",
    "toothbrush": "General",
    "scissors": "General",
    "teddy bear": "General"
}

latest_data = {
    "fps": 0.0,
    "top_label": "None",
    "top_conf": 0.0,
    "waste_type": "Unknown",
    "object_count": 0,
    "objects": []
}

prev_time = time.time()

def classify_waste(label: str) -> str:
    return WASTE_MAPPING.get(label.lower(), "Unknown")

def generate_frames():
    global prev_time, latest_data

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=416, conf=0.25, verbose=False)
        r = results[0]
        annotated = r.plot()

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time

        detected_objects = []
        top_label = "None"
        top_conf = 0.0
        waste_type = "Unknown"

        if r.boxes is not None and len(r.boxes) > 0:
            class_ids = r.boxes.cls.tolist()
            confidences = r.boxes.conf.tolist()
            names = r.names

            for cls_id, conf in zip(class_ids, confidences):
                label = names[int(cls_id)]
                waste_class = classify_waste(label)

                detected_objects.append({
                    "label": label,
                    "confidence": round(float(conf), 3),
                    "waste_type": waste_class
                })

                if conf > top_conf:
                    top_conf = float(conf)
                    top_label = label
                    waste_type = waste_class

        latest_data = {
            "fps": round(fps, 2),
            "top_label": top_label,
            "top_conf": round(top_conf, 3),
            "waste_type": waste_type,
            "object_count": len(detected_objects),
            "objects": detected_objects
        }

        cv2.putText(annotated, f"Object: {top_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Waste: {waste_type}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def detections():
    return jsonify(latest_data)

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        cap.release()