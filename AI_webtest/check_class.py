from ultralytics import YOLO
import cv2

MODEL_PATH = "best_test.pt"
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break

    results = model.predict(frame, conf=0.25, imgsz=640, verbose=False)

    annotated = results[0].plot()
    cv2.imshow("YOLO26 Webcam Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()