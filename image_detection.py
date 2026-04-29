import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# ------------------ LOAD MODEL ------------------
model = YOLO("runs/detect/ppe_detection_v25/weights/best.pt")

# ------------------ SAVE PATH ------------------
save_path = r"C:\PPE Detection\violations"
os.makedirs(save_path, exist_ok=True)

# ------------------ LOAD IMAGE ------------------
image_path = "test.jpg"   # change your image path
frame = cv2.imread(image_path)

# ------------------ DETECTION ------------------
results = model(frame, conf=0.25)
annotated_frame = results[0].plot()

# ------------------ CHECK VIOLATIONS ------------------
boxes = results[0].boxes
violation_detected = False
violation_labels = []

if boxes is not None:
    classes = boxes.cls.tolist()

    for cls in classes:
        cls = int(cls)

        if cls == 1:
            violation_detected = True
            violation_labels.append("NO HELMET")

        elif cls == 3:
            violation_detected = True
            violation_labels.append("NO VEST")

        elif cls == 5:
            violation_detected = True
            violation_labels.append("NO MASK")

# ------------------ ALERT + SAVE ------------------
if violation_detected:
    alert_text = " | ".join(set(violation_labels))

    cv2.putText(
        annotated_frame,
        f"ALERT: {alert_text}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    cv2.imwrite(os.path.join(save_path, filename), annotated_frame)

# ------------------ SHOW OUTPUT ------------------
cv2.imshow("Image Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()