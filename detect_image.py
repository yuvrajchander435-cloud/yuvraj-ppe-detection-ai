from ultralytics import YOLO

model = YOLO("runs/detect/ppe_detection_v25/weights/best.pt")

results = model.predict(
    source="C:/PPE Detection/dataset/train/images",
    conf=0.25,
    imgsz=640,
    device="cpu",
    save=True
)

print("Detection complete")
